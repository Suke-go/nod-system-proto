"""A single capture stream feeds independent bounded ASR and timing workers."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time


@dataclass
class ASRJob:
    utterance_id: str
    audio: object
    source_ms: int
    end_sample: int
    final: bool
    final_reason: str | None = None


class Segmenter:
    """Energy endpointing, not a calibrated speech/emotion classifier."""
    def __init__(self, threshold=.012, partial_ms=800, silence_ms=600, max_ms=12000):
        if not 0 < threshold < 1 or partial_ms < 200 or silence_ms < 100 or max_ms < partial_ms:
            raise ValueError('Invalid live segmentation settings')
        self.threshold,self.partial_ms,self.silence_ms,self.max_ms=threshold,partial_ms,silence_ms,max_ms
        self.pre=deque(maxlen=10); self.audio=[]; self.uid=0
        self.frames=0; self.silence=0; self.last_partial=0; self.active=False

    def push(self, frame, source_ms, end_sample):
        import numpy as np
        loud=float(np.sqrt(np.mean(frame.astype(np.float64)**2))) >= self.threshold
        onset=False; job=None
        if not self.active:
            if not loud:
                self.pre.append(frame.copy()); return onset,job
            self.uid+=1; self.active=True; onset=True
            self.audio=list(self.pre); self.pre.clear()
            self.frames=len(self.audio); self.silence=0; self.last_partial=0
        self.audio.append(frame.copy()); self.frames+=1
        self.silence=0 if loud else self.silence+20
        elapsed=self.frames*20
        final=self.silence >= self.silence_ms or elapsed >= self.max_ms
        reason='silence' if self.silence >= self.silence_ms else 'max_duration' if final else None
        if final or elapsed-self.last_partial >= self.partial_ms:
            job=ASRJob(f'live-{self.uid}',np.concatenate(self.audio),source_ms,end_sample,final,reason)
            self.last_partial=elapsed
        if final:
            self.audio=[]; self.frames=0; self.active=False; self.pre.clear()
        return onset,job

    def finish(self, source_ms, end_sample):
        import numpy as np
        if not self.active or not self.audio: return None
        job=ASRJob(f'live-{self.uid}',np.concatenate(self.audio),source_ms,end_sample,True,'input_end')
        self.audio=[]; self.active=False
        return job


class LiveInput:
    def __init__(self, runtime, asr, timing, *, device=None, wav=None, duration=300,
                 threshold=.012, partial_ms=800, stop_event=None):
        self.runtime,self.asr,self.timing=runtime,asr,timing
        self.device,self.wav,self.duration=device,wav,duration
        self.segmenter=Segmenter(threshold,partial_ms)
        self.stop=stop_event or asyncio.Event()
        self.pause=False; self.error=None; self.level=0.; self.asr_ms=None; self.timing_ms=None
        self.overwritten_asr=0; self.pending=None; self.wakeup=asyncio.Event()
        asr_cfg=getattr(runtime,'cfg',{}).get('asr',{})
        self.preserve_finals=asr_cfg.get('preserve_final_jobs',False)
        self.final_capacity=asr_cfg.get('final_queue_capacity',2)
        self.final_jobs=deque()
        self.timing_queue=asyncio.Queue(10)
        self.capture_queue=asyncio.Queue(100)
        self.capture_slots=threading.BoundedSemaphore(100)
        self.current_uid=None; self.origin=0; self.end_sample=0
        self.previous_adc_time=None

    def _job_event(self, job, state, **details):
        if self.preserve_finals:
            self.runtime.emit('asr_job',{'utterance_id':job.utterance_id,
                'source_ms':job.source_ms,'audio_end_sample':job.end_sample,
                'final':job.final,'final_reason':job.final_reason,'state':state,**details})

    def _onset(self, uid, source, end_sample):
        self.current_uid=uid
        if self.pending is not None:
            self._job_event(self.pending,'discarded',reason='new_utterance')
            self.pending=None
        if not self.pause:
            self.runtime.emit('asr',{'utterance_id':uid,'text':'','source_ms':source,
                'final':False,'audio_end_sample':end_sample,'audio_origin_ms':self.origin})

    def offer(self, job):
        if job is None:return
        if self.pause:
            self._job_event(job,'discarded',reason='paused');return
        if self.preserve_finals and job.final:
            # The acoustic boundary is available before the ASR/Jev round trip.
            self.runtime.emit('asr_boundary',{'utterance_id':job.utterance_id,
                'source_ms':job.source_ms,'final_reason':job.final_reason})
            if self.pending is not None and self.pending.utterance_id==job.utterance_id:
                self._job_event(self.pending,'discarded',reason='superseded_by_final')
                self.overwritten_asr+=1;self.pending=None
            if len(self.final_jobs)>=self.final_capacity:
                old=self.final_jobs.popleft()
                self._job_event(old,'discarded',reason='final_queue_overload')
            self.final_jobs.append(job)
        else:
            if self.pending is not None:
                self._job_event(self.pending,'discarded',reason='newer_partial')
                self.overwritten_asr+=1
            self.pending=job
        self._job_event(job,'queued');self.wakeup.set()

    async def _asr_worker(self, pool):
        loop=asyncio.get_running_loop()
        while True:
            await self.wakeup.wait(); self.wakeup.clear()
            if self.final_jobs:
                job=self.final_jobs.popleft()
            else:
                job,self.pending=self.pending,None
            if job is None: continue
            if self.final_jobs or self.pending is not None:self.wakeup.set()
            if self.pause or (job.utterance_id!=self.current_uid and not (self.preserve_finals and job.final)):
                self._job_event(job,'discarded',reason='paused' if self.pause else 'prior_utterance')
                continue
            started_ms=self.runtime.clock.now_ms()
            self._job_event(job,'started',queue_age_ms=started_ms-job.source_ms)
            begin=time.perf_counter()
            try:
                result=await loop.run_in_executor(pool,self.asr.infer,job.audio)
            except asyncio.CancelledError:
                self._job_event(job,'discarded',reason='worker_stopped');raise
            except Exception:
                self._job_event(job,'failed',reason='inference_error');raise
            self.asr_ms=round((time.perf_counter()-begin)*1000)
            if self.pause:
                self._job_event(job,'discarded',reason='paused');continue
            if job.utterance_id != self.current_uid:
                # Preserve the endpoint for offline diagnosis without resurrecting
                # old listener state or acting on the previous topic.
                if self.preserve_finals and job.final:
                    self._job_event(job,'archived',reason='prior_utterance',
                        text=result['text'],inference_ms=self.asr_ms)
                else:self._job_event(job,'discarded',reason='prior_utterance')
                continue
            self._job_event(job,'delivered',inference_ms=self.asr_ms)
            self.runtime.emit('asr',{'utterance_id':job.utterance_id,'text':result['text'],
                'source_ms':job.source_ms,'audio_end_sample':job.end_sample,'audio_origin_ms':self.origin,
                'final':job.final,'final_reason':job.final_reason,'inference_ms':self.asr_ms,
                'asr_started_ms':started_ms,'queue_age_ms':started_ms-job.source_ms,
                'audio_start_sample':job.end_sample-len(job.audio),
                'confidence_kind':'unavailable_stability_proxy_only'})

    async def _timing_worker(self, pool):
        loop=asyncio.get_running_loop()
        while True:
            audio,source,end_sample=await self.timing_queue.get()
            begin=time.perf_counter()
            score=await loop.run_in_executor(pool,self.timing.infer,audio)
            self.timing_ms=round((time.perf_counter()-begin)*1000)
            if self.runtime.clock.now_ms()-source > 1000:
                raise RuntimeError('Timing worker is over 1s behind; output stopped')
            if not self.pause:
                self.runtime.emit('opportunity',{'source_ms':source,'score':score,'audio_end_sample':end_sample,
                    'audio_origin_ms':self.origin,'inference_ms':self.timing_ms})

    async def _file_producer(self):
        import wave
        import numpy as np
        with wave.open(str(self.wav),'rb') as stream:
            if (stream.getframerate(),stream.getnchannels(),stream.getsampwidth()) != (16000,1,2):
                raise ValueError('Live file input requires 16kHz mono PCM16 WAV')
            while not self.stop.is_set():
                raw=stream.readframes(320)
                if not raw: break
                samples=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
                if len(samples)<320: samples=np.pad(samples,(0,320-len(samples)))
                self.end_sample+=320
                source=self.origin+self.end_sample//16
                await asyncio.sleep(max(0,(source-self.runtime.clock.now_ms())/1000))
                await self.capture_queue.put((samples,source,self.end_sample,False))
        # Endpoint silence also allows the final transcription to complete while audio time advances.
        for _ in range(100):
            if self.stop.is_set(): break
            self.end_sample+=320; source=self.origin+self.end_sample//16
            await asyncio.sleep(max(0,(source-self.runtime.clock.now_ms())/1000))
            await self.capture_queue.put((np.zeros(320,dtype=np.float32),source,self.end_sample,False))
        await self.capture_queue.put(None)

    def _callback(self, loop):
        def callback(data, frames, timing, status):
            if self.stop.is_set(): return
            adc=float(timing.inputBufferAdcTime) if timing is not None else None
            gap=(adc is not None and self.previous_adc_time is not None
                 and abs(adc-self.previous_adc_time-.020)>.1)
            if status or gap or frames != 320 or not self.capture_slots.acquire(blocking=False):
                loop.call_soon_threadsafe(self._capture_error,'Microphone overflow or discontinuity')
                return
            if not self.end_sample and adc is not None:
                age=max(0,round((float(timing.currentTime)-adc-frames/16000)*1000))
                self.origin=max(0,self.runtime.clock.now_ms()-age-20)
            self.previous_adc_time=adc
            self.end_sample+=frames
            source=self.origin+self.end_sample//16
            # Queue reservation bounds callbacks waiting on the asyncio loop as well as the queue itself.
            item=(data[:,0].copy(),source,self.end_sample,True)
            loop.call_soon_threadsafe(self.capture_queue.put_nowait,item)
        return callback

    def _capture_error(self,message):
        self.error=message; self.runtime.emit('input_health',{'healthy':False}); self.stop.set()

    async def _consume(self):
        import numpy as np
        chunks=[]
        last_source=self.origin
        last_end=0
        while not self.stop.is_set():
            try: item=await asyncio.wait_for(self.capture_queue.get(),.1)
            except TimeoutError: continue
            if item is None: break
            frame,source,end_sample,reserved=item
            if reserved: self.capture_slots.release()
            last_source=source
            last_end=end_sample
            # Windows' event-loop clock may wake several milliseconds early.
            # Never let that turn a captured frame into a future observation.
            while source > self.runtime.clock.now_ms():
                await asyncio.sleep(max(.001,(source-self.runtime.clock.now_ms())/1000))
            if self.runtime.clock.now_ms()-source > 1000:
                raise RuntimeError('Audio capture is over 1s behind; output stopped')
            self.level=float(np.sqrt(np.mean(frame.astype(np.float64)**2)))
            onset,job=self.segmenter.push(frame,source,end_sample)
            if onset:
                self._onset(f'live-{self.segmenter.uid}',source,end_sample)
            self.offer(job)
            chunks.append(frame)
            if len(chunks)==5:
                if self.timing_queue.full(): raise RuntimeError('Timing queue overflow; output stopped')
                self.timing_queue.put_nowait((np.concatenate(chunks),source,end_sample)); chunks=[]
            if source-self.origin >= self.duration*1000: break
        self.offer(self.segmenter.finish(last_source,last_end))

    async def run(self):
        import sounddevice as sd
        if not 1 <= self.duration <= 1800: raise ValueError('Live duration must be 1–1800 seconds')
        self.origin=self.runtime.clock.now_ms()
        loop=asyncio.get_running_loop()
        asr_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='nod-asr')
        timing_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='nod-timing')
        tasks=[]; stream=None
        try:
            if self.wav:
                tasks.append(asyncio.create_task(self._file_producer()))
            else:
                stream=sd.InputStream(samplerate=16000,blocksize=320,channels=1,dtype='float32',
                    device=self.device,callback=self._callback(loop))
                stream.start()
            worker_asr=asyncio.create_task(self._asr_worker(asr_pool))
            worker_timing=asyncio.create_task(self._timing_worker(timing_pool))
            consumer=asyncio.create_task(self._consume())
            tasks.extend([worker_asr,worker_timing,consumer])
            # File producer may finish normally before the consumer; workers must never finish silently.
            monitored=[worker_asr,worker_timing,consumer]
            if self.wav:
                async def producer_guard():
                    await tasks[0]
                    await asyncio.Event().wait()
                guard=asyncio.create_task(producer_guard()); tasks.append(guard); monitored.append(guard)
            done,_=await asyncio.wait(monitored,return_when=asyncio.FIRST_COMPLETED)
            for task in done: task.result()
            if consumer not in done: raise RuntimeError('A live inference worker stopped')
            if self.error: raise RuntimeError(self.error)
            if stream:
                stream.stop(); stream.close(); stream=None
            if self.stop.is_set():
                self.runtime.emit('input_health',{'healthy':False})
                return
            # Leave a bounded period for final ASR/Jev observations; no synthetic opportunities added.
            await asyncio.sleep(2)
            for task in (worker_asr,worker_timing):
                if task.done(): task.result()
        finally:
            self.runtime.emit('input_health',{'healthy':False})
            for job in list(self.final_jobs)+([self.pending] if self.pending is not None else []):
                self._job_event(job,'discarded',reason='input_stopped')
            self.final_jobs.clear();self.pending=None
            if stream: stream.stop(); stream.close()
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            asr_pool.shutdown(wait=False,cancel_futures=True)
            timing_pool.shutdown(wait=False,cancel_futures=True)
