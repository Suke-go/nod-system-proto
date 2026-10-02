"""Loopback-only observation UI. Browser commands are limited to ending this session."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import secrets
import threading


class Dashboard:
    def __init__(self, language, backend, port=8766, stop=None, rendered=None):
        self.state={'language':language,'backend':backend,'status':'starting','text':'','stable':'',
                    'raw':None,'belief':[.6,.25,.1,.05],'score':0,'motor':'IDLE',
                    'action':None,'actions':0,'reasons':[],'semantic_status':'waiting',
                    'prominence':'ACN unavailable: fixed low prominence feature','api_ms':None}
        self.payload=json.dumps(self.state).encode()
        self.token=secrets.token_urlsafe(32)
        self.stop=stop
        self.rendered=rendered
        owner=self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass

            def valid_host(self):
                return self.headers.get('Host') in (f'127.0.0.1:{owner.port}',f'localhost:{owner.port}')

            def reply(self,code,body=b'',kind='application/json'):
                self.send_response(code)
                self.send_header('Content-Type',kind+'; charset=utf-8')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")
                self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                if not self.valid_host(): return self.reply(403)
                if self.path=='/api/state': return self.reply(200,owner.payload)
                if self.path=='/':
                    page=files('nod').joinpath('resources/live.html').read_text(encoding='utf-8')
                    return self.reply(200,page.replace('__CONTROL_TOKEN__',owner.token).encode(),'text/html')
                self.reply(404)

            def do_POST(self):
                allowed=(f'http://127.0.0.1:{owner.port}',f'http://localhost:{owner.port}')
                if (not self.valid_host() or self.headers.get('Origin') not in allowed
                        or not secrets.compare_digest(self.headers.get('X-Control-Token',''),owner.token)):
                    return self.reply(403)
                if self.path=='/api/rendered' and owner.rendered:
                    try:
                        n=int(self.headers.get('Content-Length','0'))
                        if not 0<n<=256:raise ValueError('Invalid payload size')
                        data=json.loads(self.rfile.read(n))
                        if (not isinstance(data,dict) or set(data)!={'action_id'} or
                            not isinstance(data['action_id'],str) or not data['action_id'].startswith('a') or
                            not data['action_id'][1:].isdigit() or len(data['action_id'])>16):raise ValueError('Invalid action ID')
                    except (ValueError,TypeError):return self.reply(400)
                    owner.rendered(data);return self.reply(200,b'{"received":true}')
                if self.path!='/api/stop': return self.reply(404)
                if owner.stop: owner.stop()
                self.reply(200,b'{"stopping":true}')

        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True,name='nod-dashboard')

    def start(self): self.thread.start()

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def publish(self,**values):
        self.state.update(values)
        self.payload=json.dumps(self.state,ensure_ascii=False,allow_nan=False).encode('utf-8')

    def observe(self,event,records,engine):
        s=self.state
        s.update(elapsed_ms=event.at_ms,motor=engine.motor.state,score=engine.opportunity.score,
                 belief=list(engine.belief.at(event.at_ms)))
        if event.kind=='asr':
            s.update(text=engine.transcripts.current.text,stable=engine.transcripts.current.stable)
            if not s['text']: s.update(raw=None,semantic_status='waiting',api_ms=None)
        if event.kind=='semantic_result':
            s.update(raw=list(event.data['result']['probabilities']),
                     api_ms=event.at_ms-event.data['request']['dispatched_ms'])
            if 'frames' in event.data['result']:
                from nod.semantic.frames import FRAME_ORDER,marginals
                p=event.data['result']['frames']
                s.update(semantic_frames=dict(zip(FRAME_ORDER,p)),semantic_marginals=marginals(p))
        if event.kind=='semantic_result' and 'observation' in event.data['result']:
            s['sensor_observation']=event.data['result']['observation']
        if event.kind=='semantic_error': s['semantic_status']=event.data['reason']
        for record in records:
            if record['kind']=='semantic_disposition': s['semantic_status']=record['reason']
            if record['kind']=='decision_status': s['reasons']=record['reasons']
            if record['kind']=='decision_status' and 'interaction' in record:
                s.update(interaction=record['interaction'],unit_id=record['unit_id'],
                         utilities=record['utilities'],feedback_count=record['feedback_count'])
            if record['kind']=='decision_status' and 'listener' in record:
                s.update(listener=record['listener'],unit_id=record['unit_id'],
                         decision_context=record['decision_context'],utilities=record['utilities'],
                         retained_interpretations=record.get('retained_interpretations',[]))
            if record['kind']=='command':
                s['actions']+=1
                s['action']={**record['command'],'function':record['function'],
                             'trigger':record.get('trigger','vap')}
        self.publish()
