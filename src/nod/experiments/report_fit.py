"""Fit the normalized report-density concentrations on reviewed calibration data."""
import math
from nod.belief.reports import FAMILY, REPORTS, report_logs
from nod.belief.listener import STATES


def fit_reports(pairs):
    counts={s:0 for s in STATES};rows=[]
    for label,pred in pairs:
        if pred.get('observation_family')!=FAMILY:raise ValueError('Do not mix observation families')
        x=pred['probability_report']
        if (len(x)!=6 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in x)
            or abs(sum(x[2:])-1)>1e-5):raise ValueError('Complete probability reports required')
        counts[label['state']]+=1;rows.append((STATES.index(label['state']),x))
    if any(n<8 for n in counts.values()):raise ValueError('At least 8 reviewed calibration examples per state are required')
    model={'family':FAMILY,'reports':list(REPORTS),'probability_floor':1e-6,
           'concentration':{'perception':1.,'interpretability':1.,'appraisal':1.},'status':'fitted_on_calibration'}
    # Separable log likelihood. Bounded deterministic grid search, followed by
    # local refinement; nuisance mixtures are included in every labeled state.
    for axis in model['concentration']:
        def loss(k):
            model['concentration'][axis]=k
            return -sum(report_logs(x,model)[y] for y,x in rows)
        grid=[.05*(400**(i/200)) for i in range(201)]
        best=min(range(len(grid)),key=lambda i:loss(grid[i]))
        lo,hi=grid[max(0,best-1)],grid[min(200,best+1)]
        for _ in range(50):
            a,b=lo+(hi-lo)/3,hi-(hi-lo)/3
            if loss(a)<loss(b):hi=b
            else:lo=a
        model['concentration'][axis]=(lo+hi)/2
    return model,counts
