"""Normalized probability-report densities, distinct from classifier posteriors.

For a K-way report q and state j: Dir(q; 1 + k e_j).
Undefined axes are marginalized with a uniform categorical nuisance prior.
Reports are conditionally independent given listener state: an explicit,
unfitted engineering assumption, not a claim about human mental processes.
"""
import math

FAMILY='hierarchical_dirichlet_report'
REPORTS=('perception','resolved','neutral','positive','negative','mixed')


def report(perception, obs, argmax=False):
    if obs is None:return [perception]+[None]*5
    u=obs['interpretability']['resolved']
    a=[obs['appraisal'][k] for k in REPORTS[2:]]
    if argmax:
        u=float(u>.5)
        winner=max(range(4),key=a.__getitem__);a=[float(i==winner) for i in range(4)]
    return [perception,u,*a]


def axis_logs(q, concentration):
    if q is None:return None
    # The sensor can round probabilities to zero. Fixed, recorded smoothing
    # keeps log densities finite and treats every category symmetrically.
    q=[max(1e-6,x) for x in q];total=sum(q);q=[x/total for x in q]
    base=math.lgamma(len(q)+concentration)-math.lgamma(1+concentration)
    return [base+concentration*math.log(x) for x in q]


def mixture(logs):
    if logs is None:return 0.
    peak=max(logs)
    return peak+math.log(sum(math.exp(x-peak) for x in logs)/len(logs))


def report_logs(values,model):
    r,u,*a=values;k=model['concentration']
    lp=axis_logs([1-r,r],k['perception']) if r is not None else None
    lu=axis_logs([1-u,u],k['interpretability']) if u is not None else None
    le=axis_logs(a,k['appraisal']) if all(x is not None for x in a) else None
    p0,p1=lp if lp is not None else (0.,0.)
    u0,u1=lu if lu is not None else (0.,0.)
    return (p0+mixture(lu)+mixture(le),p1+u0+mixture(le),
            *(p1+u1+(le[i] if le is not None else 0.) for i in range(4)))
