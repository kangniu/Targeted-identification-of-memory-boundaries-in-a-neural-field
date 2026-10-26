"""Fixed, matched-budget tests separating parameter location and state excitation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import numpy as np
from scipy.optimize import brentq
from model import RingField, TRUE_PARAMETERS, learn_parameters
from rate_design import make_design

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'boundary_followup'
GROUPS=['ordinary_14','critical_half','far_boundary','near_boundary',
        'critical_shift_low','critical_shift_high']


def safe_json(x):
    if isinstance(x,dict): return {str(k):safe_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [safe_json(v) for v in x]
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.bool_): return bool(x)
    return x


def write_json(path,x):
    path.write_text(json.dumps(safe_json(x),indent=2,allow_nan=False),encoding='utf-8')


def make_followup_design(field):
    base=make_design(field)
    t=base[0]['t'];p=TRUE_PARAMETERS
    aref=field.equilibria(.55,p)[1]
    def path(mu,a0):
        a=field.trajectory(a0,mu,p,t)
        return dict(mu=float(mu),a0=float(a0),t=t,a=a,g=field.g(a,mu,p))
    def boundary(mu,bias=0.):
        au,ast=field.equilibria(mu,p);gap=ast-au
        return [path(mu,au+(bias+sign*.05)*gap) for sign in [-1,1]]
    def critical(shift=0.):
        return [path(mu+shift,aref/np.sqrt(2)) for mu in [.72,.75]]
    extra={'ordinary_14':base[12:14], 'critical_half':critical(),
           'far_boundary':boundary(.45), 'near_boundary':boundary(.735),
           'critical_shift_low':critical(-.01),'critical_shift_high':critical(.01)}
    return base[:12],extra


def equilibria_cached(field,mu,p,fold):
    af,mf=fold
    g=lambda a: float(field.g(a,mu,p))
    if mu>=mf or float(field.terms(0.,mu,p)[1])>=0 or g(af)<=0:
        raise ValueError('Expected bump/quiet bistability absent')
    return np.array([brentq(g,1e-7,af,xtol=1e-12),
                     brentq(g,af,max(p[0],2*af),xtol=1e-12)])


def reference_tests(field):
    p=TRUE_PARAMETERS
    bm=np.linspace(.30,.74,45)
    protocols=[(.45,2.),(.60,1.),(.60,2.),(.60,4.),(.70,2.)]
    trajectories=[]
    for mu in [.325,.425,.525]:
        for factor in [.85,1.15]:
            t=np.linspace(0,6,31);a0=field.equilibria(mu,p)[1]*factor
            trajectories.append(dict(mu=mu,a0=a0,t=t,a=field.trajectory(a0,mu,p,t)))
    return dict(parameters=p,fold=field.fold(p),branch_mu=bm,
                branch=np.array([field.equilibria(mu,p) for mu in bm]),
                protocols=protocols,
                pulse=np.array([field.pulse_threshold(mu,T,p) for mu,T in protocols]),
                common_initial=field.equilibria(.60,p)[1],trajectories=trajectories)


def fit_and_evaluate(seed,name,states,mus,target,ref):
    field=RingField(256)
    r=dict(seed=seed,group=name,fit_status='failed',fold_status='not_evaluated',
           parameters=None,fit=None,branch_missing_count=len(ref['branch_mu']),
           branch_status=['not_evaluated']*len(ref['branch_mu']),
           branch=[[None,None]]*len(ref['branch_mu']),
           pulse_status=['not_evaluated']*len(ref['protocols']),
           pulse_thresholds=[None]*len(ref['protocols']),pulse_relative_errors=[None]*len(ref['protocols']))
    try:
        p,info=learn_parameters(field,states,mus,target)
        r.update(parameters=p,fit=info,fit_status='success' if info['success'] else 'unconverged',
                 training_rmse=np.sqrt(np.mean((field.g(states,mus,p)-target)**2)))
    except (ValueError,RuntimeError,FloatingPointError) as error:
        r['fit_error']=str(error);return safe_json(r)
    try:
        fold=field.fold(p)
        r.update(fold=fold,fold_status='success',fold_absolute_error=abs(fold[1]-ref['fold'][1]))
    except (ValueError,RuntimeError,FloatingPointError) as error:
        r.update(fold_status='failed',fold_error=str(error));return safe_json(r)
    branch=[];status=[]
    for mu in ref['branch_mu']:
        try: branch.append(equilibria_cached(field,mu,p,fold));status.append('success')
        except (ValueError,RuntimeError): branch.append([np.nan,np.nan]);status.append('bistability_absent')
    branch=np.asarray(branch);valid=np.isfinite(branch).all(axis=1)
    r.update(branch=branch,branch_status=status,branch_missing_count=np.sum(~valid))
    for j,prefix in [(0,'saddle'),(1,'stable')]:
        delta=branch[valid,j]-ref['branch'][valid,j]
        r[prefix+'_branch_relative_rmse']=np.sqrt(np.mean((delta/ref['branch'][valid,j])**2)) if np.any(valid) else np.nan
        r[prefix+'_branch_absolute_rmse']=np.sqrt(np.mean(delta**2)) if np.any(valid) else np.nan
    for j,(mu,T) in enumerate(ref['protocols']):
        try:
            pc=field.pulse_threshold(mu,T,p)
            r['pulse_status'][j]='success' if np.isfinite(pc) else 'unreachable'
            r['pulse_thresholds'][j]=pc
            r['pulse_relative_errors'][j]=abs(pc/ref['pulse'][j]-1) if np.isfinite(pc) else None
        except (ValueError,RuntimeError,FloatingPointError) as error:
            r['pulse_status'][j]='failed: '+str(error)
    try:
        pc=field.pulse_threshold(.60,2.,p,a0=ref['common_initial'])
        r.update(common_initial_pulse=pc,common_initial_pulse_relative_error=abs(pc/ref['pulse'][2]-1),
                 common_initial_status='success' if np.isfinite(pc) else 'unreachable')
    except (ValueError,RuntimeError,FloatingPointError) as error:
        r.update(common_initial_status='failed',common_initial_error=str(error))
    errors=[]
    try:
        for q in ref['trajectories']:
            learned=field.trajectory(q['a0'],q['mu'],p,q['t'])
            errors.extend(((learned-q['a'])/q['a']).tolist())
        r.update(trajectory_status='success',trajectory_relative_rmse=np.sqrt(np.mean(np.square(errors))))
    except (ValueError,RuntimeError,FloatingPointError): r['trajectory_status']='failed'
    return safe_json(r)


def summarize(records):
    summary={}
    keys=['trajectory_relative_rmse','stable_branch_relative_rmse','saddle_branch_relative_rmse',
          'stable_branch_absolute_rmse','saddle_branch_absolute_rmse',
          'fold_absolute_error','common_initial_pulse_relative_error']
    for name in GROUPS:
        rr=[r for r in records if r['group']==name]
        summary[name]={}
        for key in keys:
            values=np.array([r.get(key) for r in rr],dtype=float);finite=values[np.isfinite(values)]
            summary[name][key]=dict(median=np.median(finite) if len(finite) else None,
                                    q10=np.quantile(finite,.1) if len(finite) else None,
                                    q90=np.quantile(finite,.9) if len(finite) else None,
                                    failed_count=len(rr)-len(finite))
        pe=np.array([r['pulse_relative_errors'] for r in rr],dtype=float)
        summary[name]['pulse_errors']=[dict(median=np.nanmedian(pe[:,j]),
                                             q10=np.nanquantile(pe[:,j],.1),q90=np.nanquantile(pe[:,j],.9),
                                             failed_count=np.sum(~np.isfinite(pe[:,j]))) for j in range(5)]
        summary[name]['quality']=dict(fit_failures=sum(r['fit_status']!='success' for r in rr),
                 fold_failures=sum(r['fold_status']!='success' for r in rr),
                 branch_missing_repeats=sum(r['branch_missing_count']>0 for r in rr),
                 branch_missing_points=sum(r['branch_missing_count'] for r in rr),
                 fit_bound_hits=sum(any(r['fit']['active_bounds']) for r in rr if r['fit']),
                 pulse_failure_counts=[sum(r['pulse_status'][j]!='success' for r in rr) for j in range(5)])
    return safe_json(summary)


def main(repeats=30,workers=2):
    OUT.mkdir(exist_ok=True)
    field=RingField(512);core,extra=make_followup_design(field);ref=reference_tests(field)
    design={name:dict(paths=[dict(mu=q['mu'],a0=q['a0'],t=q['t'],amplitude=q['a'],rate=q['g']) for q in extra[name]]) for name in GROUPS}
    protocol=dict(repeats=repeats,seeds=list(range(300,300+repeats)),noise_sd=.01,
                  rng_rule='default_rng(20263000 + repeat_id)',observations_per_group=350,
                  shared_core_paths=12,added_paths=2,observations_per_path=25,
                  common_random_numbers='Each repeat uses one 14-by-25 standard-normal array. Core observations are identical; the last two noise arrays are reused across alternative added paths.',
                  interpretation='True branches are used to prepare the boundary pairs; this is a mechanism test with prior geometry, not autonomous active learning.',
                  selection='All listed locations and offsets fixed before fitting; report every group.',
                  boundary_half_width_fraction=.05,
                  design=design,reference={k:v for k,v in ref.items() if k!='trajectories'})
    write_json(OUT/'protocol.json',protocol)
    arrays={name:np.array([q['a'] for q in extra[name]]) for name in GROUPS}
    np.savez(OUT/'training_design.npz',time=core[0]['t'],core_mu=[q['mu'] for q in core],
             core_amplitude=np.array([q['a'] for q in core]),**arrays)
    print(f'Fixed design: {len(GROUPS)} groups, 350 observations each;',repeats,'noise repeats.',flush=True)
    jobs=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed in range(300,300+repeats):
            noise=np.random.default_rng(20263000+seed).normal(0,.01,(14,25))
            for name in GROUPS:
                paths=core+extra[name]
                states=np.concatenate([q['a'] for q in paths])
                mus=np.concatenate([np.full(25,q['mu']) for q in paths])
                target=np.concatenate([q['g']+noise[i] for i,q in enumerate(paths)])
                jobs.append(pool.submit(fit_and_evaluate,seed,name,states,mus,target,ref))
        records=[]
        for n,future in enumerate(as_completed(jobs),1):
            records.append(future.result())
            if n%len(GROUPS)==0: print(f'{n}/{len(jobs)} fits evaluated',flush=True)
    records.sort(key=lambda r:(r['seed'],GROUPS.index(r['group'])))
    output=dict(protocol=protocol,summary=summarize(records),records=records)
    write_json(OUT/'results.json',output)
    for name,stats in output['summary'].items():
        print(name, 'saddle %',round(stats['saddle_branch_relative_rmse']['median']*100,3),
              'fold',format(stats['fold_absolute_error']['median'],'.3g'),
              'pulse %',round(stats['pulse_errors'][2]['median']*100,3),
              'missing',stats['quality']['branch_missing_repeats'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();main(args.repeats,args.workers)
