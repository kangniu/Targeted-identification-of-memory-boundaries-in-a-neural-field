"""Compare observed fluctuations with the previously saved covariance prediction."""
from pathlib import Path
import json
import csv
import numpy as np
from run_boundary_followup import write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'joint_initial'
GROUPS=['ordinary','true_far_boundary','true_near_boundary']
TAUS=[0.,.005,.02]
METRICS=['trajectory_relative_rmse','stable_branch_relative_rmse',
         'saddle_branch_relative_rmse','fold_absolute_error',
         'primary_pulse_relative_error','common_initial_pulse_relative_error',
         'initial_rmse','calibration_rmse']


def number(v):
    return np.nan if v is None else float(v)


def metric(row,key):
    return number(row['pulse_relative_errors'][2]) if key=='primary_pulse_relative_error' else number(row.get(key))


def summary(a):
    a=np.asarray(a,dtype=float);finite=a[np.isfinite(a)]
    return dict(median=float(np.median(finite)) if len(finite) else None,
        q10=float(np.quantile(finite,.1)) if len(finite) else None,
        q90=float(np.quantile(finite,.9)) if len(finite) else None,
        valid_count=len(finite),missing_count=len(a)-len(finite))


def bootstrap_sd(a,indices):
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all():
        return [None,None]
    values=np.std(a[indices],axis=1,ddof=1)
    return np.quantile(values,[.025,.975]).tolist()


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    rows=data['records'];protocol=data['protocol']
    expected={(s,g,t) for s in protocol['seeds'] for g in GROUPS for t in TAUS}
    actual=[(r['seed'],r['group'],r['tau']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise RuntimeError(f'Incomplete results {len(actual)}/{len(expected)}')
    n=len(protocol['seeds'])
    index=np.random.default_rng(2026102601).integers(0,n,size=(10000,n))
    cells=[];metrics=[];quality=[];pulses=[];degradation=[]
    bygroup={}
    ref=protocol['reference']
    for group in GROUPS:
        for tau in TAUS:
            rr=sorted([r for r in rows if r['group']==group and r['tau']==tau],key=lambda r:r['seed'])
            bygroup[group,tau]=rr
            prediction=next(p for p in protocol['predictions_saved_before_new_fits']
                            if p['group']==group and p['initial_calibration_sd']==tau)
            for target,values,truth,sd in [
                ('fold',[r.get('fold',[None,None])[1] for r in rr],ref['fold'][1],prediction['fold_sd']),
                ('primary_pulse',[r['pulse_thresholds'][2] for r in rr],ref['pulse'][2],prediction['primary_pulse_sd'])]:
                a=np.array([number(v) for v in values]);finite=a[np.isfinite(a)]
                empirical=float(np.std(finite,ddof=1)) if len(finite)>1 else None
                lo,hi=bootstrap_sd(a,index)
                cells.append(dict(group=group,tau=tau,target=target,reference=truth,
                    empirical_sd=empirical,predicted_sd=sd,
                    empirical_relative_sd=empirical/truth if empirical is not None else None,
                    predicted_relative_sd=sd/truth,sd_ratio=empirical/sd if empirical is not None else None,
                    sd_bootstrap_low=lo,sd_bootstrap_high=hi,
                    mean_signed_error=float(np.mean(finite-truth)) if len(finite) else None,
                    median_absolute_error=float(np.median(abs(finite-truth))) if len(finite) else None,
                    rms_error=float(np.sqrt(np.mean((finite-truth)**2))) if len(finite) else None,
                    valid_count=len(finite),missing_count=len(a)-len(finite)))
            for key in METRICS:
                metrics.append(dict(group=group,tau=tau,metric=key,**summary([metric(r,key) for r in rr])))
            for j in range(5):
                pulses.append(dict(group=group,tau=tau,protocol=j,
                    mu=ref['protocols'][j][0],duration=ref['protocols'][j][1],
                    **summary([number(r['pulse_relative_errors'][j]) for r in rr])))
            quality.append(dict(group=group,tau=tau,count=len(rr),
                fit_failures=sum(r['fit_status']!='success' for r in rr),
                parameter_bound_hits=sum(any(r['fit']['parameter_active_bounds']) for r in rr),
                initial_bound_hits=sum(any(r['fit']['initial_active_bounds']) for r in rr),
                candidate_fit_failures=sum(not s for r in rr for s in r['fit']['all_success']),
                fold_failures=sum(r['fold_status']!='success' for r in rr),
                branch_missing_repeats=sum(r['branch_missing_count']>0 for r in rr),
                branch_missing_points=sum(r['branch_missing_count'] for r in rr),
                pulse_failures=sum(s!='success' for r in rr for s in r['pulse_status']),
                common_initial_missing=sum(not np.isfinite(metric(r,'common_initial_pulse_relative_error')) for r in rr),
                evaluation_errors=sum('evaluation_error' in r for r in rr)))
    for group in GROUPS:
        for tau in [.005,.02]:
            for target in ['fold','primary_pulse']:
                def values(t):
                    return np.array([number(r.get('fold',[None,None])[1]) if target=='fold'
                                     else number(r['pulse_thresholds'][2]) for r in bygroup[group,t]])
                base,current=values(0.),values(tau)
                empirical_ratio=np.std(current,ddof=1)/np.std(base,ddof=1)
                boot=np.std(current[index],axis=1,ddof=1)/np.std(base[index],axis=1,ddof=1)
                basecell=next(c for c in cells if c['group']==group and c['tau']==0. and c['target']==target)
                cell=next(c for c in cells if c['group']==group and c['tau']==tau and c['target']==target)
                degradation.append(dict(group=group,tau=tau,target=target,
                    empirical_sd_inflation=empirical_ratio,
                    predicted_sd_inflation=cell['predicted_sd']/basecell['predicted_sd'],
                    bootstrap_low=np.quantile(boot,.025),bootstrap_high=np.quantile(boot,.975)))
    paired=[]
    for tau in TAUS:
        for compare in ['true_far_boundary','true_near_boundary']:
            for key in METRICS[:6]:
                a=np.array([metric(r,key) for r in bygroup['ordinary',tau]])
                b=np.array([metric(r,key) for r in bygroup[compare,tau]])
                valid=np.isfinite(a)&np.isfinite(b)
                paired.append(dict(tau=tau,baseline='ordinary',comparison=compare,metric=key,
                    improved_count=int(np.sum(b[valid]<a[valid])),paired_valid_count=int(np.sum(valid))))
    tables=OUT/'tables';tables.mkdir(exist_ok=True)
    output=dict(variance=cells,metrics=metrics,quality=quality,pulse=pulses,
        degradation=degradation,paired=paired,
        uncertainty='10000 paired bootstrap resamples of the 30 seeds. Percentile intervals describe finite-Monte-Carlo uncertainty, not confidence bounds for a fitted biological system.',
        selection='All groups, calibration levels and targets retained. Predictions were copied into protocol.json before fitting.')
    write_json(tables/'summary.json',output)
    for name,values in [('variance',cells),('metrics',metrics),('quality',quality),
                        ('pulse',pulses),('degradation',degradation),('paired',paired)]:
        save_csv(tables/(name+'.csv'),values)
    print('group tau target predicted_SD empirical_SD ratio')
    for c in cells:
        print(c['group'],c['tau'],c['target'],f"{c['predicted_sd']:.4g}",
              f"{c['empirical_sd']:.4g}",f"{c['sd_ratio']:.3f}")


if __name__=='__main__':main()
