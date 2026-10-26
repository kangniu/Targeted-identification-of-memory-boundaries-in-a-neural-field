"""Direct Monte Carlo test of calibration-related information loss."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import time
import numpy as np
from model import RingField, TRUE_PARAMETERS
from joint_initial_model import fit_joint, fit_exact
from state_observation_model import observe_trajectories
from run_state_observations import ordinary_design, evaluate
from run_boundary_followup import write_json, safe_json, reference_tests

ROOT = Path(__file__).resolve().parent
GROUPS = ['ordinary', 'true_far_boundary', 'true_near_boundary']
SIGMA = .02
TAUS = [0., .005, .02]


def fixed_designs():
    field = RingField(512)
    initial, mus, times = ordinary_design()
    designs = {}
    for group in GROUPS:
        a0, mu = initial.copy(), mus.copy()
        if group != 'ordinary':
            m = .45 if group == 'true_far_boundary' else .735
            lower, upper = field.equilibria(m, TRUE_PARAMETERS)
            a0[-2:] = lower+np.array([-.05, .05])*(upper-lower)
            mu[-2:] = m
        amplitude, _ = observe_trajectories(field, a0, mu, times, TRUE_PARAMETERS,
                              sensitivities=False, rtol=2e-11, atol=2e-13)
        designs[group] = dict(initial=a0, mu=mu, times=times, amplitude=amplitude)
    return designs


def one_fit(seed, group, tau, design, ref):
    start = time.perf_counter()
    field = RingField(256)
    # Separate streams make calibration errors independent of state errors.
    state_noise = np.random.default_rng(20265200+seed).normal(size=(14, 25))
    initial_noise = np.random.default_rng(20266300+seed).normal(size=14)
    observed = np.asarray(design['amplitude'])+SIGMA*state_noise
    calibrated = np.asarray(design['initial'])+tau*initial_noise
    if tau == 0:
        p, a0, info = fit_exact(field, design['initial'], design['mu'],
                                design['times'], observed, SIGMA)
    else:
        # True initial amplitudes and true boundary positions are not passed here.
        p, a0, info = fit_joint(field, design['mu'], design['times'], observed,
                                calibrated, SIGMA, tau)
    result = evaluate(p, info, ref)
    result.update(seed=seed, group=group, sigma=SIGMA, tau=tau,
        calibrated_initial=calibrated, estimated_initial=a0,
        initial_estimation_error=a0-np.asarray(design['initial']),
        initial_rmse=np.sqrt(np.mean((a0-np.asarray(design['initial']))**2)),
        calibration_rmse=np.sqrt(np.mean((calibrated-np.asarray(design['initial']))**2)),
        seconds=time.perf_counter()-start)
    return safe_json(result)


def main(repeats=30, workers=2, pilot=False):
    out = ROOT/('joint_initial_pilot' if pilot else 'joint_initial')
    out.mkdir(exist_ok=True)
    first_seed = 1100 if pilot else 1200
    designs = fixed_designs()
    ref = reference_tests(RingField(512))
    old_info = json.loads((ROOT/'state_observations/information.json').read_text(encoding='utf-8'))
    predictions = [r for r in old_info['records'] if r['group'] in GROUPS
                   and r['sigma']==SIGMA and r['initial_calibration_sd'] in TAUS]
    protocol = dict(groups=GROUPS, sigma=SIGMA, calibration_sd=TAUS,
        seeds=list(range(first_seed,first_seed+repeats)),
        design='Three fixed designs identical to the previous local-information calculation.',
        observations='350 amplitude measurements; tau>0 adds 14 noisy initial calibration observations. tau=0 is an ideal known-initial benchmark, not an equal-cost experiment.',
        unknowns='tau>0: 3 physical parameters and 14 independent initial amplitudes.',
        true_initial='Fixed across repetitions; unknown to the estimator for tau>0. Only the independent calibration measurement is provided.',
        noise='Independent Gaussian state and calibration errors. Same standardized arrays paired across designs and tau values.',
        rng_state='default_rng(20265200+seed)',rng_calibration='default_rng(20266300+seed)',
        boundary_reference='Boundary locations use true branches only to define the fixed mechanism experiment. No autonomous adaptive-selection claim.',
        parameter_bounds=[[3.,2.,-.15],[10.,40.,.45]],initial_bounds=[0.,4.],
        fit='Joint weighted nonlinear least squares, integrating parameter and initial-condition sensitivities. Three fixed starts; all outcomes retained.',
        targets='Branches, fold, five held-out erasure protocols. Main variance comparison uses the fold and mu0=.60,T=2 erasure amplitude.',
        interpretation='Calibration uncertainty of fixed unknown initials, not random actuation error or dynamic noise.',
        predictions_saved_before_new_fits=predictions,
        designs=designs,reference={k:v for k,v in ref.items() if k!='trajectories'})
    write_json(out/'protocol.json',protocol)
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(one_fit,seed,group,tau,designs[group],ref)
              for seed in range(first_seed,first_seed+repeats) for group in GROUPS for tau in TAUS]
        for done,future in enumerate(as_completed(jobs),1):
            row=future.result();records.append(row)
            records.sort(key=lambda r:(r['seed'],r['group'],r['tau']))
            write_json(out/'results.json',dict(protocol=protocol,records=records))
            if done%9==0 or done==len(jobs):
                print(f'{done}/{len(jobs)} fits; last {row["group"]}, tau={row["tau"]}, '
                      f'{row["seconds"]:.1f} s; success={row["fit_status"]}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();main(args.repeats,args.workers,args.pilot)
