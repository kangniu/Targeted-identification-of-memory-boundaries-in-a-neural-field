"""Matched-budget learning from noisy amplitude time series."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time
import numpy as np
from model import RingField, TRUE_PARAMETERS
from state_observation_model import observe_trajectories, fit_states
from run_boundary_followup import write_json, safe_json, reference_tests, equilibria_cached

ROOT = Path(__file__).resolve().parent
GROUPS = ['ordinary', 'spread', 'adaptive_boundary']
NOISE_LEVELS = [.005, .02]


def ordinary_design():
    rng = np.random.default_rng(20261026)
    mus = np.linspace(.30, .55, 14)
    rng.shuffle(mus)
    initial = np.resize([1.4, 2.2], 14)
    return initial, mus, np.linspace(.24, 6., 25)


def evaluate(p, info, ref):
    field = RingField(256)
    r = dict(parameters=p, fit=info, fit_status='success' if info['success'] else 'unconverged',
        fold_status='not_evaluated', branch_missing_count=45,
        pulse_status=['not_evaluated']*5, pulse_relative_errors=[None]*5,
        pulse_thresholds=[None]*5)
    try:
        fold = field.fold(p)
        r.update(fold=fold, fold_status='success', fold_absolute_error=abs(fold[1]-ref['fold'][1]))
        branch, status = [], []
        for mu in ref['branch_mu']:
            try:
                branch.append(equilibria_cached(field, mu, p, fold)); status.append('success')
            except (ValueError, RuntimeError):
                branch.append([np.nan, np.nan]); status.append('bistability_absent')
        branch = np.asarray(branch)
        valid = np.isfinite(branch).all(axis=1)
        r.update(branch=branch, branch_status=status, branch_missing_count=np.sum(~valid))
        for j, label in enumerate(['saddle', 'stable']):
            r[label+'_branch_relative_rmse'] = np.sqrt(np.mean(
                ((branch[valid, j]-ref['branch'][valid, j])/ref['branch'][valid, j])**2)) if np.any(valid) else None
        for j, (mu, duration) in enumerate(ref['protocols']):
            try:
                threshold = field.pulse_threshold(mu, duration, p)
                r['pulse_status'][j] = 'success' if np.isfinite(threshold) else 'unreachable'
                r['pulse_thresholds'][j] = threshold
                r['pulse_relative_errors'][j] = abs(threshold/ref['pulse'][j]-1) if np.isfinite(threshold) else None
            except (ValueError, RuntimeError) as error:
                r['pulse_status'][j] = 'failed: '+str(error)
        try:
            threshold = field.pulse_threshold(.60, 2., p, a0=ref['common_initial'])
            r.update(common_initial_pulse=threshold,
                common_initial_pulse_relative_error=abs(threshold/ref['pulse'][2]-1))
        except (ValueError, RuntimeError):
            r['common_initial_pulse_relative_error'] = None
        err = []
        for q in ref['trajectories']:
            learned = field.trajectory(q['a0'], q['mu'], p, q['t'])
            err.extend(((learned-q['a'])/q['a']).tolist())
        r['trajectory_relative_rmse'] = np.sqrt(np.mean(np.square(err)))
    except (ValueError, RuntimeError, FloatingPointError) as error:
        r['evaluation_error'] = str(error)
    return safe_json(r)


def one_repeat(seed, sigma, ref):
    truth, learner = RingField(512), RingField(256)
    a0, mu, times = ordinary_design()
    y, _ = observe_trajectories(truth, a0, mu, times, TRUE_PARAMETERS,
                               sensitivities=False, rtol=2e-11, atol=2e-13)
    standard_noise = np.random.default_rng(20264100+seed).normal(size=(14, 25))
    noisy = y+sigma*standard_noise
    started = time.perf_counter()
    pre_p, pre_info = fit_states(learner, a0[:12], mu[:12], times, noisy[:12])
    fallback = False
    fallback_reason = None
    try:
        if not pre_info['success']:
            raise RuntimeError('Preliminary fit did not converge')
        lower, upper = learner.equilibria(.45, pre_p)
        prepared = lower+np.array([-.05, .05])*(upper-lower)
    except (ValueError, RuntimeError) as error:
        prepared = np.array([.4, 1.3])
        fallback = True; fallback_reason = str(error)
    # Reference geometry is used only after choosing the new initial conditions.
    true_lower, true_upper = truth.equilibria(.45, TRUE_PARAMETERS)
    preparation = dict(seed=seed, sigma=sigma, preliminary_parameters=pre_p,
        preliminary_fit=pre_info, initial_amplitudes=prepared,
        fallback=fallback, fallback_reason=fallback_reason,
        straddles_reference_saddle=bool(prepared[0]<true_lower<prepared[1]),
        center_error_fraction_of_true_gap=(np.mean(prepared)-true_lower)/(true_upper-true_lower))
    rows = []
    # All three final fits use the same preliminary estimate and two fixed starts.
    starts = [pre_p, [5.5, 7., .1], [6.5, 18., .2]]
    for group in GROUPS:
        initial, mus = a0.copy(), mu.copy()
        target = noisy.copy()
        if group != 'ordinary':
            initial[-2:] = [.4, 1.3] if group == 'spread' else prepared
            mus[-2:] = .45
            extra, _ = observe_trajectories(truth, initial[-2:], mus[-2:], times,
                TRUE_PARAMETERS, sensitivities=False, rtol=2e-11, atol=2e-13)
            target[-2:] = extra+sigma*standard_noise[-2:]
        p, info = fit_states(learner, initial, mus, times, target, starts)
        row = evaluate(p, info, ref)
        row.update(seed=seed, sigma=sigma, group=group, initial_amplitudes=initial.tolist(),
                   training_mu=mus.tolist())
        rows.append(row)
    preparation['seconds'] = time.perf_counter()-started
    return rows, safe_json(preparation)


def main(repeats=30, workers=2, pilot=False):
    output = ROOT/('state_observations_pilot' if pilot else 'state_observations')
    output.mkdir(exist_ok=True)
    initial, mus, times = ordinary_design()
    first_seed = 800 if pilot else 900
    ref = reference_tests(RingField(512))
    protocol = dict(observation='Independent Gaussian noise added to amplitude samples; no rates used in fitting.',
        model='Same correctly specified three-parameter smooth Amari cosine ring.',
        noise_sd=NOISE_LEVELS, groups=GROUPS, seeds=list(range(first_seed, first_seed+repeats)),
        rng='default_rng(20264100+seed), common standardized noise across designs and noise levels.',
        total_observations=350, common_observations=300, added_observations=50,
        times=times, ordinary_initial_amplitudes=initial, ordinary_mu=mus,
        initial_conditions='Known and exactly prepared. Fixed ordinary initial amplitudes 1.4 and 2.2 use no true branch.',
        fit='Fit observed states by integrating the amplitude equation and its parameter sensitivities; no numerical differentiation.',
        adaptive='Fit the first 12 trajectories; at mu=.45 prepare estimated saddle +/- 5% estimated branch gap; use fixed .4,1.3 fallback if unavailable.',
        control='Ordinary adds two upper-basin trajectories; spread adds fixed amplitudes .4,1.3 at mu=.45.',
        limitations='One fixed ordinary design; repeated observation noise, not random layouts; exact known initial states and control values; no structural model error.',
        reference={k:v for k,v in ref.items() if k!='trajectories'})
    write_json(output/'protocol.json', protocol)
    records, preparations = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(one_repeat, seed, sigma, ref) for sigma in NOISE_LEVELS
                for seed in range(first_seed, first_seed+repeats)]
        for done, future in enumerate(as_completed(jobs), 1):
            rows, preparation = future.result()
            records.extend(rows); preparations.append(preparation)
            records.sort(key=lambda r:(r['sigma'],r['seed'],r['group']))
            preparations.sort(key=lambda r:(r['sigma'],r['seed']))
            write_json(output/'results.json', dict(protocol=protocol, records=records, preparations=preparations))
            print(f'{done}/{len(jobs)}: seed {preparation["seed"]}, sigma {preparation["sigma"]}, '
                  f'{preparation["seconds"]:.1f} s, straddles {preparation["straddles_reference_saddle"]}', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    main(args.repeats,args.workers,args.pilot)
