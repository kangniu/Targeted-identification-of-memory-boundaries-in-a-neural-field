"""Write the manuscript tables and prediction summaries from saved fits."""
from pathlib import Path
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'tables'
RATE_GROUPS = ['ordinary_14', 'critical_half', 'far_boundary', 'near_boundary',
               'critical_shift_low', 'critical_shift_high']
STATE_GROUPS = ['ordinary', 'spread', 'adaptive_boundary']
JOINT_GROUPS = ['ordinary', 'true_far_boundary', 'true_near_boundary']


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row, key):
    raw = row['pulse_relative_errors'][2] if key == 'principal_erasure' else row.get(key)
    return np.nan if raw is None else float(raw)


def distribution(values):
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    quantiles = np.quantile(finite, [.1, .5, .9]) if len(finite) else [None]*3
    return dict(q10=quantiles[0], median=quantiles[1], q90=quantiles[2],
                valid_count=len(finite), unavailable_count=len(data)-len(finite))


def cells(data, groups, levels, key):
    expected = {(seed, group, level) for seed in data['protocol']['seeds']
                for group in groups for level in levels}
    actual = [(r['seed'], r['group'], r.get(key, levels[0])) for r in data['records']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('The saved comparison is incomplete or contains duplicate fits.')
    for level in levels:
        for group in groups:
            rows = sorted((r for r in data['records']
                           if r['group'] == group and r.get(key, levels[0]) == level),
                          key=lambda r: r['seed'])
            yield group, level, rows


def main():
    state = load('state_observations/results.json')
    rate = load('boundary_followup/results.json')
    joint = load('joint_initial/results.json')
    information = load('followup_information.json')
    OUT.mkdir(exist_ok=True)
    quality, pulse, table1, local = [], [], [], []
    comparisons = [
        ('state', state, STATE_GROUPS, [.005, .02], 'sigma'),
        ('rate', rate, RATE_GROUPS, [.01], 'sigma'),
        ('calibration', joint, JOINT_GROUPS, [0., .005, .02], 'tau'),
    ]
    metrics = [('trajectory_relative_rmse', 100., 'percent'),
               ('stable_branch_relative_rmse', 100., 'percent'),
               ('saddle_branch_relative_rmse', 100., 'percent'),
               ('fold_absolute_error', 1., 'absolute'),
               ('principal_erasure', 100., 'percent'),
               ('common_initial_pulse_relative_error', 100., 'percent')]
    for comparison, data, groups, levels, level_key in comparisons:
        output = []
        for group, level, rows in cells(data, groups, levels, level_key):
            label = dict(comparison=comparison, group=group,
                         noise_sd=level if level_key == 'sigma' else .02,
                         calibration_sd=level if level_key == 'tau' else 0.)
            for key, scale, unit in metrics:
                output.append(dict(**label, metric=key, unit=unit,
                                   **distribution([scale*value(r, key) for r in rows])))
            for index, (mu, duration) in enumerate(data['protocol']['reference']['protocols']):
                pulse.append(dict(**label, mu=mu, duration=duration,
                                  **distribution([np.nan if r['pulse_relative_errors'][index] is None
                                                  else 100*r['pulse_relative_errors'][index]
                                                  for r in rows])))
            quality.append(dict(**label, fits=len(rows),
                                fit_failures=sum(r['fit_status'] != 'success' for r in rows),
                                incomplete_branches=sum(r['branch_missing_count'] > 0 for r in rows),
                                missing_branch_points=sum(r['branch_missing_count'] for r in rows),
                                unavailable_thresholds=sum(s != 'success' for r in rows
                                                           for s in r['pulse_status'])))
            if comparison == 'state' and level == .02:
                table1.append(dict(design=group,
                                   recovery_relative_rmse_percent=np.median([100*value(r, metrics[0][0]) for r in rows]),
                                   stable_branch_relative_rmse_percent=np.median([100*value(r, metrics[1][0]) for r in rows]),
                                   incomplete_fits=sum(r['branch_missing_count'] > 0 for r in rows),
                                   total_fits=len(rows)))
            if comparison == 'rate':
                for target, values, predicted in [
                    ('fold', [r['fold'][1] for r in rows], information['groups'][group]['fold_sd']),
                    ('principal_erasure', [r['pulse_thresholds'][2] for r in rows],
                     information['groups'][group]['primary_pulse_sd'])]:
                    empirical = np.std(values, ddof=1)
                    local.append(dict(group=group, target=target, empirical_sd=empirical,
                                      predicted_sd=predicted, ratio=empirical/predicted))
        write_csv(f'{comparison}_metrics.csv', output)
    write_csv('table1.csv', table1)
    write_csv('quality.csv', quality)
    write_csv('pulse_errors_percent.csv', pulse)
    write_csv('rate_information.csv', local)
    print('Seven CSV tables written to tables/. All saved fits are retained.')


if __name__ == '__main__':
    main()
