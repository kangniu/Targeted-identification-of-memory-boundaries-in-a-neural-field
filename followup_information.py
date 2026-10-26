"""Local target uncertainty for fixed neural-field sampling protocols.

No training errors or held-out errors are used to select a protocol.  States
are treated as known covariates of noisy velocity measurements.  The pulse
gradient is obtained from the variational equation and checked independently.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from model import RingField, TRUE_PARAMETERS
from rate_design import make_design

ROOT = Path(__file__).resolve().parent
SCALE = np.array([6., 10., .15])
SIGMA = .01


def fixed_designs(field):
    p = TRUE_PARAMETERS
    original = make_design(field, extended=True)
    times = original[0]['t']
    common = original[:12]

    def path(mu, a0):
        a = field.trajectory(a0, mu, p, times)
        return dict(mu=float(mu), a0=float(a0), t=times, a=a,
                    g=field.g(a, mu, p))

    def boundary(mu, center_shift=0.):
        au, ast = field.equilibria(mu, p)
        gap = ast-au
        return [path(mu, au+(center_shift+s*.05)*gap) for s in [-1, 1]]

    a_half = field.equilibria(.55, p)[1]/np.sqrt(2)
    added = {
        'critical_half': original[16:18],
        'far_boundary': boundary(.45),
        'near_boundary': boundary(.735),
        'critical_shift_low': [path(m, a_half) for m in [.71, .74]],
        'critical_shift_high': [path(m, a_half) for m in [.73, .76]],
    }
    return dict(ordinary_14=original[:14],
                **{name: common+paths for name, paths in added.items()})


def equilibrium_gradient(field, a, mu, p):
    return -field.parameter_jacobian(a, mu, p)/field.terms(a, mu, p)[1]


def pulse_gradient(field, mu, duration, p):
    au, ast = field.equilibria(mu, p)
    pc = field.pulse_threshold(mu, duration, p)
    upper_grad = equilibrium_gradient(field, ast, mu, p)
    lower_grad = equilibrium_gradient(field, au, mu, p)

    def rhs(t, y):
        g, ga, gm = field.terms(y[0], mu+pc, p)
        gp = field.parameter_jacobian(y[0], mu+pc, p)
        return np.r_[g, ga*y[1:4]+gp, ga*y[4]+gm]

    sol = solve_ivp(rhs, [0., duration], np.r_[ast, upper_grad, 0.],
                    method='DOP853', rtol=2e-10, atol=2e-12)
    final = sol.y[:, -1]
    grad = -(final[1:4]-lower_grad)/final[4]
    fd = []
    for k, scale in enumerate(SCALE):
        delta = np.zeros(3)
        delta[k] = scale*1e-4
        fd.append((field.pulse_threshold(mu, duration, p+delta)
                   -field.pulse_threshold(mu, duration, p-delta))/(2*delta[k]))
    fd = np.asarray(fd)
    return pc, grad, dict(endpoint_residual=float(final[0]-au),
                         endpoint_pulse_derivative=float(final[4]),
                         finite_difference_gradient=fd.tolist(),
                         gradient_relative_difference=float(
                             np.linalg.norm((grad-fd)*SCALE)/np.linalg.norm(grad*SCALE)))


def main():
    field = RingField(512)
    p = TRUE_PARAMETERS
    designs = fixed_designs(field)
    af, mf = field.fold(p)
    fold_gradient = -field.parameter_jacobian(af, mf, p)/field.terms(af, mf, p)[2]
    branch_mu = np.linspace(.30, .74, 45)
    branch = np.array([field.equilibria(m, p) for m in branch_mu])
    branch_gradient = np.array([[equilibrium_gradient(field, a, m, p) for a in aa]
                               for m, aa in zip(branch_mu, branch)])
    pc, pc_gradient, pulse_check = pulse_gradient(field, .60, 2., p)
    result = dict(parameter_scale=SCALE.tolist(), noise_standard_deviation=SIGMA,
                  reference_parameters=p.tolist(), reference_fold=[float(af), float(mf)],
                  branch_mu=branch_mu.tolist(), reference_branch=branch.tolist(),
                  fold_gradient=fold_gradient.tolist(), branch_gradient=branch_gradient.tolist(),
                  primary_pulse=dict(mu=.60, duration=2., threshold=float(pc),
                                     gradient=pc_gradient.tolist(), verification=pulse_check),
                  assumptions='Conditional on known states; correct model; independent equal-variance '
                  'rate noise. Linearized covariance is not a finite-noise error guarantee.', groups={})
    for name, paths in designs.items():
        a = np.concatenate([q['a'] for q in paths])
        mu = np.concatenate([np.full_like(q['a'], q['mu']) for q in paths])
        scaled_jac = field.parameter_jacobian(a, mu, p)*SCALE
        _, singular, vh = np.linalg.svd(scaled_jac, full_matrices=False)
        covariance_scaled = (vh.T*(SIGMA/singular)**2)@vh
        covariance = covariance_scaled*SCALE[:, None]*SCALE[None, :]
        branch_variance = np.einsum('...i,ij,...j->...', branch_gradient, covariance, branch_gradient)
        path_details = []
        for q in paths[-2:]:
            au, ast = field.equilibria(q['mu'], p)
            state_jac = field.parameter_jacobian(q['a'], q['mu'], p)*SCALE
            state_singular = np.linalg.svd(state_jac, compute_uv=False)
            path_details.append(dict(mu=q['mu'], a0=q['a0'],
                amin=float(np.min(q['a'])), amax=float(np.max(q['a'])),
                amplitude_span=float(np.ptp(q['a'])), afinal=float(q['a'][-1]),
                saddle=float(au), stable=float(ast),
                normalized_initial_distance_to_saddle=float((q['a0']-au)/(ast-au)),
                saddle_growth_rate=float(field.terms(au, q['mu'], p)[1]),
                rate_rms=float(np.sqrt(np.mean(q['g']**2))),
                scaled_jacobian_singular_values=state_singular.tolist()))
        result['groups'][name] = dict(observations=len(a),
            scaled_jacobian_singular_values=singular.tolist(),
            scaled_jacobian_condition=float(singular[0]/singular[-1]),
            weakest_relative_parameter_direction=vh[-1].tolist(),
            parameter_sd=np.sqrt(np.diag(covariance)).tolist(),
            parameter_covariance=covariance.tolist(),
            fold_sd=float(np.sqrt(fold_gradient@covariance@fold_gradient)),
            branch_relative_rms_sd=np.sqrt(np.mean(branch_variance/branch**2, axis=0)).tolist(),
            primary_pulse_sd=float(np.sqrt(pc_gradient@covariance@pc_gradient)),
            primary_pulse_relative_sd=float(np.sqrt(pc_gradient@covariance@pc_gradient)/pc),
            last_two_paths=path_details)
    target = ROOT/'followup_information.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, row in result['groups'].items():
        print(name, 's_min:', round(row['scaled_jacobian_singular_values'][-1], 5),
              'fold_sd:', f"{row['fold_sd']:.3g}",
              'saddle_rel_sd:', f"{row['branch_relative_rms_sd'][0]:.3g}",
              'pulse_rel_sd:', f"{row['primary_pulse_relative_sd']:.3g}")
    print('Pulse gradient check:', pulse_check)


if __name__ == '__main__':
    main()
