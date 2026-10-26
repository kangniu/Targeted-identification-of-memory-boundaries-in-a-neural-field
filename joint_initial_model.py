"""Joint inference of physical parameters and calibrated initial amplitudes."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit
from state_observation_model import fit_states


def joint_flow(field, initial, mus, times, p, rtol=2e-8, atol=2e-10):
    initial, mus, times = map(np.asarray, (initial, mus, times))
    n = len(initial)
    y0 = np.zeros((n, 5))
    y0[:, 0] = initial
    y0[:, 4] = 1.
    coupling, beta, theta = p
    c = field.c[None, :]

    def rhs(t, y):
        v = y.reshape(n, 5)
        z = v[:, :1]*c-mus[:, None]-theta
        s = expit(beta*z)
        d = s*(1-s)
        cs = np.mean(c*s, axis=1)
        fa = -1+coupling*beta*np.mean(c*c*d, axis=1)
        fp = np.stack([cs, coupling*np.mean(c*d*z, axis=1),
                       -coupling*beta*np.mean(c*d, axis=1)], axis=1)
        out = np.empty_like(v)
        out[:, 0] = -v[:, 0]+coupling*cs
        out[:, 1:4] = fa[:, None]*v[:, 1:4]+fp
        out[:, 4] = fa*v[:, 4]
        return out.ravel()

    sol = solve_ivp(rhs, (0., float(times[-1])), y0.ravel(), t_eval=times,
                    method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    v = sol.y.reshape(n, 5, len(times))
    return v[:, 0], v[:, 1:4].transpose(0, 2, 1), v[:, 4]


def joint_residual_jacobian(field, mus, times, observed, calibrated, sigma, tau, x,
                             rtol=2e-8, atol=2e-10):
    a, s, k = joint_flow(field, x[3:], mus, times, x[:3], rtol, atol)
    n, nt = a.shape
    residual = np.r_[((a-observed)/sigma).ravel(), (x[3:]-calibrated)/tau]
    jac = np.zeros((n*nt+n, 3+n))
    jac[:n*nt, :3] = s.reshape(-1, 3)/sigma
    jac[np.arange(n*nt), 3+np.repeat(np.arange(n), nt)] = k.ravel()/sigma
    jac[n*nt:, 3:] = np.eye(n)/tau
    return residual, jac


def fit_joint(field, mus, times, observed, calibrated, sigma, tau, starts=None):
    if tau <= 0:
        raise ValueError('Use the exact-initial benchmark for tau=0.')
    if starts is None:
        starts = [[5.5, 7., .1], [6.5, 18., .2], [7., 30., .3]]
    n = len(mus)
    cached_x, cached_value = None, None

    def evaluate(x):
        nonlocal cached_x, cached_value
        if cached_x is None or not np.array_equal(x, cached_x):
            cached_value = joint_residual_jacobian(field, mus, times, observed,
                                                   calibrated, sigma, tau, x)
            cached_x = x.copy()
        return cached_value

    solutions = []
    for p0 in starts:
        x0 = np.r_[p0, np.clip(calibrated, 1e-6, 4.-1e-6)]
        sol = least_squares(lambda x: evaluate(x)[0], x0,
            jac=lambda x: evaluate(x)[1],
            bounds=(np.r_[[3., 2., -.15], np.zeros(n)],
                    np.r_[[10., 40., .45], np.full(n, 4.)]),
            x_scale=np.r_[[6., 10., .2], np.ones(n)],
            ftol=2e-9, xtol=2e-9, gtol=2e-9, max_nfev=220)
        solutions.append(sol)
    best = min(solutions, key=lambda q: q.cost)
    nstate = observed.size
    info = dict(success=bool(best.success), cost=float(best.cost),
        optimality=float(best.optimality), active_bounds=best.active_mask.tolist(),
        parameter_active_bounds=best.active_mask[:3].tolist(),
        initial_active_bounds=best.active_mask[3:].tolist(),
        nfev=int(best.nfev), all_costs=[float(q.cost) for q in solutions],
        all_success=[bool(q.success) for q in solutions],
        training_rmse=float(np.sqrt(np.mean((best.fun[:nstate]*sigma)**2))),
        calibration_residual_rmse=float(np.sqrt(np.mean((best.x[3:]-calibrated)**2))))
    return best.x[:3], best.x[3:], info


def fit_exact(field, initial, mus, times, observed, sigma):
    p, info = fit_states(field, initial, mus, times, observed)
    info['cost'] /= sigma**2
    info['all_costs'] = [c/sigma**2 for c in info['all_costs']]
    info['parameter_active_bounds'] = info['active_bounds']
    info['initial_active_bounds'] = []
    info['calibration_residual_rmse'] = 0.
    return p, np.array(initial), info
