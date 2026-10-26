"""Parameter inference from amplitude observations with forward sensitivities."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit


def observe_trajectories(field, initial, mus, times, p, sensitivities=True,
                         rtol=2e-8, atol=2e-10):
    initial, mus, times = map(np.asarray, (initial, mus, times))
    count = len(initial)
    width = 4 if sensitivities else 1
    y0 = np.zeros((count, width))
    y0[:, 0] = initial
    coupling, beta, theta = p
    cosine = field.c[None, :]

    def rhs(t, y):
        state = y.reshape(count, width)
        a = state[:, :1]
        z = a*cosine-mus[:, None]-theta
        sigmoid = expit(beta*z)
        d = sigmoid*(1-sigmoid)
        c_sigmoid = np.mean(cosine*sigmoid, axis=1)
        out = np.empty_like(state)
        out[:, 0] = -state[:, 0]+coupling*c_sigmoid
        if sensitivities:
            ga = -1+coupling*beta*np.mean(cosine**2*d, axis=1)
            gp = np.stack([c_sigmoid,
                coupling*np.mean(cosine*d*z, axis=1),
                -coupling*beta*np.mean(cosine*d, axis=1)], axis=1)
            out[:, 1:] = ga[:, None]*state[:, 1:]+gp
        return out.ravel()

    sol = solve_ivp(rhs, (0., float(times[-1])), y0.ravel(), t_eval=times,
                    method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    values = sol.y.reshape(count, width, len(times))
    a = values[:, 0, :]
    jac = values[:, 1:, :].transpose(0, 2, 1) if sensitivities else None
    return a, jac


def fit_states(field, initial, mus, times, observed, starts=None):
    if starts is None:
        starts = [[5.5, 7., .1], [6.5, 18., .2], [7., 30., .3]]
    cached_p = None
    cached_value = None

    def evaluate(p):
        nonlocal cached_p, cached_value
        if cached_p is None or not np.array_equal(p, cached_p):
            a, jac = observe_trajectories(field, initial, mus, times, p)
            cached_p = p.copy()
            cached_value = ((a-observed).ravel(), jac.reshape(-1, 3))
        return cached_value

    solutions = []
    for p0 in starts:
        sol = least_squares(lambda p: evaluate(p)[0], p0,
            jac=lambda p: evaluate(p)[1],
            bounds=([3., 2., -.15], [10., 40., .45]),
            x_scale=[6., 10., .2], ftol=2e-9, xtol=2e-9, gtol=2e-9,
            max_nfev=180)
        solutions.append(sol)
    best = min(solutions, key=lambda s: s.cost)
    return best.x, dict(success=bool(best.success), cost=float(best.cost),
        optimality=float(best.optimality), active_bounds=best.active_mask.tolist(),
        nfev=int(best.nfev), all_costs=[float(s.cost) for s in solutions],
        all_success=[bool(s.success) for s in solutions],
        training_rmse=float(np.sqrt(2*best.cost/observed.size)))
