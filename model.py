"""A smooth Amari ring with an exactly closed cosine amplitude equation."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.optimize import root, brentq, least_squares
from scipy.integrate import solve_ivp, quad

TRUE_PARAMETERS = np.array([6.0, 10.0, 0.15])  # coupling, sigmoid gain, offset


@dataclass
class RingField:
    n: int = 256

    def __post_init__(self):
        self.x = np.arange(self.n) * (2 * np.pi / self.n)
        self.c = np.cos(self.x)

    def terms(self, a, mu, p):
        a, mu = np.broadcast_arrays(np.asarray(a), np.asarray(mu))
        j, beta, theta = p
        z = a[..., None] * self.c - mu[..., None] - theta
        s = expit(beta * z)
        d = s * (1 - s)
        g = -a + j * np.mean(self.c * s, axis=-1)
        ga = -1 + j * beta * np.mean(self.c**2 * d, axis=-1)
        gm = -j * beta * np.mean(self.c * d, axis=-1)
        return g, ga, gm

    def g(self, a, mu, p):
        a, mu = np.broadcast_arrays(np.asarray(a), np.asarray(mu))
        j, beta, theta = p
        s = expit(beta * (a[..., None] * self.c - mu[..., None] - theta))
        return -a + j * np.mean(self.c * s, axis=-1)

    def parameter_jacobian(self, a, mu, p):
        a, mu = np.broadcast_arrays(np.asarray(a), np.asarray(mu))
        j, beta, theta = p
        z = a[..., None] * self.c - mu[..., None] - theta
        s = expit(beta * z)
        d = s * (1 - s)
        return np.stack([np.mean(self.c*s, axis=-1),
                         j*np.mean(self.c*d*z, axis=-1),
                         -j*beta*np.mean(self.c*d, axis=-1)], axis=-1)

    def fold(self, p):
        guess = [p[0] / (np.sqrt(2) * np.pi), p[0] / (2*np.pi) - p[2] - .035]
        sol = root(lambda z: self.terms(z[0], z[1], p)[:2], guess, tol=1e-10)
        if not sol.success and np.linalg.norm(sol.fun) > 1e-8:
            raise RuntimeError('Fold calculation failed')
        if sol.x[0] < .1 or np.linalg.norm(sol.fun) > 1e-7:
            raise RuntimeError('Non-bump root in fold calculation')
        return np.asarray(sol.x)

    def equilibria(self, mu, p):
        af, mf = self.fold(p)
        if mu >= mf:
            raise ValueError('No bistable bump pair at this parameter')
        g = lambda a: float(self.g(a, mu, p))
        if float(self.terms(0., mu, p)[1]) >= 0 or g(af) <= 0:
            raise ValueError('Bump/off bistability is absent')
        au = brentq(g, 1e-7, af, xtol=1e-12)
        ast = brentq(g, af, max(p[0], af*2), xtol=1e-12)
        return np.array([au, ast])

    def trajectory(self, a0, mu, p, times):
        sol = solve_ivp(lambda t, y: [float(self.g(y[0], mu, p))],
                        [float(times[0]), float(times[-1])], [float(a0)],
                        t_eval=times, rtol=2e-9, atol=2e-11, method='DOP853')
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[0]

    def pulse_threshold(self, mu, duration, p, a0=None):
        """Exact scalar separatrix condition, evaluated by the travel-time integral."""
        au, ast = self.equilibria(mu, p)
        a0 = ast if a0 is None else float(a0)
        if a0 <= au:
            raise ValueError('The initial state is outside the memory basin')
        if duration <= np.log(a0/au):
            return np.inf
        af, mf = self.fold(p)
        if not au < af < a0:
            raise ValueError('Pulse travel interval does not contain the fold')
        def travel(pulse):
            integral, _ = quad(lambda a: -1.0/float(self.g(a, mu+pulse, p)),
                               au, a0, points=[af], epsabs=2e-8,
                               epsrel=2e-8, limit=150)
            return integral
        lo = mf-mu+1e-4
        hi = max(lo+.15, .3)
        while travel(hi) > duration:
            hi *= 1.5
            if hi > 20:
                raise RuntimeError('No finite pulse threshold bracket')
        if travel(lo) < duration:
            lo = mf-mu+1e-7
        return brentq(lambda pulse: travel(pulse)-duration, lo, hi, xtol=2e-9)

    def pulse_trace(self, mu, duration, pulse, p, a0=None, recovery=30.):
        if a0 is None:
            a0 = self.equilibria(mu, p)[1]
        t1 = np.linspace(0,duration,101)
        a1 = self.trajectory(a0,mu+pulse,p,t1)
        t2 = np.linspace(0,recovery,301)
        a2 = self.trajectory(a1[-1],mu,p,t2)
        return np.r_[t1,t2[1:]+duration],np.r_[a1,a2[1:]]


def learn_parameters(field, a, mu, observed_rate, starts=None):
    if starts is None:
        starts = [[5.5,7.,.1],[6.5,18.,.2],[7.,30.,.3]]
    solutions=[]
    for p0 in starts:
        sol=least_squares(lambda p: field.g(a,mu,p)-observed_rate,p0,
                          jac=lambda p: field.parameter_jacobian(a,mu,p),
                          bounds=([3.,2.,-.15],[10.,40.,.45]),
                          x_scale=[6.,10.,.2],ftol=1e-11,xtol=1e-11,
                          gtol=1e-11,max_nfev=300)
        solutions.append(sol)
    best=min(solutions,key=lambda s:s.cost)
    return best.x,dict(cost=float(best.cost),success=bool(best.success),
                       optimality=float(best.optimality),
                       active_bounds=best.active_mask.tolist())
