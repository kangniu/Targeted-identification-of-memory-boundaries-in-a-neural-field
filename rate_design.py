"""Fixed trajectory design for the rate-observation experiments."""
import numpy as np
from model import TRUE_PARAMETERS


def make_design(field,extended=False):
    rng=np.random.default_rng(20261025)
    times=np.linspace(0.,6.,25)
    mu=np.linspace(.30,.55,14)
    rng.shuffle(mu)
    # All training paths start in the stable memory basin.
    initial_factors=np.resize([.8,1.2],14)+rng.uniform(-.025,.025,14)
    paths=[]
    for k,m in enumerate(mu):
        a0=field.equilibria(m,TRUE_PARAMETERS)[1]*initial_factors[k]
        a=field.trajectory(a0,m,TRUE_PARAMETERS,times)
        paths.append(dict(kind='ordinary',mu=float(m),a0=float(a0),
                          t=times,a=a,g=field.g(a,m,TRUE_PARAMETERS)))
    for m in [.72,.75]:
        a0=field.equilibria(.55,TRUE_PARAMETERS)[1]
        a=field.trajectory(a0,m,TRUE_PARAMETERS,times)
        paths.append(dict(kind='near_fold',mu=m,a0=float(a0),
                          t=times,a=a,g=field.g(a,m,TRUE_PARAMETERS)))
    if extended:
        # Fixed preparation: half the squared norm of the existing reference bump.
        # The same preparation is tested in ordinary and near-fold parameter regions.
        a0=field.equilibria(.55,TRUE_PARAMETERS)[1]/np.sqrt(2)
        for kind,parameters in [('critical_half',[.72,.75]),('ordinary_half',[.30,.55])]:
            for m in parameters:
                a=field.trajectory(a0,m,TRUE_PARAMETERS,times)
                paths.append(dict(kind=kind,mu=m,a0=float(a0),
                                  t=times,a=a,g=field.g(a,m,TRUE_PARAMETERS)))
    return paths
