"""Independent checks of trajectory integration and its parameter Jacobian."""
from pathlib import Path
import time
import numpy as np
from model import RingField, TRUE_PARAMETERS
from state_observation_model import observe_trajectories, fit_states
from run_state_observations import ordinary_design
from run_boundary_followup import write_json

ROOT=Path(__file__).resolve().parent
out=ROOT/'state_observations'
out.mkdir(exist_ok=True)
field=RingField(256)
initial, mus, times=ordinary_design()
au, ast=field.equilibria(.45,TRUE_PARAMETERS)
initial[-2:]=au+np.array([-.05,.05])*(ast-au)
mus[-2:]=.45
amplitude,jac=observe_trajectories(field,initial,mus,times,TRUE_PARAMETERS)
fulltime=np.r_[0.,times]
independent=np.array([RingField(512).trajectory(a0,mu,TRUE_PARAMETERS,fulltime)[1:]
                      for a0,mu in zip(initial,mus)])
fd=np.empty_like(jac)
for j,step in enumerate([2e-5,2e-5,2e-7]):
    dp=np.eye(3)[j]*step
    plus,_=observe_trajectories(RingField(512),initial,mus,times,TRUE_PARAMETERS+dp,
        sensitivities=False,rtol=1e-11,atol=1e-13)
    minus,_=observe_trajectories(RingField(512),initial,mus,times,TRUE_PARAMETERS-dp,
        sensitivities=False,rtol=1e-11,atol=1e-13)
    fd[:,:,j]=(plus-minus)/(2*step)
relative=np.linalg.norm(jac-fd)/np.linalg.norm(fd)
grid=[]
for p in [TRUE_PARAMETERS, np.array([5.5,7.,.1]), np.array([6.5,18.,.2]),
          np.array([7.,30.,.3]), np.array([6.,39.,.15])]:
    a256,_=observe_trajectories(field,initial,mus,times,p,rtol=1e-10,atol=1e-12)
    a512,_=observe_trajectories(RingField(512),initial,mus,times,p,rtol=1e-10,atol=1e-12)
    grid.append(dict(parameters=p,max_absolute_difference=np.max(abs(a256-a512))))
start=time.perf_counter()
fit,info=fit_states(field,initial,mus,times,independent)
result=dict(independent_trajectory_max_error=np.max(abs(amplitude-independent)),
    sensitivity_relative_difference=relative,
    sensitivity_component_relative_differences=[np.linalg.norm(jac[:,:,j]-fd[:,:,j])/np.linalg.norm(fd[:,:,j]) for j in range(3)],
    quadrature_checks=grid,noiseless_parameters=fit,noiseless_scaled_error=np.linalg.norm((fit-TRUE_PARAMETERS)/[6.,10.,.15]),
    noiseless_fit=info,noiseless_fit_seconds=time.perf_counter()-start)
write_json(out/'validation.json',result)
print(result,flush=True)
assert result['independent_trajectory_max_error']<2e-6
assert relative<2e-5
assert result['noiseless_scaled_error']<2e-5
assert max(q['max_absolute_difference'] for q in grid)<1e-5
