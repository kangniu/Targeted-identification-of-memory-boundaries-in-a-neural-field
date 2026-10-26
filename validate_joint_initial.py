"""Validate all parameter and initial-state derivatives and the Schur covariance."""
from pathlib import Path
import json
import numpy as np
from model import RingField, TRUE_PARAMETERS
from joint_initial_model import joint_residual_jacobian, fit_joint
from state_observation_model import observe_trajectories
from run_joint_initial import fixed_designs
from run_boundary_followup import write_json

ROOT=Path(__file__).resolve().parent
field=RingField(256)
designs=fixed_designs()
info=json.loads((ROOT/'state_observations/information.json').read_text(encoding='utf-8'))
checks=[]
for name in ['true_far_boundary','true_near_boundary']:
    q=designs[name]
    x=np.r_[TRUE_PARAMETERS,q['initial']]
    residual,jac=joint_residual_jacobian(field,q['mu'],q['times'],q['amplitude'],
        q['initial'],.02,.02,x,rtol=1e-10,atol=1e-12)
    fd=np.zeros_like(jac)
    def independent(value):
        a,_=observe_trajectories(RingField(512),value[3:],q['mu'],q['times'],value[:3],
            sensitivities=False,rtol=1e-11,atol=1e-13)
        return np.r_[((a-q['amplitude'])/.02).ravel(),(value[3:]-q['initial'])/.02]
    for j in range(len(x)):
        h=[2e-5,2e-5,2e-7][j] if j<3 else 2e-7
        delta=np.eye(len(x))[j]*h
        fd[:,j]=(independent(x+delta)-independent(x-delta))/(2*h)
    scales=np.r_[[6.,10.,.15],np.ones(14)]
    weighted=jac*scales
    _,s,vh=np.linalg.svd(weighted,full_matrices=False)
    cov=(vh.T/s**2)@vh
    cov=cov*scales[:,None]*scales[None,:]
    predicted=next(r['parameter_covariance'] for r in info['records'] if r['group']==name
                   and r['sigma']==.02 and r['initial_calibration_sd']==.02)
    covariance_difference=np.linalg.norm(cov[:3,:3]-predicted)/np.linalg.norm(predicted)
    p,a0,fit=fit_joint(field,q['mu'],q['times'],q['amplitude'],q['initial'],.02,.02)
    check=dict(group=name,jacobian_relative_error=np.linalg.norm(jac-fd)/np.linalg.norm(fd),
        maximum_column_relative_error=max(np.linalg.norm(jac[:,j]-fd[:,j])/np.linalg.norm(fd[:,j]) for j in range(len(x))),
        covariance_relative_error=covariance_difference,
        noiseless_parameter_scaled_error=np.linalg.norm((p-TRUE_PARAMETERS)/[6.,10.,.15]),
        noiseless_initial_max_error=max(abs(a0-q['initial'])),fit=fit)
    checks.append(check)
    assert check['maximum_column_relative_error']<2e-5
    assert covariance_difference<2e-7
    assert check['noiseless_parameter_scaled_error']<2e-5
    assert check['noiseless_initial_max_error']<2e-6
out=ROOT/'joint_initial';out.mkdir(exist_ok=True)
write_json(out/'validation.json',checks)
print(json.dumps(checks,indent=2),flush=True)
