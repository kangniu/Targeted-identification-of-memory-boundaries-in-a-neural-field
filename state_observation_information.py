"""Local information from state trajectories, including uncertain preparation."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import RingField, TRUE_PARAMETERS
from state_observation_model import observe_trajectories
from run_state_observations import ordinary_design
from run_boundary_followup import write_json
from followup_information import pulse_gradient, equilibrium_gradient

ROOT=Path(__file__).resolve().parent
SCALE=np.array([6.,10.,.15])


def initial_sensitivity(field,initial,mus,times,p):
    count=len(initial)
    def rhs(t,y):
        a,z=y[:count],y[count:]
        f,fa,_=field.terms(a,mus,p)
        return np.r_[f,fa*z]
    sol=solve_ivp(rhs,(0.,times[-1]),np.r_[initial,np.ones(count)],t_eval=times,
        method='DOP853',rtol=1e-10,atol=1e-12)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y[count:]


def covariance(s,z,sigma,tau):
    # Each unknown preparation has its own independent calibration observation.
    blocks=[]
    for sj,zj in zip(s,z):
        if tau==0:
            blocks.append(sj/sigma)
        else:
            precision=0. if np.isinf(tau) else (sigma/tau)**2
            # Symmetric square root of I-zz^T/(z^Tz+precision).
            norm2=zj@zj
            alpha=1-np.sqrt(precision/(norm2+precision))
            blocks.append((sj-alpha*np.outer(zj,zj@sj)/norm2)/sigma)
    _,values,vh=np.linalg.svd(np.concatenate(blocks),full_matrices=False)
    cov=(vh.T/values**2)@vh
    return cov,values


def main():
    field=RingField(512);p=TRUE_PARAMETERS
    ordinary_a,ordinary_mu,times=ordinary_design()
    af,mf=field.fold(p)
    gf=-field.parameter_jacobian(af,mf,p)/field.terms(af,mf,p)[2]
    pc,gp,check=pulse_gradient(field,.60,2.,p)
    branch_mu=np.linspace(.3,.74,45)
    branch=np.array([field.equilibria(mu,p) for mu in branch_mu])
    gb=np.array([[equilibrium_gradient(field,a,mu,p) for a in aa]
                 for mu,aa in zip(branch_mu,branch)])
    results=[];designs={}
    for name in ['ordinary','spread','true_far_boundary','true_near_boundary']:
        initial,mus=ordinary_a.copy(),ordinary_mu.copy()
        if name=='spread': initial[-2:]=[.4,1.3];mus[-2:]=.45
        elif name.startswith('true_'):
            mu=.45 if name=='true_far_boundary' else .735
            lower,upper=field.equilibria(mu,p)
            initial[-2:]=lower+np.array([-.05,.05])*(upper-lower);mus[-2:]=mu
        a,jac=observe_trajectories(field,initial,mus,times,p,rtol=1e-10,atol=1e-12)
        z=initial_sensitivity(field,initial,mus,times,p)
        designs[name]=dict(initial=initial,mus=mus,amplitude=a,
            scaled_singular_values=np.linalg.svd(jac.reshape(-1,3)*SCALE,compute_uv=False))
        for sigma in [.005,.02]:
            for tau in [0.,.005,.02,np.inf]:
                cov_scaled,singular=covariance(jac*SCALE,z,sigma,tau)
                cov=cov_scaled*SCALE[:,None]*SCALE[None,:]
                vbranch=np.einsum('...i,ij,...j->...',gb,cov,gb)
                results.append(dict(group=name,sigma=sigma,initial_calibration_sd='unmeasured' if np.isinf(tau) else tau,
                    fold_sd=np.sqrt(gf@cov@gf),primary_pulse_sd=np.sqrt(gp@cov@gp),
                    primary_pulse_relative_sd=np.sqrt(gp@cov@gp)/pc,
                    branch_relative_rms_sd=np.sqrt(np.mean(vbranch/branch**2,axis=0)),
                    parameter_covariance=cov,weighted_singular_values=singular))
    lower=field.equilibria(.45,p)[0]
    lam=field.terms(lower,.45,p)[1]
    forcing=field.parameter_jacobian(lower,.45,p)
    _,jac=observe_trajectories(field,[lower],[.45],times,p,rtol=1e-11,atol=1e-13)
    analytic=np.expm1(lam*times)[:,None]/lam*forcing
    output=dict(interpretation='Local covariance calculations, not new noisy-data fits. True boundary designs are mechanism references, not tested adaptive estimates.',
        initial_uncertainty='Each initial amplitude is a nuisance parameter with an independent calibration SD tau. tau=0 means known exactly; unmeasured has no calibration.',
        ordinary_and_spread='Fixed designs coincide with the new state-observation experiments. Adaptive unconditional covariance is not computed.',
        parameter_scale=SCALE,fold_gradient=gf,pulse_gradient=gp,pulse_gradient_check=check,
        exact_saddle=dict(mu=.45,amplitude=lower,growth_rate=lam,
            analytic_relative_error=np.linalg.norm(jac[0]-analytic)/np.linalg.norm(analytic),
            scaled_singular_values=np.linalg.svd(jac[0]*SCALE,compute_uv=False)),
        time=times,designs=designs,records=results)
    out=ROOT/'state_observations';out.mkdir(exist_ok=True)
    write_json(out/'information.json',output)
    for r in results:
        if r['sigma']==.02:
            print(r['group'],r['initial_calibration_sd'],'pulse SD %',round(100*r['primary_pulse_relative_sd'],3),flush=True)
    print('Exact-saddle relative sensitivity check',output['exact_saddle']['analytic_relative_error'])


if __name__=='__main__': main()
