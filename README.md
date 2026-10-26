# Targeted identification of memory boundaries in a neural field

Python code and saved numerical results for the manuscript by Xia-Guang Duan,
Jiaxin Sun, and Qiang Niu. The code uses a smooth cosine-kernel neural field and
its scalar amplitude equation to compare ordinary and targeted observations.

## Installation

Tested with Python 3.12.10. NumPy, SciPy, and Matplotlib are the only direct
dependencies; their tested versions are pinned in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

No GPU, external dataset, or LaTeX installation is needed.

## Reproduce the figures and tables

Run these commands from this directory:

```bash
python plot_figures.py
python summarize.py
```

The first command reads the included results and writes the four manuscript
figures to `figures/`, as vector PDF and 450 dpi PNG. The second writes seven
CSV tables to `tables/`, including Table 1, branch/fold/erasure errors, missing
outcomes, and empirical-to-predicted SD ratios. No parameter fits are performed
by either command.

| Manuscript | Numerical comparison | Figure |
| --- | --- | --- |
| Example 1 | Ordinary, spread, and adaptive state observations | 1, 2 |
| Example 2 | Six fixed rate-observation designs | 3 |
| Example 3 | Joint estimation with initial calibration noise | 4 |

Figure 3 displays four of the six rate designs; the two boundary-pair designs
are also retained for the comparisons discussed in the text.

## Recompute the experiments

For a full rerun, use a fresh copy and execute the following commands in order.
They rebuild the included results in place. Each experiment defaults to the
30 paired random seeds used in the manuscript; `--workers` controls parallelism.

```bash
python followup_information.py
python state_observation_information.py
python run_state_observations.py --repeats 30 --workers 2
python run_boundary_followup.py --repeats 30 --workers 2
python run_joint_initial.py --repeats 30 --workers 2
python analyze_joint_initial.py
python summarize.py
python plot_figures.py
```

The state-information calculation must precede the joint experiment because
the latter stores the local covariance predictions before fitting. The joint
analysis recomputes the SDs and the 10,000 paired bootstrap resamples used in
Figure 4. Fitting results may vary slightly with library versions and
floating-point hardware.

## Numerical checks

```bash
python validate_state_observations.py
python validate_joint_initial.py
```

These checks compare variational derivatives with finite differences,
refine the spatial quadrature, recover noiseless parameters, and compare the
joint covariance with the Schur-complement calculation. They use the included
state-information file, which can also be regenerated as above.

## Files

- `model.py`: amplitude equation, equilibria, fold, and finite-time erasure threshold.
- `state_observation_model.py`, `joint_initial_model.py`: state and joint estimators.
- `rate_design.py`: fixed rate-observation trajectories.
- `run_*.py`: the three numerical experiments.
- `*_information.py`: target sensitivities and local uncertainty calculations.
- `analyze_joint_initial.py`, `summarize.py`: statistical summaries.
- `plots/`, `plot_figures.py`: the four manuscript figures.
- `state_observations/`, `boundary_followup/`, `joint_initial/`: saved results and predictions.

The saved results contain 180 state fits and 60 preliminary preparations,
180 rate fits, and 270 joint fits. They retain every repetition of the designs
reported in the manuscript, including incomplete branches and unavailable
thresholds. Branch RMSEs use existing branch points; missing pairs are counted
separately. The result files contain fitted outputs and protocols; the noisy
observation arrays are regenerated from the recorded designs and random seeds.

The state experiment assumes known initial amplitudes. The rate and calibration
comparisons use fixed locations defined by the reference branches. The model
has three unknown parameters and is correctly specified; these experiments do
not test unknown kernels or biological data.

The archive contains the manuscript experiments only. Generated figures,
tables, caches, and exploratory studies are excluded. The `.gitignore` keeps
the required saved results and Figure 4 summary available for a fresh clone.

## Citation and contact

Please cite the accompanying manuscript, *Targeted identification of memory
boundaries in a neural field*, by Xia-Guang Duan, Jiaxin Sun, and Qiang Niu.
`CITATION.cff` supplies the repository citation metadata.

Corresponding author: Jiaxin Sun, sgsun29@liverpool.ac.uk.
