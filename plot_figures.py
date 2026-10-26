"""Reproduce the four manuscript figures from the saved results."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
sys.path.insert(0, str(ROOT / 'plots'))
from figure_branches import render as branches
from figure_state_comparison import render as states
from figure_coverage import render as coverage
from figure_calibration import render as calibration


def configure():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9.5, 'axes.labelsize': 9.5,
        'axes.titlesize': 9.5, 'legend.fontsize': 8.5, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 450,
        'axes.titleweight': 'normal', 'legend.frameon': False,
    })


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.035,
                facecolor='white', metadata={'CreationDate': None})
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=.035,
                facecolor='white', dpi=450)
    plt.close(fig)


def main():
    inputs = [ROOT/'state_observations/results.json',
              ROOT/'boundary_followup/results.json',
              ROOT/'joint_initial/results.json',
              ROOT/'joint_initial/tables/summary.json']
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    OUT.mkdir(exist_ok=True)
    configure()
    data = json.loads(inputs[0].read_text(encoding='utf-8'))
    figure, _ = branches(data)
    save(figure, 'fig01_memory_branches')
    save(states(data), 'fig02_noisy_state_learning')
    save(coverage(), 'fig03_coverage')
    save(calibration(), 'fig04_calibration')
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest
           for p, digest in before.items()):
        raise RuntimeError('Plotting changed a saved input file.')
    print('Four figures written to figures/ (vector PDF and 450 dpi PNG).')


if __name__ == '__main__':
    main()
