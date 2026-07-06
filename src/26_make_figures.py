# -*- coding: utf-8 -*-
"""
STEP 26: FIGURES  (regenerate all results figures from the committed CSVs)
==========================================================================
Reproducible figure generation, so every figure in the paper traces to a
`results/*.csv`. Outputs PNG (200 dpi) + PDF to figures/.

  Fig 1  benchmark forest plot         <- results/table_ci.csv
  Fig 3  leakage (honest vs leaky)     <- results/honest_vs_leaky_ci.csv
  Fig 4  confound-matching collapse    <- results/confound_matched.csv
  Fig 5  cross-cohort contrast         <- results/table_ci.csv + results/ds2_suite.csv

(Fig 2, the NeuroFormer architecture schematic, is a hand-drawn vector figure and
is not generated here.)

Run: python src/26_make_figures.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
def rd(p): return list(csv.DictReader(open(p)))
def save(fig, name):
    fig.savefig(f'figures/{name}.png', dpi=200); fig.savefig(f'figures/{name}.pdf'); plt.close(fig)
    print(f'  figures/{name}.png / .pdf')


def fig1_forest():
    rows = [r for r in rd('results/table_ci.csv') if r['status'] == 'ok']
    d = sorted([(r['model'], float(r['auc_oof']), float(r['ci_boot_lo']), float(r['ci_boot_hi']))
                for r in rows], key=lambda x: x[1])
    lab = [m.replace(' (concat)', '').replace(' (temporal)', '').replace(' (3-modality)', '') for m, *_ in d]
    auc = [x[1] for x in d]; lo = [x[1]-x[2] for x in d]; hi = [x[3]-x[1] for x in d]
    demo = [x for x in d if 'Demographics' in x[0]][0]
    fig, ax = plt.subplots(figsize=(7, 5.2)); y = range(len(d))
    ax.axvspan(demo[2], demo[3], color='#f2c744', alpha=0.18, zorder=0)
    ax.axvline(demo[1], color='#d4a017', lw=1.6, ls='--', label=f'Demographics baseline ({demo[1]:.2f})')
    ax.axvline(0.5, color='grey', lw=1, ls=':', label='Chance (0.50)')
    ax.errorbar(auc, y, xerr=[lo, hi], fmt='o', color='#34495e', ecolor='#95a5a6', elinewidth=2, capsize=3, ms=6, zorder=3)
    for i, x in enumerate(d):
        c = '#c0392b' if 'NeuroFormer' in x[0] else ('#d4a017' if 'Demographics' in x[0] else '#34495e')
        ax.plot(x[1], i, 'o', color=c, ms=7, zorder=4)
    ax.set_yticks(list(y)); ax.set_yticklabels(lab, fontsize=9); ax.set_xlim(0.30, 0.95)
    ax.set_xlabel('ROC-AUC (out-of-fold, 95% bootstrap CI)')
    ax.set_title('Leakage-free benchmark, MIPDB (N=92): no model exceeds the\ndemographic baseline (NeuroFormer in red)', fontsize=10.5)
    ax.legend(loc='lower right', fontsize=8); ax.grid(axis='x', alpha=0.25)
    fig.tight_layout(); save(fig, 'fig1_benchmark_forest')


def fig3_leakage():
    hl = rd('results/honest_vs_leaky_ci.csv')
    models = ['LogReg (concat)', 'SVM-RBF (concat)', 'RandomForest (concat)', 'MLP early-fusion']
    short = ['LogReg', 'SVM-RBF', 'RandomForest', 'MLP']
    honest = {r['model']: float(r['auc']) for r in hl if 'honest' in r['cv']}
    leaky = {r['model']: float(r['auc']) for r in hl if 'leaky' in r['cv']}
    fig, ax = plt.subplots(figsize=(7, 4.2)); y = np.arange(len(models)); w = 0.38
    ax.barh(y+w/2, [leaky[m] for m in models], w, color='#c0392b', label='Window-level (leaky)')
    ax.barh(y-w/2, [honest[m] for m in models], w, color='#34495e', label='Subject-level (honest)')
    ax.axvline(0.5, color='grey', ls=':', lw=1)
    ax.set_yticks(y); ax.set_yticklabels(short); ax.set_xlim(0.4, 1.02); ax.set_xlabel('ROC-AUC')
    ax.set_title('Evaluation leakage inflates AUC to ~1.0\n(window-level) vs ~0.5 (honest subject-level), MIPDB')
    ax.legend(loc='lower right', fontsize=9); ax.grid(axis='x', alpha=0.25)
    fig.tight_layout(); save(fig, 'fig3_leakage')


def fig4_confound():
    cm = rd('results/confound_matched.csv')
    sexm = {r['model']: float(r['auc_mean']) for r in cm if r['match'] == 'sex'}
    jm = {r['model']: float(r['auc_mean']) for r in cm if r['match'] == 'joint'}
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for m in sexm:
        h = 'Demographics' in m
        ax.plot([0, 1], [sexm[m], jm[m]], '-o', color='#c0392b' if h else '#95a5a6',
                lw=2.2 if h else 1.2, ms=6 if h else 4, zorder=3 if h else 2,
                label='Demographics (age+sex)' if h else None)
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['sex-matched', 'joint age+sex-matched'])
    ax.set_ylabel('ROC-AUC'); ax.set_xlim(-0.2, 1.2)
    ax.set_title('Removing the age+sex difference collapses the signal\n(demographics 0.71 -> 0.40), MIPDB')
    ax.legend(loc='upper right', fontsize=9); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); save(fig, 'fig4_confound_collapse')


def fig5_cross_cohort():
    mip = {r['model']: float(r['auc_oof']) for r in rd('results/table_ci.csv') if r['status'] == 'ok'}
    ds2 = {r['model']: float(r['auc_oof']) for r in rd('results/ds2_suite.csv') if r['status'] == 'run'}
    data = {'MIPDB (paediatric)': (mip['RandomForest (concat)'], mip['Demographics (age+sex)']),
            'ds003478 (adult)': (ds2['LogReg (concat)'], ds2['Demographics (age+sex)'])}
    fig, ax = plt.subplots(figsize=(6.5, 4.4)); x = np.arange(2); w = 0.36
    eeg = [data[c][0] for c in data]; demo = [data[c][1] for c in data]
    ax.bar(x-w/2, eeg, w, color='#2980b9', label='Best EEG-based model')
    ax.bar(x+w/2, demo, w, color='#d4a017', label='Demographics (age+sex)')
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    for i in range(2):
        ax.text(i-w/2, eeg[i]+.01, f'{eeg[i]:.2f}', ha='center', fontsize=9)
        ax.text(i+w/2, demo[i]+.01, f'{demo[i]:.2f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(list(data)); ax.set_ylim(0, 0.9); ax.set_ylabel('ROC-AUC')
    ax.set_title('EEG below demographics in the confounded paediatric cohort,\nabove demographics in the independent adult cohort')
    ax.legend(loc='upper left', fontsize=9); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); save(fig, 'fig5_cross_cohort')


def fig2_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    INK, TEAL, GOLD, BLUE, GREY, SOFT = '#1A2530', '#14646E', '#9C7211', '#2980b9', '#9aa7b1', '#E3EEEF'
    fig, ax = plt.subplots(figsize=(9.6, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis('off')

    def box(x, y, w, h, label, fc='white', ec=INK, tc=INK, fs=9, weight='normal', align='center'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2,rounding_size=2.5",
                                    fc=fc, ec=ec, lw=1.4, zorder=2))
        ax.text(x + (w/2 if align == 'center' else 2), y+h/2, label, ha=align, va='center',
                fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=INK, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                     color=color, lw=1.6, ls=ls, zorder=1, shrinkA=2, shrinkB=2))

    box(1, 40, 20, 10, 'EEG\n1665 feat → top-128', fc=SOFT, ec=TEAL, tc=TEAL, weight='bold')
    box(1, 24, 20, 9, 'Eye-tracking\n10 oculomotor feats', fc='#eef4fb', ec=BLUE, tc=BLUE, weight='bold')
    box(1, 10, 20, 8, 'Behaviour\n(absent in cohort)', fc='#f2f3f4', ec=GREY, tc=GREY, fs=8.5)
    box(30, 33, 28, 22, 'Hierarchical EEG encoder\n───────\nEuclidean align\nEEGNet tokenizer\ncriss-cross attention\nfrequency-aware attention\nresidual MLP',
        ec=TEAL, fs=8, align='left')
    box(30, 24, 20, 7, 'Linear projection', ec=BLUE, fs=8.5)
    box(30, 10, 20, 7, '(projection)', fc='#f2f3f4', ec=GREY, tc=GREY, fs=8)
    box(64, 22, 22, 20, 'Confidence-gated\nfusion\n──────\nper-modality\nconfidence → softmax\ngate → weighted sum', fc=SOFT, ec=TEAL, fs=8)
    box(64, 46, 26, 8, 'availability mask · modality dropout', fc='#fbf3dd', ec=GOLD, tc=GOLD, fs=8, weight='bold')
    arrow(77, 46, 75, 42, GOLD, ls=(0, (4, 2)))
    box(90, 28, 9, 9, 'Classifier\n→ DX', fc=INK, ec=INK, tc='white', fs=8.5, weight='bold')
    arrow(21, 45, 30, 44, TEAL); arrow(21, 28, 30, 27.5, BLUE); arrow(21, 14, 30, 13.5, GREY, ls=(0, (3, 2)))
    arrow(58, 44, 64, 36, TEAL); arrow(50, 27.5, 64, 32, BLUE); arrow(50, 13.5, 64, 26, GREY, ls=(0, (3, 2)))
    arrow(86, 32, 90, 32.5, INK)
    ax.text(50, 56, 'Figure 2. NeuroFormer: modality-specific encoders + confidence-gated fusion with a masked missing-modality path',
            ha='center', va='center', fontsize=9.5, color=INK, weight='bold')
    fig.tight_layout(); save(fig, 'fig2_architecture')


if __name__ == '__main__':
    print('Generating figures ->')
    fig1_forest(); fig2_architecture(); fig3_leakage(); fig4_confound(); fig5_cross_cohort()
    print('done.')
