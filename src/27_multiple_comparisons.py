# -*- coding: utf-8 -*-
"""
STEP 27: MULTIPLE-COMPARISON CORRECTION
=======================================
Applies Benjamini-Hochberg FDR and Bonferroni correction to the permutation
p-values (14 models) and the DeLong-vs-demographics p-values (13 models), so the
manuscript reports corrected significance instead of raw p's (reviewer R2.2).

Emits *_corrected.csv with q_fdr (BH) and p_bonf columns, and prints how many
tests survive at 0.05 under each.

Run: python src/27_multiple_comparisons.py
"""
import csv
import numpy as np


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    out = np.empty(m); out[order] = np.clip(q, 0, 1)
    return out


def correct(path, pcol, label, out):
    rows = [r for r in csv.DictReader(open(path)) if r.get(pcol) not in (None, '', '-')]
    p = [float(r[pcol]) for r in rows]
    m = len(p)
    q = bh_fdr(p)
    bonf = np.clip(np.array(p) * m, 0, 1)
    for r, qq, bb in zip(rows, q, bonf):
        r['q_fdr'] = round(float(qq), 4); r['p_bonf'] = round(float(bb), 4)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    nfdr = int((q < 0.05).sum()); nbonf = int((bonf < 0.05).sum())
    print(f"\n{label}  ({m} tests)  -> {out}")
    print(f"  survive FDR q<0.05 : {nfdr}/{m}")
    print(f"  survive Bonferroni : {nbonf}/{m}  (alpha={0.05/m:.4f})")
    return rows


def main():
    # permutation: above-chance per model
    r1 = correct('results/table_perm.csv', 'p_value', 'Permutation (above chance)',
                 'results/table_perm_corrected.csv')
    sig = [r['model'] for r in r1 if float(r['q_fdr']) < 0.05]
    print('  above chance after FDR:', ', '.join(sig) if sig else 'none')

    # DeLong each model vs demographics
    r2 = correct('results/delong_pairwise_vs_demographics.csv', 'p_value',
                 'DeLong vs demographics', 'results/delong_vs_demographics_corrected.csv')
    sig2 = [r['model'] for r in r2 if float(r['q_fdr']) < 0.05]
    print('  differ from demographics after FDR:', ', '.join(sig2) if sig2 else 'NONE')

    # full DeLong matrix (91 pairs)
    correct('results/delong_pairwise.csv', 'p_value', 'DeLong pairwise (all pairs)',
            'results/delong_pairwise_corrected.csv')


if __name__ == '__main__':
    main()
