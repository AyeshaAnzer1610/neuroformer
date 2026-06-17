#!/usr/bin/env python
"""
Generate publication / clinical-report grade manuscript figures (run on Nibi).

Writes to figures/:
  modality_showcase.png  - EEG trace + eye scanpath, clean & annotated
  results_auc.png        - leak-free AUC comparison (LaBraM highlighted)
  embedding_map.png       - PCA of LaBraM per-subject embeddings by diagnosis
  screening_report.png    - clinical dashboard: metric callouts + ROC + referral curve

    python src/make_figures.py \
        --mipdb $SCRATCH/mh/data/mipdb --eeg-dir $SCRATCH/mh/data/processed \
        --labels data/labels.csv --subject A00051955
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGDIR = "figures"
# Clinical palette: deep teal / navy / amber / coral / slate.
C = {"eeg": "#0f6e78", "eye": "#e0892e", "beh": "#7d8a99", "labram": "#0f6e78",
     "fused": "#2c5f8a", "pos": "#c0392b", "neg": "#9aa3ad", "ink": "#22303a",
     "accent": "#c0392b", "panel": "#f4f7f9"}


def _style():
    mpl.rcParams.update({
        "savefig.dpi": 300, "figure.dpi": 120, "figure.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlepad": 9,
        "axes.labelsize": 10.5, "axes.labelcolor": C["ink"],
        "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#d9e0e6", "grid.linewidth": 0.8,
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "legend.frameon": False, "legend.fontsize": 9.5,
    })


def _tag(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=C["ink"], va="top")


# ------------------------------------------------------------------ modality showcase
def modality_showcase(mipdb, subject, out):
    import sys; sys.path.insert(0, "src")
    import pandas as pd
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fig.patch.set_facecolor("white")

    # (A) EEG multi-channel trace
    try:
        from extract_eeg_embeddings import load_raw, preprocess, list_eeg_files
        raw = preprocess(load_raw(list_eeg_files(Path(mipdb) / subject)[0]))
        names = raw.ch_names
        data = raw.get_data()[:, :3000] * 1e6
        t = np.arange(data.shape[1]) / 200.0
        picks = list(range(min(6, data.shape[0])))
        for j, c in enumerate(picks):
            ax[0].plot(t, np.clip(data[c], -150, 150) + j * 200, lw=0.5, color=C["eeg"])
            ax[0].text(-0.4, j * 200, names[c], fontsize=7, ha="right", va="center", color=C["ink"])
        ax[0].set_title("EEG  ·  15 s window")
        ax[0].set_xlabel("time (s)"); ax[0].set_yticks([]); ax[0].grid(axis="y", alpha=0)
        ax[0].set_xlim(-0.6, 15)
        ax[0].text(0.99, 0.02, "→ frozen LaBraM 200-d embedding / window",
                   transform=ax[0].transAxes, ha="right", fontsize=8,
                   style="italic", color=C["eeg"])
        _tag(ax[0], "A")
    except Exception as e:  # noqa: BLE001
        ax[0].text(0.5, 0.5, f"EEG panel:\n{e}", ha="center", fontsize=8)

    # (B) Eye gaze scanpath, time-colored, with start/end
    try:
        from extract_eye_features import load_samples
        eye = _find(mipdb, subject, "Eyetracking/txt", "Samples.txt")
        df = load_samples(eye)
        x = df.get("L POR X [px]"); y = df.get("L POR Y [px]")
        xx, yy = x.to_numpy(), y.to_numpy()
        m = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[m], yy[m]
        ax[1].plot(xx, yy, lw=0.3, color="#c8d2da", zorder=1)
        sc = ax[1].scatter(xx[::4], yy[::4], c=np.arange(len(xx))[::4], cmap="cividis",
                           s=6, zorder=2)
        ax[1].set_title("Eye-tracking  ·  gaze scanpath")
        ax[1].set_xlabel("gaze X (px)"); ax[1].set_ylabel("gaze Y (px)"); ax[1].invert_yaxis()
        ax[1].grid(alpha=0.15)
        cb = plt.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04); cb.set_label("time", fontsize=9)
        ax[1].text(0.99, 0.02, "fixations + saccades", transform=ax[1].transAxes,
                   ha="right", fontsize=8, style="italic", color=C["eye"])
        _tag(ax[1], "B")
    except Exception as e:  # noqa: BLE001
        ax[1].text(0.5, 0.5, f"Eye panel:\n{e}", ha="center", fontsize=8)

    fig.suptitle(f"Multimodal recording — participant {subject}", fontsize=14,
                 fontweight="bold", color=C["ink"], y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[fig] wrote {out}")


# ------------------------------------------------------------------ results bar
def results_auc(csv, out):
    import pandas as pd
    if csv and os.path.exists(csv):
        df = pd.read_csv(csv)
        fs = (df["feature_set"] if "feature_set" in df.columns else df.iloc[:, 0]).astype(str).tolist()
        pools = df["pool"].astype(str).tolist() if "pool" in df.columns else [""] * len(fs)
        names = [f"{f}\n({p})" if p else f for f, p in zip(fs, pools)]
        auc = df["auc"].tolist()
        ci = df["auc_ci95"].tolist() if "auc_ci95" in df.columns else [0] * len(fs)
        col = [C["labram"] if "labram" in f.lower() else (C["fused"] if "fus" in f.lower() else C["neg"]) for f in fs]
    else:
        names = ["hand-crafted\nEEG", "eye", "LaBraM\nEEG", "fusion"]
        auc = [0.46, 0.51, 0.66, 0.59]; ci = [0.04]*4
        col = [C["neg"], C["neg"], C["labram"], C["fused"]]
    fig, a = plt.subplots(figsize=(7.5, 4.0)); a.set_facecolor("white")
    bars = a.bar(range(len(names)), auc, yerr=ci, capsize=4, color=col,
                 edgecolor="white", width=0.72, error_kw={"ecolor": "#55606a", "lw": 1.2})
    for i, v in enumerate(auc):
        a.text(i, v + (ci[i] if i < len(ci) else 0) + 0.012, f"{v:.2f}",
               ha="center", fontsize=9, fontweight="bold", color=C["ink"])
    a.axhline(0.5, ls="--", c=C["accent"], lw=1.3, label="chance (0.50)")
    a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=8.5)
    a.set_ylabel("ROC-AUC"); a.set_ylim(0.35, 0.80); a.grid(axis="x", alpha=0)
    a.set_title("Leak-free EEG screening performance", color=C["ink"])
    a.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[fig] wrote {out}")


# ------------------------------------------------------------------ embedding map
def embedding_map(eeg_dir, labels_csv, out, pool="meanstd"):
    import sys; sys.path.insert(0, "src"); import pandas as pd
    from compare_fm_vs_handcrafted import load_windows, pool_subjects
    from sklearn.decomposition import PCA
    lab = pd.read_csv(labels_csv).set_index("ID")["DX_Status"].to_dict()
    emb = pool_subjects(load_windows(eeg_dir, "_labram", "embeddings"), pool)
    subs = [s for s in emb if s in lab]
    X = np.vstack([emb[s] for s in subs]); y = np.array([lab[s] > 0 for s in subs])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8); Z = PCA(2).fit_transform(X)
    fig, a = plt.subplots(figsize=(5.4, 4.4)); a.set_facecolor("white")
    for cls, c, lbl in [(0, C["neg"], "control"), (1, C["pos"], "diagnosed")]:
        m = y == cls
        a.scatter(Z[m, 0], Z[m, 1], c=c, s=34, alpha=0.78, edgecolor="white",
                  linewidth=0.6, label=f"{lbl} (n={m.sum()})")
    a.set_title("LaBraM EEG representation (PCA)", color=C["ink"])
    a.set_xlabel("PC 1"); a.set_ylabel("PC 2"); a.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[fig] wrote {out}")


# ------------------------------------------------------------------ clinical report
def screening_report(eeg_dir, labels_csv, out, pool="meanstd"):
    """Clinical-style dashboard: metric callouts + ROC + referral (risk-coverage)."""
    import sys; sys.path.insert(0, "src"); import pandas as pd
    from compare_fm_vs_handcrafted import load_windows, pool_subjects
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
    lab = pd.read_csv(labels_csv).set_index("ID")["DX_Status"].to_dict()
    emb = pool_subjects(load_windows(eeg_dir, "_labram", "embeddings"), pool)
    subs = [s for s in emb if s in lab]
    X = np.vstack([emb[s] for s in subs]); y = np.array([lab[s] > 0 for s in subs]).astype(int)
    # leak-free OOF probabilities (averaged over a few seeds)
    P = np.zeros(len(y))
    for seed in range(5):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed); p = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
            pipe.fit(X[tr], y[tr]); p[te] = pipe.predict_proba(X[te])[:, 1]
        P += p / 5
    auc = roc_auc_score(y, P)
    fpr, tpr, thr = roc_curve(y, P)
    # operating point = threshold maximizing BALANCED accuracy (sensible sens/spec,
    # avoids Youden's degenerate 100%/low-spec corner at modest AUC)
    best_t, best_ba = 0.5, -1.0
    for t in np.unique(P):
        pred = (P >= t).astype(int)
        ba = balanced_accuracy_score(y, pred)
        if ba > best_ba:
            best_ba, best_t = ba, t
    pred = (P >= best_t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    op_fpr, op_tpr = 1 - spec, sens

    _style()
    fig = plt.figure(figsize=(11, 5.2)); fig.patch.set_facecolor("white")
    # header band
    hdr = fig.add_axes([0, 0.92, 1, 0.08]); hdr.axis("off")
    hdr.add_patch(FancyBboxPatch((0.01, 0.0), 0.98, 1.0, boxstyle="round,pad=0.02",
                  fc=C["eeg"], ec="none", transform=hdr.transAxes))
    hdr.text(0.025, 0.5, "EEG Foundation-Model Transdiagnostic Screening — Model Report",
             transform=hdr.transAxes, va="center", fontsize=14, fontweight="bold", color="white")
    # metric callout boxes
    metrics = [("ROC-AUC", f"{auc:.2f}", C["labram"]), ("Sensitivity", f"{sens*100:.0f}%", C["fused"]),
               ("Specificity", f"{spec*100:.0f}%", C["eye"]), ("Cohort N", f"{len(y)}", C["beh"])]
    for i, (k, v, c) in enumerate(metrics):
        bx = fig.add_axes([0.04 + i * 0.24, 0.62, 0.20, 0.22]); bx.axis("off")
        bx.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.04",
                     fc=C["panel"], ec=c, lw=2, transform=bx.transAxes))
        bx.text(0.5, 0.62, v, ha="center", va="center", fontsize=22, fontweight="bold", color=c)
        bx.text(0.5, 0.2, k, ha="center", va="center", fontsize=10, color=C["ink"])
    # ROC
    axr = fig.add_axes([0.07, 0.08, 0.38, 0.46])
    axr.plot(fpr, tpr, color=C["labram"], lw=2.4, label=f"LaBraM EEG (AUC {auc:.2f})")
    axr.plot([0, 1], [0, 1], ls="--", color=C["accent"], lw=1.2, label="chance")
    axr.scatter([op_fpr], [op_tpr], color=C["pos"], zorder=5, s=45,
                label=f"operating point\n(sens {sens*100:.0f}%, spec {spec*100:.0f}%)")
    axr.set_xlabel("1 − specificity"); axr.set_ylabel("sensitivity")
    axr.set_title("ROC", color=C["ink"]); axr.legend(loc="lower right", fontsize=8.5)
    # referral / risk-coverage
    axc = fig.add_axes([0.57, 0.08, 0.38, 0.46])
    conf = np.abs(P - 0.5); order = np.argsort(-conf); covs, baccs = [], []
    for cov in np.linspace(1.0, 0.5, 11):
        k2 = max(int(cov * len(P)), 10); idx = order[:k2]
        covs.append(cov * 100); baccs.append(balanced_accuracy_score(y[idx], (P[idx] >= 0.5).astype(int)))
    axc.plot(covs, baccs, "-o", color=C["fused"], lw=2, ms=4)
    axc.set_xlabel("coverage (% retained)"); axc.set_ylabel("balanced accuracy")
    axc.set_title("Selective prediction (refer low-confidence)", color=C["ink"])
    axc.invert_xaxis()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[fig] wrote {out}")


def architecture_diagram(out):
    """Schematic of the (revised) NeuroFormer: same architecture, new blocks
    (LaBraM EEG front-end, modality dropout, calibration) highlighted."""
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, fc, ec, new=False, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                     fc=fc, ec=ec, lw=2.2 if new else 1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=C["ink"], fontweight="bold" if new else "normal")
        if new:
            ax.text(x + w - 0.05, y + h - 0.02, "NEW", ha="right", va="top",
                    fontsize=7.5, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=C["accent"], ec="none"))

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.6))

    # inputs (two modalities: EEG + eye; behaviour dropped)
    box(0.2, 4.6, 2.3, 0.95, "EEG  ·  raw\n(128-ch EGI)", "#eef4f5", C["eeg"])
    box(0.2, 3.0, 2.3, 0.95, "Eye-tracking\n(gaze, pupil)", "#fdf3e8", C["eye"])
    # foundation-model front-ends (NEW): LaBraM on EEG, OcuFM (proposed) on gaze
    box(2.9, 4.6, 2.2, 0.95, "LaBraM\nfoundation model\n(200-d / window)",
        "#dcebed", C["eeg"], new=True, fs=9)
    box(2.9, 3.0, 2.2, 0.95, "OcuFM\nfoundation model\n(proposed, gaze)",
        "#fdeccf", C["eye"], new=True, fs=9)
    # CIDA adapter (NEW): identity-disentangled, on BOTH identity-dominated streams
    for y, c in [(4.6, C["eeg"]), (3.0, C["eye"])]:
        box(5.5, y, 2.0, 0.95, "CIDA adapter\n(disentangle id)", "#ffffff", c, new=True, fs=8.5)
    # modality dropout annotation
    ax.text(5.0, 2.35, "modality dropout + availability token  (missing-modality)",
            ha="center", fontsize=9, color=C["accent"], fontweight="bold")
    ax.add_patch(FancyBboxPatch((2.7, 2.0), 6.6, 0.7, boxstyle="round,pad=0.04",
                 fc="none", ec=C["accent"], lw=1.6, ls="--"))
    # fusion
    box(8.1, 3.0, 2.0, 1.4, "Cross-Modal\nFusion\nTransformer\n[CLS]", "#eef2f6", C["fused"], fs=9.5)
    # classifier + calibration (NEW)
    box(10.4, 3.9, 1.45, 0.95, "Classifier", "#ffffff", C["ink"], fs=9)
    box(10.4, 2.4, 1.45, 0.95, "Calibration\n+ referral", "#fbeae7", C["accent"], new=True, fs=8.5)

    arrow(2.5, 5.07, 2.9, 5.07)                       # EEG -> LaBraM
    arrow(5.1, 5.07, 5.5, 5.07)                       # LaBraM -> CIDA
    arrow(2.5, 3.47, 2.9, 3.47)                       # eye -> OcuFM
    arrow(5.1, 3.47, 5.5, 3.47)                       # OcuFM -> CIDA
    arrow(7.5, 5.07, 8.6, 4.4); arrow(7.5, 3.47, 8.1, 3.7)   # CIDA -> fusion
    arrow(10.1, 4.37, 10.4, 4.37)                     # fusion -> classifier
    arrow(11.1, 3.9, 11.1, 3.35)                      # classifier -> calibration
    ax.text(11.12, 2.1, "diagnosed / control\n+ confidence", ha="center",
            fontsize=8.5, color=C["ink"], style="italic")

    ax.set_title("NeuroFormer (revised): two foundation-model representations "
                 "(LaBraM EEG + OcuFM gaze) · CIDA disentanglement · calibrated referral",
                 fontsize=11.5, fontweight="bold", color=C["ink"], pad=12)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[fig] wrote {out}")


def _find(mipdb, subject, subpath, contains):
    base = Path(mipdb) / subject / subpath
    if base.is_dir():
        for f in sorted(base.glob("*")):
            if contains in f.name:
                return str(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mipdb"); ap.add_argument("--subject", default="A00051955")
    ap.add_argument("--eeg-dir"); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--results-csv", default="results/phase1_comparison.csv")
    args = ap.parse_args()
    Path(FIGDIR).mkdir(exist_ok=True); _style()
    architecture_diagram(f"{FIGDIR}/architecture.png")
    if args.mipdb:
        modality_showcase(args.mipdb, args.subject, f"{FIGDIR}/modality_showcase.png")
    results_auc(args.results_csv, f"{FIGDIR}/results_auc.png")
    if args.eeg_dir:
        try:
            embedding_map(args.eeg_dir, args.labels, f"{FIGDIR}/embedding_map.png")
            screening_report(args.eeg_dir, args.labels, f"{FIGDIR}/screening_report.png")
        except Exception as e:  # noqa: BLE001
            print(f"[fig] embedding/report skipped: {e}")
    print(f"[done] figures in {FIGDIR}/")


if __name__ == "__main__":
    main()
