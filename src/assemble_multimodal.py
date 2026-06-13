#!/usr/bin/env python
"""
Phase 2 alignment core — assemble EEG + eye + behaviour per subject.

Everything keys on **subject ID** (A00...). Each modality is reduced to one
vector per subject (mean over its blocks/windows), then all modalities + labels
are joined on ID. Because subjects are missing *different* modalities, we emit an
explicit **availability mask** so the fusion model can train with modality dropout
and run on any subset.

Modality sources:
- EEG       : per-window LaBraM .npz (dir), pooled per subject (mean / meanstd).
- eye       : data/eye_features.csv  (11 feats x ~10 task blocks/subject).
- behaviour : data/behaviour_features.csv (6 feats x task blocks/subject).

Returns, for a chosen subject set (complete-case intersection or union):
    subjects : list[str]
    y        : np.ndarray [n]            (binary any-DX-vs-none by default)
    X        : {modality: np.ndarray [n, d_mod]}  missing rows = column mean
    mask     : np.ndarray [n, n_modalities] bool  (True = modality present)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from compare_fm_vs_handcrafted import load_windows, pool_subjects

MODALITIES = ["eeg", "eye", "behaviour"]
_SID = re.compile(r"(A\d+)")


def _sid(name):
    m = _SID.match(str(name))
    return m.group(1) if m else None


def load_eye(csv_path="data/eye_features.csv"):
    """{sid: mean-over-blocks vector}, feature_names. Drops the file-name col."""
    df = pd.read_csv(csv_path)
    id_col = "file_name" if "file_name" in df.columns else df.columns[-1]
    df["_sid"] = df[id_col].map(_sid)
    feats = [c for c in df.columns if c not in (id_col, "_sid")]
    g = df.groupby("_sid")[feats].mean()
    print(f"[eye ] {len(g)} subjects x {len(feats)} feats from {csv_path}")
    return {s: g.loc[s].to_numpy(float) for s in g.index if s}, feats


def load_behaviour(csv_path="data/behaviour_features.csv"):
    """{sid: mean-over-blocks vector}, feature_names."""
    df = pd.read_csv(csv_path)
    id_col = "file" if "file" in df.columns else df.columns[0]
    df["_sid"] = df[id_col].map(_sid)
    feats = [c for c in df.columns if c not in (id_col, "_sid")]
    g = df.groupby("_sid")[feats].mean()
    print(f"[beh ] {len(g)} subjects x {len(feats)} feats from {csv_path}")
    return {s: g.loc[s].to_numpy(float) for s in g.index if s}, feats


def load_eeg(npz_dir, pool="meanstd"):
    """{sid: pooled LaBraM vector} from a dir of *_labram.npz."""
    return pool_subjects(load_windows(npz_dir, "_labram", "embeddings"), pool)


def assemble(modality_dicts, labels_df, task="binary", how="union"):
    """Align modalities + labels on subject ID.

    how="intersection" -> complete-case (subjects with ALL modalities present).
    how="union"        -> every subject with a label AND >=1 modality (missing
                          modalities are mask=False, rows filled with col mean).
    """
    lab = labels_df.set_index("ID")["DX_Status"].to_dict()
    mods = list(modality_dicts)
    have = {m: set(d) for m, d in modality_dicts.items()}

    if how == "intersection":
        subjects = set(lab)
        for s in have.values():
            subjects &= s
    elif how == "union":
        subjects = set(lab) & set().union(*have.values())
    else:
        raise ValueError(f"how must be intersection|union, got {how}")
    subjects = sorted(subjects)
    if not subjects:
        raise SystemExit("No subjects after alignment — check IDs/paths.")

    y_raw = np.array([lab[s] for s in subjects])
    y = (y_raw > 0).astype(int) if task == "binary" else y_raw.astype(int)

    X, mask = {}, np.zeros((len(subjects), len(mods)), dtype=bool)
    for j, m in enumerate(mods):
        d = modality_dicts[m]
        dim = len(next(iter(d.values())))
        mat = np.full((len(subjects), dim), np.nan)
        for i, s in enumerate(subjects):
            if s in d:
                mat[i] = d[s]
                mask[i, j] = True
        # Fill missing rows with the column mean (post-standardization this is 0;
        # the mask tells the model which rows are real).
        col_mean = np.nanmean(mat, axis=0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_mean, inds[1])
        X[m] = mat

    pos = int(y.sum()) if task == "binary" else None
    cover = ", ".join(f"{m}={int(mask[:, j].sum())}" for j, m in enumerate(mods))
    print(f"[align] how={how}: {len(subjects)} subjects | "
          f"{'positives=%d ' % pos if pos is not None else ''}| present: {cover}")
    return subjects, y, X, mask


if __name__ == "__main__":  # quick alignment sanity check (eye + behaviour + labels)
    labels_df = pd.read_csv("data/labels.csv")
    eye, _ = load_eye()
    beh, _ = load_behaviour()
    mods = {"eye": eye, "behaviour": beh}
    for how in ("union", "intersection"):
        subjects, y, X, mask = assemble(mods, labels_df, how=how)
        print(f"  -> X dims: " + ", ".join(f"{m}={v.shape}" for m, v in X.items())
              + f" | mask sums {mask.sum(axis=0).tolist()}")
