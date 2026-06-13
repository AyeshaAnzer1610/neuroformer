#!/usr/bin/env python
"""
Phase 1 baseline — hand-crafted EEG features (the pre-foundation-model method).

Re-implements the original Multisource EEG feature set so the Phase 1 gate is an
honest head-to-head against frozen LaBraM embeddings on the *same* raw EEG, same
subjects, same windows, same CV — only the representation differs. The original
pipeline (EEGFinal.ipynb) computed, after band-pass + normalization, per channel:
**Hjorth activity, mobility, complexity** and **spectral entropy** (from a Welch
PSD). We reproduce exactly those four features per channel here.

We deliberately reuse load_raw/preprocess/make_windows from the LaBraM extractor
so both feature sets see identical preprocessing and identical windows.

Single subject:
    python src/extract_handcrafted_eeg.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/ \
        --output $SCRATCH/mh/data/processed/<subject>_handcrafted.npz

Whole cohort (one .npz per subject; failures logged and skipped):
    python src/extract_handcrafted_eeg.py --batch \
        --input      $SCRATCH/mh/data/mipdb/ \
        --output-dir $SCRATCH/mh/data/processed_handcrafted/

Output .npz holds:
- features  : [n_windows, n_ch * 4] float32 (per-channel Hjorth x3 + spec-entropy)
- feature_names : [n_ch * 4] str, e.g. "E12_hjorth_mobility"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse the exact same load/preprocess/window pipeline as the LaBraM extractor
# so the two feature sets are computed on identical inputs.
from extract_eeg_embeddings import load_raw, preprocess, make_windows


# --------------------------- feature functions ----------------------------

def hjorth(x, axis=-1):
    """Hjorth activity, mobility, complexity along `axis`.

    activity   = var(x)
    mobility   = sqrt(var(x') / var(x))
    complexity = mobility(x') / mobility(x)
    """
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    var_x = np.var(x, axis=axis)
    var_dx = np.var(dx, axis=axis)
    var_ddx = np.var(ddx, axis=axis)
    eps = 1e-12
    mobility = np.sqrt(var_dx / (var_x + eps))
    mob_dx = np.sqrt(var_ddx / (var_dx + eps))
    complexity = mob_dx / (mobility + eps)
    return var_x, mobility, complexity


def spectral_entropy(x, sfreq, axis=-1):
    """Normalized Shannon entropy of the Welch power spectral density."""
    from scipy.signal import welch
    # nperseg capped to the window length; default Welch segmenting otherwise.
    nperseg = min(256, x.shape[axis])
    _, psd = welch(x, fs=sfreq, nperseg=nperseg, axis=axis)
    psd = psd + 1e-12
    psd = psd / psd.sum(axis=axis, keepdims=True)         # normalize to a pmf
    ent = -np.sum(psd * np.log2(psd), axis=axis)
    ent = ent / np.log2(psd.shape[axis])                  # 0..1 normalized
    return ent


def handcrafted_features(windows, ch_names, sfreq):
    """windows [n_win, n_ch, n_samp] -> features [n_win, n_ch*4] + names."""
    activity, mobility, complexity = hjorth(windows, axis=-1)   # each [n_win, n_ch]
    sent = spectral_entropy(windows, sfreq, axis=-1)            # [n_win, n_ch]
    # Stack so each channel's 4 features are contiguous: [n_win, n_ch, 4] -> flat
    stacked = np.stack([activity, mobility, complexity, sent], axis=-1)
    n_win, n_ch, _ = stacked.shape
    feats = stacked.reshape(n_win, n_ch * 4).astype(np.float32)
    kinds = ["hjorth_activity", "hjorth_mobility", "hjorth_complexity",
             "spectral_entropy"]
    names = [f"{ch}_{k}" for ch in ch_names for k in kinds]
    print(f"[hand] features {feats.shape} ({n_ch} ch x 4)")
    return feats, np.array(names)


# ------------------------------- pipeline ---------------------------------

def extract_for_input(input_path, notch=60.0, window_sec=15.0, dry_run=False):
    raw = load_raw(input_path)
    raw = preprocess(raw, notch=notch)
    windows = make_windows(raw, window_sec=window_sec)
    if dry_run:
        print("[dry-run] OK — data path validated; no features extracted.")
        return None, None
    return handcrafted_features(windows, raw.ch_names, raw.info["sfreq"])


def save_npz(out_path, feats, names):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feats, feature_names=names)


# --------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract hand-crafted EEG features "
                                             "(Hjorth + spectral entropy).")
    ap.add_argument("--input", required=True,
                    help="EEG file or subject dir; with --batch, a parent dir.")
    ap.add_argument("--output", help="Output .npz (single subject).")
    ap.add_argument("--batch", action="store_true",
                    help="Treat --input as a parent dir of subject folders.")
    ap.add_argument("--output-dir", help="Output directory for --batch mode.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract subjects whose .npz already exists (--batch).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel subject workers in --batch mode (subjects are "
                         "independent; set ~= requested cores).")
    ap.add_argument("--notch", type=float, default=60.0, help="Line-noise Hz (US=60)")
    ap.add_argument("--window-sec", type=float, default=15.0,
                    help="Window length (s); matches the LaBraM extractor's default.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load + preprocess + window only; skip feature computation.")
    args = ap.parse_args()

    if args.batch:
        run_batch(args)
        return

    feats, names = extract_for_input(args.input, notch=args.notch,
                                     window_sec=args.window_sec,
                                     dry_run=args.dry_run)
    if args.dry_run:
        return
    if not args.output:
        sys.exit("Provide --output (or use --dry-run).")
    save_npz(args.output, feats, names)
    print(f"[done] saved {feats.shape} -> {args.output}")


def _extract_one(task):
    """Worker: extract + save one subject's hand-crafted features. Picklable."""
    subj_path, out_path, notch, window_sec = task
    name = Path(subj_path).name
    try:
        feats, names = extract_for_input(subj_path, notch=notch,
                                         window_sec=window_sec)
        save_npz(out_path, feats, names)
        return name, "done", str(feats.shape)
    except SystemExit as e:
        return name, "failed", f"skipped: {e}"
    except Exception as e:  # noqa: BLE001 - keep the cohort going
        return name, "failed", f"{type(e).__name__}: {e}"


def run_batch(args):
    parent = Path(args.input)
    if not parent.is_dir():
        sys.exit(f"--batch needs a directory; got {parent}")
    if not args.output_dir:
        sys.exit("--batch requires --output-dir.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p for p in parent.iterdir() if p.is_dir())
    if not subjects:
        sys.exit(f"No subject subdirectories found under {parent}")

    tasks, skipped = [], []
    for subj in subjects:
        out_path = out_dir / f"{subj.name}_handcrafted.npz"
        if out_path.exists() and not args.overwrite:
            skipped.append(subj.name)
            continue
        tasks.append((str(subj), str(out_path), args.notch, args.window_sec))
    print(f"[batch] {len(subjects)} subjects: {len(tasks)} to extract, "
          f"{len(skipped)} already done | jobs={args.jobs}")

    done, failed = [], []

    def record(i, name, status, msg):
        tag = "saved" if status == "done" else "FAILED"
        print(f"[batch {i}/{len(tasks)}] {name}: {tag} {msg}")
        (done if status == "done" else failed).append(name)

    if args.jobs <= 1:
        for i, task in enumerate(tasks, 1):
            record(i, *_extract_one(task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_extract_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                record(i, *fut.result())

    print(f"\n[batch] done={len(done)} skipped(existing)={len(skipped)} "
          f"failed={len(failed)}")
    if failed:
        print(f"[batch] failed subjects: {', '.join(failed)}")


if __name__ == "__main__":
    main()
