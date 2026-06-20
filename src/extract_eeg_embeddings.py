#!/usr/bin/env python
"""
Phase 1 — extract frozen LaBraM embeddings from MIPDB EEG.

Pipeline: load raw EEG (MNE) -> LaBraM-spec preprocessing (0.1-75 Hz band-pass,
line-noise notch, resample to 200 Hz) -> non-overlapping windows scaled to
microvolts (LaBraM's trained unit) -> frozen LaBraM forward -> per-window
embeddings cached to .npz.

Run the data path first WITHOUT a model to validate loading/preprocessing:

    python src/extract_eeg_embeddings.py --input <eeg_file_or_subject_dir> --dry-run

Then extract embeddings (pretrained weights auto-download from Hugging Face;
run on a login node the first time so they cache):

    python src/extract_eeg_embeddings.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/ \
        --output $SCRATCH/mh/data/processed/<subject>_labram.npz

Whole cohort (one .npz per subject; failures are logged and skipped, and
already-extracted subjects are skipped unless --overwrite):

    python src/extract_eeg_embeddings.py --batch \
        --input      $SCRATCH/mh/data/mipdb/ \
        --output-dir $SCRATCH/mh/data/processed/

Notes / assumptions (verify for your data):
- MIPDB was recorded in the US (CMI, NY) on a 128-ch EGI net at 500 Hz, so mains
  line noise is **60 Hz** (not 50). We notch 60 Hz here; change with --notch.
- LaBraM expects 200 Hz input and a 200-sample (1 s) patch size. We map our 129
  EGI channels onto LaBraM's canonical 128 via braindecode InterpolatedLaBraM,
  which needs electrode positions (set from the GSN-HydroCel-129 montage).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ----------------------------- preprocessing -----------------------------

def load_raw(input_path: str):
    """Load an EEG file with MNE, auto-detecting common EGI/BIDS formats."""
    import mne

    p = Path(input_path)
    if p.is_dir():
        # Pick the LARGEST EEG file in a subject dir: the main recording, not a
        # short calibration block (alphabetically-first was sometimes too short
        # to make even one window).
        cands = [*p.rglob("*.raw"), *p.rglob("*.mff"), *p.rglob("*.edf"),
                 *p.rglob("*.set"), *p.rglob("*.fif")]
        if not cands:
            sys.exit(f"No EEG files (.raw/.mff/.edf/.set/.fif) found under {p}")
        p = max(cands, key=lambda f: f.stat().st_size)
        print(f"[load] directory given; using largest EEG file: {p}")

    suffix = p.suffix.lower()
    readers = {
        ".raw":  mne.io.read_raw_egi,
        ".mff":  mne.io.read_raw_egi,
        ".edf":  mne.io.read_raw_edf,
        ".set":  mne.io.read_raw_eeglab,
        ".fif":  mne.io.read_raw_fif,
    }
    if suffix not in readers:
        sys.exit(f"Unsupported EEG format: {suffix}")
    raw = readers[suffix](str(p), preload=True, verbose="ERROR")
    print(f"[load] {p.name}: {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']} Hz, "
          f"{raw.n_times / raw.info['sfreq']:.1f} s")
    return raw


def preprocess(raw, l_freq=0.1, h_freq=75.0, notch=60.0, target_sfreq=200.0):
    """Apply LaBraM-spec preprocessing in place and return the Raw object."""
    import mne

    raw.pick("eeg")                                   # drop non-EEG channels
    # MIPDB EGI files name the vertex reference 'E129'; the GSN-HydroCel-129
    # montage calls it 'Cz'. Rename so every channel gets a 3D position
    # (otherwise the missing position is NaN and the sphere fit fails).
    if "E129" in raw.ch_names:
        raw.rename_channels({"E129": "Cz"})
    # Attach 3D electrode positions so InterpolatedLaBraM can map our montage
    # onto LaBraM's canonical channels. MIPDB uses the EGI 128 HydroCel net.
    try:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        raw.set_montage(montage, on_missing="ignore", match_case=False,
                        verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        print(f"[prep] montage note: {type(e).__name__}: {e}")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    if notch:
        raw.notch_filter(freqs=[notch], verbose="ERROR")
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose="ERROR")
    print(f"[prep] band-pass {l_freq}-{h_freq} Hz, notch {notch} Hz, "
          f"resampled to {target_sfreq} Hz -> {len(raw.ch_names)} EEG ch")
    return raw


def make_windows(raw, window_sec=1.0, clip_uv=800.0):
    """Split into non-overlapping windows -> array [n_windows, n_ch, n_samples].

    Data is scaled volts->uV; values are clipped to +/-clip_uv to bound gross
    artifacts (set clip_uv=0/None to disable).
    """
    sfreq = raw.info["sfreq"]
    win = int(round(window_sec * sfreq))
    data = raw.get_data()                              # [n_ch, n_times], VOLTS
    # LaBraM was trained on EEG in microvolts (braindecode applies no internal
    # scaling — it feeds x straight to the patch embedding). MNE returns volts,
    # so without this *1e6 the frozen model sees ~1e-5 amplitudes and produces
    # meaningless embeddings with no crash/warning. Convert volts -> uV.
    data = data * 1e6
    # Robust amplitude diagnostic. Real EEG bulk is ~tens of uV; min/max alone
    # is misleading because a few bad peripheral electrodes / blinks spike to
    # thousands. The MEDIAN |x| and 95th percentile reveal the true scale; a
    # large gap to max => outlier channels, not a global scaling error.
    absd = np.abs(data)
    med = np.median(absd)
    p95, p99 = np.percentile(absd, [95, 99.5])
    ch_std = data.std(axis=1)
    n_hot = int((ch_std > 200).sum())                  # channels >200uV std
    print(f"[win ] uV amplitude: median|x|={med:.1f}  p95={p95:.1f}  "
          f"p99.5={p99:.1f}  max={absd.max():.1f}  "
          f"(expect median ~5-30 uV); hot channels (std>200uV)={n_hot}/{data.shape[0]}")
    # Gentle artifact clip. The median (~14 uV) confirms correct scaling, but a
    # handful of bad electrodes / blinks spike to thousands of uV. Hjorth
    # activity (variance) and the frozen FM are both distorted by those, so clip
    # to a physiological bound (>p99.5, so real EEG is untouched; only gross
    # artifacts are capped). Applied to BOTH feature sets via make_windows.
    if clip_uv:
        clipped = float((absd > clip_uv).mean()) * 100.0
        data = np.clip(data, -clip_uv, clip_uv)
        print(f"[win ] clipped to +/-{clip_uv:.0f} uV ({clipped:.3f}% of samples)")
    n_ch, n_times = data.shape
    n_win = n_times // win
    if n_win == 0:
        raise ValueError(f"Recording shorter than one {window_sec}s window.")
    windows = data[:, : n_win * win].reshape(n_ch, n_win, win).transpose(1, 0, 2)
    print(f"[win ] {n_win} windows of {win} samples ({window_sec}s), "
          f"shape {windows.shape}")
    return windows.astype(np.float32)


# ------------------------------ LaBraM ------------------------------------

def extract_labram_embeddings(windows, chs_info, batch_size=8, device="cuda"):
    """Run frozen InterpolatedLaBraM over windows -> [n_windows, embed_dim].

    Uses braindecode's InterpolatedLaBraM, which prepends a frozen spatial-
    interpolation layer mapping our montage (chs_info, with 3D positions) onto
    LaBraM's canonical 128 channels, and loads the pretrained weights from the
    Hugging Face Hub. Requires internet (run on a login node the first time so
    weights cache to ~/.cache/huggingface).
    """
    import torch
    from braindecode.models import InterpolatedLaBraM

    n_times = windows.shape[-1]
    if n_times % 200 != 0:
        print(f"[labram] warning: n_times={n_times} is not a multiple of the "
              f"200-sample patch size; consider --window-sec in steps of 1.0.")

    model = InterpolatedLaBraM.from_pretrained(
        "braindecode/labram-pretrained",
        chs_info=chs_info,
        n_times=n_times,
        n_outputs=2,            # arbitrary; we read features, not the head
        patch_size=200,
        strict=False,           # interpolation matrix isn't in the checkpoint
    )
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print(f"[labram] InterpolatedLaBraM loaded on {device}; "
          f"input {windows.shape}")

    embs = []
    with torch.no_grad():
        x = torch.from_numpy(windows).float()              # keep on CPU/host
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size].to(device)          # move ONLY this batch
            out = model(xb, return_features=True)
            feats = out.get("cls_token")
            if feats is None:                               # fall back to pooled tokens
                feats = out["features"]
                if feats.ndim == 3:
                    feats = feats.mean(dim=1)
            embs.append(feats.cpu().numpy())
            del xb, out                                     # free promptly (many windows)
    out = np.concatenate(embs, axis=0)
    print(f"[labram] embeddings: {out.shape}")
    return out


# ------------------------------ pipeline ----------------------------------

# Quality gate for picking a recording: median |x| must look like real EEG
# (not flat/dead, not an impedance/calibration file at hundreds-thousands uV),
# and not be mostly artifact (clipped).
QC_MEDIAN_UV = (3.0, 150.0)
QC_MAX_CLIP_FRAC = 0.15


def list_eeg_files(input_path, match=None):
    """EEG file(s) for a path: the file itself, or a subject dir's candidates
    sorted largest-first (the main recording tends to be the biggest).
    If `match` is given, keep only files whose name contains it (e.g. a BIDS task
    like 'RestingState') — used to pick a consistent paradigm across subjects."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    cands = [*p.rglob("*.raw"), *p.rglob("*.mff"), *p.rglob("*.edf"),
             *p.rglob("*.set"), *p.rglob("*.fif")]
    if match:
        cands = [f for f in cands if match in f.name]
    return sorted(cands, key=lambda f: f.stat().st_size, reverse=True)


def load_and_window(input_path, notch=60.0, window_sec=15.0, clip_uv=800.0,
                    max_candidates=6, match=None, all_recordings=False, target_sfreq=200.0):
    """Window a subject's EEG.

    Default: pick the best-quality single recording (first QC-passing candidate).
    all_recordings=True: aggregate windows from EVERY QC-passing recording (e.g.
    all MIPDB paradigms) -> more windows / brain states per subject, a richer
    representation. Junk files (flat/dead or impedance/calibration at thousands of
    uV) are dropped by QC. Raises if no candidate qualifies. Returns (raw, windows).
    """
    cands = list_eeg_files(input_path, match=match)
    if not cands:
        raise ValueError(f"No EEG files"
                         + (f" matching '{match}'" if match else "")
                         + f" under {input_path}")
    cap = 15 if all_recordings else max_candidates
    reason = "no candidates tried"
    kept_windows, kept_raw = [], None
    for f in cands[:cap]:
        raw = preprocess(load_raw(f), notch=notch, target_sfreq=target_sfreq)
        try:
            windows = make_windows(raw, window_sec=window_sec, clip_uv=clip_uv)
        except ValueError as e:                       # too short -> try next file
            reason = str(e)
            continue
        absd = np.abs(windows)
        med = float(np.median(absd))
        clip_frac = float((absd >= clip_uv - 1e-6).mean())
        if QC_MEDIAN_UV[0] <= med <= QC_MEDIAN_UV[1] and clip_frac < QC_MAX_CLIP_FRAC:
            print(f"[qc  ] accepted {f.name}: median|x|={med:.1f} uV, "
                  f"clip={clip_frac*100:.2f}%, {len(windows)} windows")
            if not all_recordings:
                return raw, windows
            kept_windows.append(windows)
            if kept_raw is None:
                kept_raw = raw
            continue
        reason = (f"median|x|={med:.1f} uV, clip={clip_frac*100:.1f}% "
                  f"(QC needs {QC_MEDIAN_UV} uV, <{QC_MAX_CLIP_FRAC*100:.0f}% clip)")
        print(f"[qc  ] rejected {f.name}: {reason}")
    if all_recordings and kept_windows:
        allw = np.concatenate(kept_windows, axis=0)
        print(f"[qc  ] aggregated {len(kept_windows)} recordings -> {len(allw)} windows")
        return kept_raw, allw
    raise ValueError(f"No physiological-quality recording among "
                     f"{min(len(cands), cap)} candidates: {reason}")


def extract_for_input(input_path, notch=60.0, window_sec=15.0, dry_run=False,
                      match=None, all_recordings=False):
    """Run the full QC-select -> LaBraM pipeline for one subject.

    Returns the [n_windows, embed_dim] embeddings, or None on --dry-run.
    """
    raw, windows = load_and_window(input_path, notch=notch, window_sec=window_sec,
                                   match=match, all_recordings=all_recordings)
    if dry_run:
        print("[dry-run] OK — data path validated; no embeddings extracted.")
        return None
    return extract_labram_embeddings(windows, raw.info["chs"])


# -------------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract frozen LaBraM EEG embeddings.")
    ap.add_argument("--input", required=True,
                    help="EEG file or subject dir; with --batch, a PARENT dir "
                         "whose immediate subdirectories are per-subject folders.")
    ap.add_argument("--output", help="Output .npz (single-subject; or use --dry-run)")
    ap.add_argument("--batch", action="store_true",
                    help="Treat --input as a parent dir; extract every subject "
                         "subdirectory to --output-dir/<subject>_labram.npz.")
    ap.add_argument("--output-dir", help="Output directory for --batch mode.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract subjects whose .npz already exists (--batch).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel subject workers in --batch mode. Subjects are "
                         "independent; set ~= the cores you requested (e.g. 8).")
    ap.add_argument("--notch", type=float, default=60.0, help="Line-noise Hz (US=60)")
    ap.add_argument("--window-sec", type=float, default=15.0,
                    help="Window length in seconds. 15 s = 3000 samples @200Hz, "
                         "which matches the pretrained LaBraM temporal embedding.")
    ap.add_argument("--match", default=None,
                    help="Only consider files whose name contains this (e.g. a "
                         "BIDS task like 'RestingState') — pick a consistent paradigm.")
    ap.add_argument("--all-recordings", action="store_true",
                    help="Aggregate windows from ALL QC-passing recordings per "
                         "subject (e.g. all paradigms) for a richer representation.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load + preprocess + window only; skip the model.")
    args = ap.parse_args()

    if args.batch:
        run_batch(args)
        return

    embs = extract_for_input(args.input, notch=args.notch,
                             window_sec=args.window_sec, dry_run=args.dry_run,
                             match=args.match, all_recordings=args.all_recordings)
    if args.dry_run:
        return
    if not args.output:
        sys.exit("Provide --output (or use --dry-run).")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embs)
    print(f"[done] saved {embs.shape} -> {args.output}")


def _extract_one(task):
    """Worker: extract + save one subject. Module-level so it is picklable.

    Pin torch to 1 thread so N parallel workers don't oversubscribe the cores
    (each subject's forward is tiny; the win is across subjects, not within).
    """
    subj_path, out_path, notch, window_sec, match, all_recordings = task
    name = Path(subj_path).name
    try:
        import torch
        torch.set_num_threads(1)
        embs = extract_for_input(subj_path, notch=notch, window_sec=window_sec,
                                 match=match, all_recordings=all_recordings)
        np.savez_compressed(out_path, embeddings=embs)
        return name, "done", str(embs.shape)
    except SystemExit as e:               # load/window guards call sys.exit
        return name, "failed", f"skipped: {e}"
    except Exception as e:                # noqa: BLE001 - keep the cohort going
        return name, "failed", f"{type(e).__name__}: {e}"


def run_batch(args):
    """Extract every subject subdirectory under --input; one .npz each.

    Failures are logged and skipped so one bad subject can't abort the cohort.
    Already-extracted subjects are skipped unless --overwrite. With --jobs>1,
    subjects are processed in parallel (they are independent).
    """
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
        out_path = out_dir / f"{subj.name}_labram.npz"
        if out_path.exists() and not args.overwrite:
            skipped.append(subj.name)
            continue
        tasks.append((str(subj), str(out_path), args.notch, args.window_sec,
                      args.match, args.all_recordings))
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
        from concurrent.futures import (ProcessPoolExecutor, as_completed,
                                         BrokenExecutor)
        try:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(_extract_one, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    record(i, *fut.result())
        except BrokenExecutor:
            sys.exit("\n[batch] FATAL: a worker was killed (likely OOM). "
                     "Lower --jobs or raise the job's memory, then re-run "
                     "(finished subjects are skipped).")

    print(f"\n[batch] done={len(done)} skipped(existing)={len(skipped)} "
          f"failed={len(failed)}")
    if failed:
        print(f"[batch] failed subjects: {', '.join(failed)}")


if __name__ == "__main__":
    main()
