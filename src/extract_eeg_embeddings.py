#!/usr/bin/env python
"""
Phase 1 — extract frozen LaBraM embeddings from MIPDB EEG.

Pipeline: load raw EEG (MNE) -> LaBraM-spec preprocessing (0.1-75 Hz band-pass,
line-noise notch, resample to 200 Hz) -> non-overlapping 1 s windows -> frozen
LaBraM forward -> per-window embeddings cached to .npz.

Run the data path first WITHOUT a model to validate loading/preprocessing:

    python src/extract_eeg_embeddings.py --input <eeg_file_or_subject_dir> --dry-run

Then extract embeddings (pretrained weights auto-download from Hugging Face;
run on a login node the first time so they cache):

    python src/extract_eeg_embeddings.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/ \
        --output $SCRATCH/mh/data/processed/<subject>_labram.npz

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
        # pick the first readable EEG file in a subject dir
        cands = sorted([*p.rglob("*.raw"), *p.rglob("*.mff"), *p.rglob("*.edf"),
                        *p.rglob("*.set"), *p.rglob("*.fif")])
        if not cands:
            sys.exit(f"No EEG files (.raw/.mff/.edf/.set/.fif) found under {p}")
        p = cands[0]
        print(f"[load] directory given; using first EEG file: {p}")

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


def make_windows(raw, window_sec=1.0):
    """Split into non-overlapping windows -> array [n_windows, n_ch, n_samples]."""
    sfreq = raw.info["sfreq"]
    win = int(round(window_sec * sfreq))
    data = raw.get_data()                              # [n_ch, n_times], volts
    n_ch, n_times = data.shape
    n_win = n_times // win
    if n_win == 0:
        sys.exit(f"Recording shorter than one {window_sec}s window.")
    windows = data[:, : n_win * win].reshape(n_ch, n_win, win).transpose(1, 0, 2)
    print(f"[win ] {n_win} windows of {win} samples ({window_sec}s), "
          f"shape {windows.shape}")
    return windows.astype(np.float32)


# ------------------------------ LaBraM ------------------------------------

def extract_labram_embeddings(windows, chs_info, batch_size=32, device="cuda"):
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
        x = torch.from_numpy(windows).float().to(device)   # [N, C, T]
        for i in range(0, len(x), batch_size):
            out = model(x[i : i + batch_size], return_features=True)
            feats = out.get("cls_token")
            if feats is None:                               # fall back to pooled tokens
                feats = out["features"]
                if feats.ndim == 3:
                    feats = feats.mean(dim=1)
            embs.append(feats.cpu().numpy())
    out = np.concatenate(embs, axis=0)
    print(f"[labram] embeddings: {out.shape}")
    return out


# -------------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract frozen LaBraM EEG embeddings.")
    ap.add_argument("--input", required=True, help="EEG file or subject directory")
    ap.add_argument("--output", help="Output .npz (required unless --dry-run)")
    ap.add_argument("--notch", type=float, default=60.0, help="Line-noise Hz (US=60)")
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Load + preprocess + window only; skip the model.")
    args = ap.parse_args()

    raw = load_raw(args.input)
    raw = preprocess(raw, notch=args.notch)
    windows = make_windows(raw, window_sec=args.window_sec)

    if args.dry_run:
        print("[dry-run] OK — data path validated; no embeddings extracted.")
        return
    if not args.output:
        sys.exit("Provide --output (or use --dry-run).")

    embs = extract_labram_embeddings(windows, raw.info["chs"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embs)
    print(f"[done] saved {embs.shape} -> {args.output}")


if __name__ == "__main__":
    main()
