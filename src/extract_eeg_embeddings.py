#!/usr/bin/env python
"""
Phase 1 — extract frozen LaBraM embeddings from MIPDB EEG.

Pipeline: load raw EEG (MNE) -> LaBraM-spec preprocessing (0.1-75 Hz band-pass,
line-noise notch, resample to 200 Hz) -> non-overlapping 1 s windows -> frozen
LaBraM forward -> per-window embeddings cached to .npz.

Run the data path first WITHOUT a model to validate loading/preprocessing:

    python src/extract_eeg_embeddings.py --input <eeg_file_or_subject_dir> --dry-run

Then extract embeddings once you have a LaBraM checkpoint:

    python src/extract_eeg_embeddings.py \
        --input  $SCRATCH/mh/data/mipdb/<subject>/<...>.raw \
        --output $SCRATCH/mh/data/processed/<subject>_labram.npz \
        --checkpoint /path/to/labram-base.pth

Notes / assumptions (verify for your data):
- MIPDB was recorded in the US (CMI, NY) on a 128-ch EGI net at 500 Hz, so mains
  line noise is **60 Hz** (not 50). We notch 60 Hz here; change with --notch.
- LaBraM expects ~200 Hz input and patches of 1 s (200 samples). See
  https://github.com/935963004/LaBraM. The embedding call is isolated in
  extract_labram_embeddings() so you can wire it to your exact checkpoint API.
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
    raw.pick("eeg")                                   # drop non-EEG channels
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

def extract_labram_embeddings(windows, checkpoint, device="cuda"):
    """Run frozen LaBraM over windows -> [n_windows, embed_dim].

    Isolated so you can adapt it to your exact checkpoint. Default implementation
    targets the official repo (https://github.com/935963004/LaBraM): build the
    base model, load weights, run the encoder, and mean-pool the patch tokens.
    Verify the model builder / forward signature against your checkpoint version.
    """
    import torch

    # --- build model (ADAPT to your LaBraM install) ---------------------
    # Official repo route (expects LaBraM on PYTHONPATH):
    #   from modeling_finetune import labram_base_patch200_200
    #   model = labram_base_patch200_200(num_classes=0)        # 0 -> no head
    # braindecode route:
    #   from braindecode.models import Labram
    #   model = Labram(n_times=windows.shape[-1], n_chans=windows.shape[1], ...)
    try:
        from modeling_finetune import labram_base_patch200_200  # type: ignore
        model = labram_base_patch200_200(num_classes=0)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            "Could not import the LaBraM model builder.\n"
            f"  ({type(e).__name__}: {e})\n"
            "Clone https://github.com/935963004/LaBraM and add it to PYTHONPATH,\n"
            "or switch to the braindecode route in extract_labram_embeddings()."
        )

    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state)                  # unwrap common wrappers
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[labram] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys (ok if just the head).")

    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    embs = []
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)       # [N, C, T]
        for i in range(0, len(x), 64):                 # batch to bound memory
            batch = x[i : i + 64]
            feats = model.forward_features(batch)       # [B, tokens, D] or [B, D]
            if feats.ndim == 3:
                feats = feats.mean(dim=1)               # mean-pool patch tokens
            embs.append(feats.cpu().numpy())
    out = np.concatenate(embs, axis=0)
    print(f"[labram] embeddings: {out.shape}")
    return out


# -------------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract frozen LaBraM EEG embeddings.")
    ap.add_argument("--input", required=True, help="EEG file or subject directory")
    ap.add_argument("--output", help="Output .npz (required unless --dry-run)")
    ap.add_argument("--checkpoint", help="LaBraM weights (.pth)")
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
    if not args.output or not args.checkpoint:
        sys.exit("Provide --output and --checkpoint (or use --dry-run).")

    embs = extract_labram_embeddings(windows, args.checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embs)
    print(f"[done] saved {embs.shape} -> {args.output}")


if __name__ == "__main__":
    main()
