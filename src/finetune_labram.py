#!/usr/bin/env python
"""
Fine-tune LaBraM end-to-end for transdiagnostic screening (GPU).

The frozen-embedding pipeline caps at ~0.66; fine-tuning the foundation model is
the main lever for a higher number, and it pays off most with more data (scaled
HBN). Window-level training with subject-level CV (no leakage):
  - per subject -> QC-selected EEG windows (raw), label = subject diagnosis
  - train InterpolatedLaBraM on windows (optionally freeze backbone, train head)
  - evaluate per subject by mean-pooling its window probabilities -> ROC-AUC

Run inside a GPU SLURM job with the CUDA env (scripts/setup_env_gpu.sh).

FIRST validate the env fast (~1 min, fails immediately on a torch/CUDA mismatch):
    python src/finetune_labram.py --input $SCRATCH/mh/data/mipdb \
        --labels data/labels.csv --max-subj 4 --smoke

Then the full run (mixed precision on by default; --max-windows bounds memory):
    python src/finetune_labram.py \
        --input $SCRATCH/mh/data/mipdb --labels data/labels.csv \
        --all-recordings --epochs 8 --freeze-backbone --max-windows 40
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from extract_eeg_embeddings import load_and_window


def _load_labels(path):
    """Binary {subject_id: 0/1} from MIPDB (ID,DX_Status) OR HBN
    (participant_id, continuous p_factor -> median split). Auto-detects sep/cols."""
    df = pd.read_csv(path, sep=None, engine="python")
    idc = "participant_id" if "participant_id" in df.columns else "ID"
    if "DX_Status" in df.columns:
        s = df.set_index(idc)["DX_Status"]
        return {k: int(v > 0) for k, v in s.items()}
    if "p_factor" in df.columns:
        s = df.set_index(idc)["p_factor"].astype(float)
        thr = s.median()
        print(f"[labels] HBN p_factor median split at {thr:.3f}")
        return {k: int(v > thr) for k, v in s.items() if np.isfinite(v)}
    sys.exit("labels need a DX_Status or p_factor column")


def load_all(input_dir, labels_csv, match, all_recordings, window_sec=15.0, max_subj=None,
             sfreq=200.0):
    """Per-subject QC windows + labels. Returns list of (sid, windows, y), chs_info.

    sfreq resamples per backbone (LaBraM 200; CBraMod/EEGPT 250)."""
    lab = _load_labels(labels_csv)
    from pathlib import Path
    subs = sorted(p for p in Path(input_dir).iterdir() if p.is_dir())
    if max_subj:
        subs = subs[:max_subj]
    data, chs = [], None
    for p in subs:
        sid = p.name
        if sid not in lab:
            continue
        try:
            raw, w = load_and_window(p, match=match, window_sec=window_sec,
                                     all_recordings=all_recordings, target_sfreq=sfreq)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {sid}: {e}"); continue
        if chs is None:
            chs = raw.info["chs"]
        data.append((sid, w.astype(np.float32), int(lab[sid])))
        print(f"[load] {sid}: {len(w)} windows, y={int(lab[sid])}", flush=True)
    print(f"[load] {len(data)} subjects with EEG+label")
    return data, chs


HEAD_KEYS = ("head", "classif", "final", "fc_norm", "logit")
# Attribute-name patterns for the Linear layers LoRA adapts (attention + MLP).
LORA_PATTERNS = ("qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value",
                 "out_proj", "proj", "fc1", "fc2", "mlp")


def inject_lora(model, r, alpha):
    """Replace backbone nn.Linear layers (attention/MLP) with LoRALinear in place.

    The base weight stays frozen; only the rank-r A/B update trains. Returns the
    number of layers wrapped (printed so a cluster run can confirm injection took)."""
    from cida import LoRALinear
    n = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(p in child_name.lower() for p in LORA_PATTERNS):
                setattr(module, child_name,
                        LoRALinear(child.in_features, child.out_features, r=r, alpha=alpha, base=child))
                n += 1
    return n


def _build_backbone(chs, n_times, args):
    """Construct a pretrained EEG foundation backbone (n_outputs=2) per --backbone.

    All braindecode FMs share the from_pretrained pattern but differ in expected
    sfreq/channels (LaBraM 200 Hz montage-interpolated; CBraMod/EEGPT 250 Hz). The
    data sfreq/window must match (--sfreq/--window-sec); smoke-validate each on Nibi."""
    bb = getattr(args, "backbone", "labram")
    n_chans = len(chs)
    if bb == "labram":
        from braindecode.models import InterpolatedLaBraM
        return InterpolatedLaBraM.from_pretrained(
            "braindecode/labram-pretrained", chs_info=chs, n_times=n_times,
            n_outputs=2, patch_size=200, strict=False)
    if bb == "cbramod":                                   # channel-agnostic; 250 Hz
        from braindecode.models import CBraMod
        return CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained", n_outputs=2, n_chans=n_chans,
            n_times=n_times, sfreq=int(getattr(args, "sfreq", 250)))
    if bb == "eegpt":                                     # 250 Hz; channel projection
        from braindecode.models import EEGPT
        return EEGPT.from_pretrained(
            "braindecode/eegpt-pretrained", n_chans=n_chans, n_outputs=2,
            chan_proj_type="linear")
    raise SystemExit(f"unknown --backbone {bb} (labram|cbramod|eegpt)")


def build_model(chs, n_times, args, device):
    m = _build_backbone(chs, n_times, args)
    print(f"[backbone] {getattr(args, 'backbone', 'labram')}  n_chans={len(chs)}  "
          f"n_times={n_times}  total_params={sum(p.numel() for p in m.parameters()):,}", flush=True)
    bw = getattr(args, "backbone_weights", None)
    if bw:                                   # load TAPT-adapted backbone over pretrained
        sd = torch.load(bw, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[backbone] loaded adapted weights from {bw} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    if getattr(args, "lora", False):
        # PEFT: freeze ALL backbone weights, inject LoRA, keep the head trainable.
        for par in m.parameters():
            par.requires_grad = False
        n_lora = inject_lora(m, args.lora_r, args.lora_alpha)
        for name, par in m.named_parameters():
            if any(k in name.lower() for k in HEAD_KEYS):
                par.requires_grad = True
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in m.parameters())
        print(f"[lora] wrapped {n_lora} Linear layers (r={args.lora_r}); "
              f"trainable {tr:,}/{tot:,} = {100*tr/tot:.2f}%", flush=True)
        if n_lora == 0:
            sys.exit("[lora] no Linear layers matched LORA_PATTERNS — inspect "
                     "model.named_modules() on the cluster and adjust LORA_PATTERNS.")
    elif args.freeze_backbone:
        npar = 0
        for name, par in m.named_parameters():
            keep = any(k in name.lower() for k in ("head", "classif", "final", "fc", "norm"))
            par.requires_grad = keep
            npar += par.requires_grad
        if npar == 0:                          # fallback: unfreeze everything
            for par in m.parameters():
                par.requires_grad = True
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in m.parameters())
        # WARN if "frozen" accidentally became full FT (head keys didn't match this model)
        tag = "OK frozen-head" if (0 < tr < 0.5 * tot) else "WARNING: froze 0 -> training ALL"
        print(f"[freeze] trainable {tr:,}/{tot:,} = {100*tr/tot:.2f}%  [{tag}]", flush=True)
    return m.to(device)


def _cap_windows(windows, max_windows, rng):
    """Optionally subsample a subject's windows (memory bound)."""
    if max_windows and len(windows) > max_windows:
        sel = rng.choice(len(windows), max_windows, replace=False)
        return windows[sel]
    return windows


def _forward_logits(model, xb):
    out = model(xb)
    return out if torch.is_tensor(out) else out["logits"]


def run_fold(train, test, chs, args, device):
    n_times = train[0][1].shape[-1]
    model = build_model(chs, n_times, args, device)
    # window-level training tensors
    Xtr = np.concatenate([w for _, w, _ in train], 0)
    ytr = np.concatenate([[y] * len(w) for _, w, y in train])
    pos = int(ytr.sum()); neg = len(ytr) - pos
    w_ = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w_)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=1e-4)
    use_amp = (device == "cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    idx = np.arange(len(Xtr))
    model.train()
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(ytr[b]).long().to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = lossf(_forward_logits(model, xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
    # subject-level eval: mean window prob
    model.eval(); ys, ps, preds = [], [], []
    with torch.no_grad():
        for sid, w, y in test:
            x = torch.from_numpy(w).to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = _forward_logits(model, x)
            p = torch.softmax(logits.float(), -1)[:, 1].mean().item()
            ys.append(y); ps.append(p); preds.append((sid, y, p))
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else np.nan
    return auc, preds


def smoke_test(data, chs, args, device):
    """Fast pass/fail: build the model and run one train step + one eval forward
    on a tiny batch. Validates the CUDA env in ~1 min before the full run."""
    print("[smoke] building model + one fwd/bwd step...", flush=True)
    n_times = data[0][1].shape[-1]
    model = build_model(chs, n_times, args, device)
    use_amp = (device == "cuda") and not args.no_amp
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    xb = torch.from_numpy(data[0][1][: min(4, len(data[0][1]))]).to(device)
    yb = torch.zeros(xb.size(0), dtype=torch.long, device=device)
    model.train(); opt.zero_grad()
    with torch.autocast(device_type="cuda", enabled=use_amp):
        loss = nn.functional.cross_entropy(_forward_logits(model, xb), yb)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] OK  loss={loss.item():.3f}  peak GPU mem={mem:.2f} GB  amp={use_amp}")
    else:
        print(f"[smoke] OK on CPU  loss={loss.item():.3f}")
    print("[smoke] CUDA env is good — safe to launch the full run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--match", default=None); ap.add_argument("--all-recordings", action="store_true")
    ap.add_argument("--backbone", default="labram", choices=["labram", "cbramod", "eegpt"],
                    help="pretrained EEG foundation backbone (C1: test the freeze finding "
                         "across FMs, not just LaBraM)")
    ap.add_argument("--sfreq", type=float, default=200.0,
                    help="resample rate; LaBraM=200, CBraMod/EEGPT=250")
    ap.add_argument("--window-sec", type=float, default=15.0, help="window length (s)")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--lora", action="store_true",
                    help="PEFT: freeze backbone, train only rank-r LoRA updates + head "
                         "(the parameter-efficient middle ground between frozen and full FT)")
    ap.add_argument("--lora-r", type=int, default=8); ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--backbone-weights", default=None,
                    help="load a TAPT-adapted backbone state_dict over the pretrained "
                         "weights (src/tapt_pretrain.py output) before freeze/readout")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-subj", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap windows per subject (memory bound)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--smoke", action="store_true",
                    help="build + one fwd/bwd step only, to validate the CUDA env, then exit")
    ap.add_argument("--out", default="results/finetune_labram.csv")
    ap.add_argument("--save-preds", default=None,
                    help="CSV path to dump per-subject out-of-fold predictions "
                         "(repeat, subject, y_true, p) for calibration/operating-point analysis")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Startup diagnostics — surfaces torch/CUDA mismatches immediately.
    print(f"[env] torch {torch.__version__}  cuda_build={torch.version.cuda}  "
          f"cuda_available={torch.cuda.is_available()}", flush=True)
    if device == "cuda":
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}  "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)
    else:
        print("[warn] no GPU — fine-tuning on CPU will be very slow.")

    rng = np.random.default_rng(args.seed)
    data, chs = load_all(args.input, args.labels, args.match, args.all_recordings,
                         window_sec=args.window_sec, max_subj=args.max_subj, sfreq=args.sfreq)
    if args.max_windows:
        data = [(sid, _cap_windows(w, args.max_windows, rng), y) for sid, w, y in data]
    if not data:
        sys.exit("no subjects with EEG+label loaded")
    if args.smoke:
        smoke_test(data, chs, args, device)
        return
    y = np.array([d[2] for d in data])
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    aucs = []
    pred_rows = []
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), 1):
        print(f"[fold {k}/{args.folds*args.repeats}] fine-tuning "
              f"(freeze_backbone={args.freeze_backbone})...", flush=True)
        auc, preds = run_fold([data[i] for i in tr], [data[i] for i in te], chs, args, device)
        aucs.append(auc); print(f"[fold {k}] AUC={auc:.3f}", flush=True)
        repeat = (k - 1) // args.folds        # 0-based repeat index
        for sid, yt, p in preds:
            pred_rows.append({"repeat": repeat, "subject": sid, "y_true": yt, "p": p})
    aucs = np.array([a for a in aucs if not np.isnan(a)])
    ci = 1.96 * aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else float("nan")
    print(f"\n[finetune] {args.folds}x{args.repeats} CV ROC-AUC = {aucs.mean():.3f} "
          f"+/- {ci:.3f}  (refs: pooled-logreg readout ~0.66; frozen head-only ~0.76)")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"auc": aucs}).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")
    if args.save_preds:
        Path(args.save_preds).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(args.save_preds, index=False)
        print(f"[done] wrote {args.save_preds} ({len(pred_rows)} per-subject OOF preds)")


if __name__ == "__main__":
    main()
