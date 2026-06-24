#!/usr/bin/env python
"""
NeuroFormer behaviour features from raw MIPDB — RT + accuracy on the Contrast
Change Detection task (SAIIT_2AFC, a 2-alternative forced-choice perceptual
decision). Produces the original 6 behaviour stats per subject:
  [rt_mean, rt_std, rt_median, rt_min, rt_max, accuracy]
saved per subject as <sid>_behaviour.npz so it pools per subject like EEG/eye.

Behaviour is NOT in the laptop repo — run this on Nibi where raw MIPDB lives
($SCRATCH/mh/data/mipdb/<subject>/), same as the eye extractor.

STEP 1 — confirm the raw layout (so the parser targets the right file):
    python src/extract_behaviour.py --probe --input $SCRATCH/mh/data/mipdb/A00051826/
  This prints every non-Samples file under the subject and previews likely
  behaviour/event sources (Events.txt, *.csv, behavioural logs). Paste that back
  so the message/column mapping below can be finalized if needed.

STEP 2 — extract one subject / whole cohort:
    python src/extract_behaviour.py --input  $SCRATCH/mh/data/mipdb/A00051826/ \
                                    --output $SCRATCH/mh/data/processed_beh/A00051826_behaviour.npz
    python src/extract_behaviour.py --batch --input $SCRATCH/mh/data/mipdb/ \
                                    --output-dir $SCRATCH/mh/data/processed_beh/
"""
import argparse
import re
import sys
from pathlib import Path

# numpy/pandas imported lazily inside the extraction functions so that --probe
# (pure stdlib: just lists/preview files) runs on the login node without a venv.

BEH_FEATURES = ["rt_mean", "rt_std", "rt_median", "rt_min", "rt_max", "accuracy"]

# MIPDB SAIIT_2AFC behaviour:
#   accuracy  <- Behavioral/csv/<sid>_SAIIT_2AFC_Block*.csv  (trialLR vs RespLR)
#   RT        <- EEG/preprocessed/csv_format/*_events.csv     (latency CD_RESP - CD_TG)
# Target/response trigger codes (CD_TG, CD_RESP) are auto-read from row 1 of each
# behavioural block, then matched in the events files -> no hard-coded code map.
SFREQ_DEFAULT = 500.0   # MIPDB EEG native sampling rate (events latency in samples)


# ---------------------------------------------------------------- probe mode
def probe(subject_dir):
    p = Path(subject_dir)
    allf = [f for f in p.rglob("*") if f.is_file()]
    samples = [f for f in allf if f.name.endswith("Samples.txt")]
    others = [f for f in allf if not f.name.endswith("Samples.txt")]
    print(f"[probe] {p}")
    print(f"[probe] {len(allf)} files total | {len(samples)} *Samples.txt | "
          f"{len(others)} other")
    cand = [f for f in others if re.search(r"Events|\.csv$|\.log$|behav|resp|2AFC|SAIIT",
                                           f.name, re.I)]
    print(f"\n[probe] candidate behaviour/event files ({len(cand)}):")
    for f in cand[:40]:
        print(f"   {f.relative_to(p)}   ({f.stat().st_size} B)")
    print(f"\n[probe] other (non-Samples) extensions seen:")
    exts = sorted({f.suffix.lower() for f in others})
    print("  ", exts)
    if cand:
        f = cand[0]
        print(f"\n[probe] head of first candidate -> {f.relative_to(p)}")
        with open(f, "r", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 25:
                    break
                print("   |", line.rstrip()[:160])


# --------------------------------------------------- accuracy from behavioural CSV
def _accuracy_and_codes(beh_files):
    """From SAIIT_2AFC block CSVs: accuracy = mean(trialLR == RespLR) over valid
    trials, plus the CD_TG / CD_RESP trigger codes read from row 1."""
    import numpy as np
    import pandas as pd
    correct = total = 0
    cd_tg = cd_resp = None
    for f in beh_files:
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if "trialLR" in df.columns and "RespLR" in df.columns:
            tl = pd.to_numeric(df["trialLR"], errors="coerce")
            rl = pd.to_numeric(df["RespLR"], errors="coerce")
            v = tl.notna() & rl.notna()
            correct += int((tl[v] == rl[v]).sum()); total += int(v.sum())
        if cd_tg is None and "CD_TG" in df.columns and "CD_RESP" in df.columns:
            try:
                cd_tg = int(float(df["CD_TG"].dropna().iloc[0]))
                cd_resp = int(float(df["CD_RESP"].dropna().iloc[0]))
            except Exception:  # noqa: BLE001
                pass
    acc = (correct / total) if total else float("nan")
    return acc, total, cd_tg, cd_resp


# ------------------------------------------------------ RT from EEG events files
def _parse_events(path):
    """Repeating-header events CSV -> list of (code:int, latency:int)."""
    out = []
    for line in open(path, "r", errors="replace"):
        s = line.strip()
        if not s or s.startswith('"type"') or s.startswith("type"):
            continue
        parts = [x.strip().strip('"').strip() for x in s.split(",")]
        if len(parts) < 3:
            continue
        try:
            out.append((int(float(parts[0])), int(float(parts[2]))))
        except (ValueError, IndexError):
            continue
    return out


def _rts_from_events(events, cd_tg, cd_resp):
    """Pair each target (CD_TG) with the next response (CD_RESP); RT in samples."""
    rts, last_tg = [], None
    for code, lat in events:
        if code == cd_tg:
            last_tg = lat
        elif code == cd_resp and last_tg is not None:
            dt = lat - last_tg
            if dt > 0:
                rts.append(dt)
            last_tg = None
    return rts


def behaviour_features(subject_dir, sfreq=SFREQ_DEFAULT):
    import numpy as np
    p = Path(subject_dir)
    beh_files = sorted(p.glob("Behavioral/csv/*SAIIT_2AFC*.csv")) \
        or sorted(p.rglob("*SAIIT_2AFC*.csv"))
    if not beh_files:
        raise ValueError("no SAIIT_2AFC behavioural CSVs (run --probe)")
    acc, n_trials, cd_tg, cd_resp = _accuracy_and_codes(beh_files)

    rts = []
    if cd_tg is not None:
        for ef in sorted(p.rglob("*_events.csv")):
            ev = _parse_events(ef)
            codes = {c for c, _ in ev}
            if cd_tg in codes and cd_resp in codes:
                rts += _rts_from_events(ev, cd_tg, cd_resp)
    rt = np.array(rts, float) / float(sfreq)            # samples -> seconds
    rt = rt[(rt > 0.05) & (rt < 5.0)]                   # plausible RT window

    if rt.size == 0:
        print(f"[beh ] {p.name}: WARN no RT pairs (cd_tg={cd_tg} cd_resp={cd_resp}); "
              f"accuracy-only. acc={acc:.3f} n_trials={n_trials}")
        feats = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, acc], np.float32)
    else:
        feats = np.array([rt.mean(), rt.std(), np.median(rt), rt.min(), rt.max(), acc],
                         dtype=np.float32)
        print(f"[beh ] {p.name}: {rt.size} RT trials (codes {cd_tg}->{cd_resp}) | "
              f"rt_mean={feats[0]:.3f}s acc={acc:.3f} n_acc_trials={n_trials}")
    return feats.reshape(1, -1)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Extract MIPDB SAIIT_2AFC behaviour features.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--output-dir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--probe", action="store_true", help="inspect raw layout, no extraction")
    args = ap.parse_args()

    if args.probe:
        probe(args.input); return
    import numpy as np
    if args.batch:
        parent = Path(args.input)
        if not args.output_dir:
            sys.exit("--batch requires --output-dir.")
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        subs = sorted(p for p in parent.iterdir() if p.is_dir())
        done = failed = 0
        for i, subj in enumerate(subs, 1):
            out = out_dir / f"{subj.name}_behaviour.npz"
            if out.exists() and not args.overwrite:
                continue
            try:
                arr = behaviour_features(subj)
                np.savez_compressed(out, features=arr, feature_names=np.array(BEH_FEATURES))
                print(f"[batch {i}/{len(subs)}] {subj.name}: saved {arr.shape}"); done += 1
            except Exception as e:  # noqa: BLE001
                print(f"[batch {i}/{len(subs)}] {subj.name}: skip ({e})"); failed += 1
        print(f"\n[batch] done={done} failed/no-beh={failed}")
        return

    arr = behaviour_features(args.input)
    if not args.output:
        sys.exit("Provide --output (or --probe).")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=arr, feature_names=np.array(BEH_FEATURES))
    print(f"[done] saved {arr.shape} -> {args.output}")


if __name__ == "__main__":
    main()
