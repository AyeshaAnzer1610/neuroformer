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

# SAIIT_2AFC task tag and the SMI message/column patterns we look for. These are
# best-effort defaults; --probe shows the real strings so they can be corrected.
TASK_PAT = re.compile(r"SAIIT|2AFC|Contrast", re.I)
STIM_MSG = re.compile(r"stim|onset|target|trial.?start", re.I)
RESP_MSG = re.compile(r"resp|button|key|answer|press", re.I)
CORRECT_MSG = re.compile(r"correct|hit", re.I)
INCORRECT_MSG = re.compile(r"incorrect|wrong|miss|error", re.I)


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


# ------------------------------------------------------------- SMI Events parse
def _read_events(path):
    """Read an SMI *Events.txt: tab-separated after a '##'/header preamble."""
    import pandas as pd
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines)
                if ("Time" in l and ("Type" in l or "Trial" in l)) or l.startswith("Time\t")),
               None)
    if hdr is None:
        return None
    df = pd.read_csv(path, sep="\t", skiprows=hdr, low_memory=False, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def _rt_accuracy_from_events(df):
    """Best-effort: pair stimulus->response messages for RT; correct/incorrect tags
    for accuracy. Returns (rt_array_seconds, accuracy_float) or (None, None)."""
    import numpy as np
    import pandas as pd
    msgcol = next((c for c in df.columns
                   if re.search(r"msg|message|descr|event|text|comment", c, re.I)), None)
    timecol = next((c for c in df.columns if re.search(r"time", c, re.I)), None)
    if msgcol is None or timecol is None:
        return None, None
    t = pd.to_numeric(df[timecol], errors="coerce")
    msg = df[msgcol].astype(str)
    rts, last_stim = [], None
    n_corr = n_total = 0
    for ti, mi in zip(t, msg):
        if STIM_MSG.search(mi):
            last_stim = ti
        elif RESP_MSG.search(mi) and last_stim is not None and np.isfinite(ti):
            rts.append((ti - last_stim))
            last_stim = None
        if CORRECT_MSG.search(mi):
            n_corr += 1; n_total += 1
        elif INCORRECT_MSG.search(mi):
            n_total += 1
    rts = np.array([r for r in rts if np.isfinite(r) and r > 0], float)
    # SMI times are microseconds -> seconds if values are large
    if rts.size and np.median(rts) > 1e4:
        rts = rts / 1e6
    acc = (n_corr / n_total) if n_total else np.nan
    return (rts if rts.size else None), acc


def behaviour_features(subject_dir):
    import numpy as np
    p = Path(subject_dir)
    ev_files = [f for f in p.rglob("*Events.txt") if TASK_PAT.search(f.name)] \
        or list(p.rglob("*Events.txt"))
    all_rt, accs = [], []
    for f in ev_files:
        df = _read_events(f)
        if df is None:
            continue
        rt, acc = _rt_accuracy_from_events(df)
        if rt is not None:
            all_rt.append(rt)
        if acc is not None and np.isfinite(acc):
            accs.append(acc)
    if not all_rt:
        raise ValueError("no RT pairs parsed from *Events.txt (run --probe to inspect)")
    rt = np.concatenate(all_rt)
    acc = float(np.mean(accs)) if accs else np.nan
    feats = np.array([rt.mean(), rt.std(), np.median(rt), rt.min(), rt.max(), acc],
                     dtype=np.float32)
    print(f"[beh ] {subject_dir}: {rt.size} trials | rt_mean={feats[0]:.3f}s acc={acc}")
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
