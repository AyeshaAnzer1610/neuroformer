#!/usr/bin/env python
"""Generate the supervisor-meeting deck (NeuroFormer major revision).

    pip install python-pptx
    python scripts/make_slides.py
-> docs/resubmission/NeuroFormer-Supervisor-Meeting.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

NAVY = RGBColor(0x1F, 0x33, 0x55)
BLUE = RGBColor(0x2E, 0x6F, 0xB5)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    tf = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf


def title_bar(s, text, color=NAVY):
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(1.1))
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    tf = bar.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.4)
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return s


def bullets(tf, items, size=18):
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt; p.level = lvl
        p.font.size = Pt(size - 2 * lvl)
        p.font.color.rgb = NAVY if lvl == 0 else GREY
        if lvl == 0:
            p.font.bold = True
        p.space_after = Pt(6)


def add_table(s, rows, l, t, w, h, header_color=BLUE, fs=13):
    nr, nc = len(rows), len(rows[0])
    tbl = s.shapes.add_table(nr, nc, Inches(l), Inches(t), Inches(w), Inches(h)).table
    for j in range(nc):
        for i in range(nr):
            c = tbl.cell(i, j); c.text = str(rows[i][j])
            par = c.text_frame.paragraphs[0]
            par.font.size = Pt(fs)
            if i == 0:
                par.font.bold = True; par.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                c.fill.solid(); c.fill.fore_color.rgb = header_color
            else:
                par.font.color.rgb = NAVY
                c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else RGBColor(0xFF, 0xFF, 0xFF)
    return tbl


# ---- 1. Title ----------------------------------------------------------------
s = slide()
r = s.shapes.add_shape(1, 0, 0, W, prs.slide_height); r.fill.solid()
r.fill.fore_color.rgb = NAVY; r.line.fill.background()
tf = box(s, 0.9, 2.2, 11.5, 3)
p = tf.paragraphs[0]; p.text = "NeuroFormer: Major Revision"
p.font.size = Pt(44); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
for line in ["Multimodal transformer for transdiagnostic mental-health screening",
             "From a leaky 0.91 to an honest, rigorously-validated 0.75 — with foundation-model EEG",
             "Supervisor meeting briefing"]:
    q = tf.add_paragraph(); q.text = line
    q.font.size = Pt(20); q.font.color.rgb = RGBColor(0xD8, 0xE2, 0xF0)

# ---- 2. The story ------------------------------------------------------------
s = slide(); title_bar(s, "What we did and why")
tf = box(s, 0.6, 1.4, 12.1, 5.6)
bullets(tf, [
    ("The model (unchanged): NeuroFormer fuses EEG + eye-tracking + behaviour to screen 'any DSM diagnosis vs none'. This is a REVISION, so we kept the architecture.", 0),
    ("Problem we found: the original 0.91 was inflated by DATA LEAKAGE — the same person's recordings were in both train and test, so the model memorised people, not the condition.", 0),
    ("Fix 1 — honest evaluation: strict subject-level split (all of one person's data on one side), repeated cross-validation, confidence intervals.", 1),
    ("Fix 2 — the big upgrade: replace weak hand-crafted EEG features with a PRETRAINED EEG FOUNDATION MODEL (LaBraM). Like using a pretrained language model instead of counting words.", 1),
    ("Result: EEG flips from our WEAKEST to our STRONGEST modality — honest leak-free AUC ~ 0.75.", 0),
])

# ---- 3. Contributions: original vs now --------------------------------------
s = slide(); title_bar(s, "Contributions: original vs. now")
add_table(s, [
    ["Original submission", "This revision (kept the model, fixed everything around it)"],
    ["Hierarchical cross-modal transformer (EEG+eye+behaviour)", "SAME architecture — compact variant (2.9M vs 18.9M params)"],
    ["Hand-crafted 5-channel EEG features", "Foundation-model EEG (frozen LaBraM, 200-D)"],
    ["Reported 0.91 (single split)", "Leak-free 0.75 (subject-level, 5x10 CV, CIs)"],
    ["Single cohort, no external test", "External validation on HBN (~2,000 subjects)"],
    ["Strong 3-modality claim", "Honest: fusion does not beat EEG-alone here"],
    ["Plain classifier output", "Calibrated, refer-to-clinician (triage) output"],
    ["—", "Missing-modality design + recent baselines + ablations"],
], 0.5, 1.35, 12.3, 5.6, fs=13)

# ---- 4. Key results ----------------------------------------------------------
s = slide(); title_bar(s, "Key results (honest, leak-free)")
add_table(s, [
    ["Approach", "ROC-AUC", "Read"],
    ["Hand-crafted EEG", "0.43-0.47", "~ chance (old features)"],
    ["Foundation-model EEG (window-level)", "0.748 +/- 0.029", "THE WIN"],
    ["Eye-tracking / behaviour alone", "0.51 / 0.40", "no signal here"],
    ["Multimodal fusion (best variant)", "<= 0.66", "fusion doesn't help"],
    ["Full end-to-end fine-tuning", "0.69 (+/-0.12)", "overfits -> less is more"],
    ["External (HBN, N~2,000)", "~ 0.50", "does not transfer"],
], 0.7, 1.4, 9.2, 4.6, fs=15)
tf = box(s, 10.2, 1.5, 2.7, 4.5)
bullets(tf, [("Plain takeaway:", 0),
             ("EEG (done right) carries the signal", 1),
             ("Other modalities add nothing here", 1),
             ("Doesn't generalise to a very different cohort", 1),
             ("We report all of this straight", 1)], size=15)

# ---- 5. Reviewer comments -> responses --------------------------------------
s = slide(); title_bar(s, "Reviewer comments -> how we addressed them")
add_table(s, [
    ["Reviewer said", "We did"],
    ["Is EEG worth including? Eye dominates.", "Foundation-model EEG is now the STRONGEST (0.75 vs 0.43)"],
    ["18.9M params too big for ~90 subjects.", "Compact 2.9M model; showed full fine-tuning overfits"],
    ["Small N — sensitive to folds.", "Repeated CV (5x10) + confidence intervals"],
    ["No external/real-world validation.", "Added HBN (~2,000) — reported it doesn't transfer"],
    ["Soften the 3-modality claim.", "Stated plainly: fusion does not beat EEG-alone"],
    ["Add recent baselines.", "Single-stage transformer + gated fusion, same protocol"],
    ["Clinical-screening claim overstated.", "Calibration + refer-to-clinician; proof-of-concept framing"],
    ["Fusion premise weak (different tasks).", "Reframed as participant-level (not synchronized) fusion"],
], 0.5, 1.35, 12.3, 5.6, fs=12)

# ---- 6. Literature gap -------------------------------------------------------
s = slide(); title_bar(s, "Where we sit in the literature (the gap we fill)")
tf = box(s, 0.6, 1.35, 12.1, 5.7)
bullets(tf, [
    ("EEG foundation models exist — but not used for multimodal psychiatric screening:", 0),
    ("LaBraM (ICLR'24), EEGPT (NeurIPS'24), CBraMod (ICLR'25) — we adopt LaBraM.", 1),
    ("EEG+eye fusion exists — but disorder-specific, single-cohort, no external test:", 0),
    ("MTNet (BSPC'24), MSNet (ETRA'25) fuse EEG+eye for DEPRESSION. We do not claim 'first EEG+eye'.", 1),
    ("Missing-modality clinical fusion exists — in other domains:", 0),
    ("DrFuse (AAAI'24), M3Care (KDD'22) — we bring this idea to psychiatric screening.", 1),
    ("External cohort + calibrated triage are largely absent in this niche:", 0),
    ("HBN-EEG (NeurIPS'25 challenge) for external test; conformal clinical triage (Sci.Rep.'25) for referral.", 1),
    ("OUR GAP: first to COMBINE foundation-model EEG + missing-modality + transdiagnostic + external validation + calibrated referral, under leak-free evaluation.", 0),
])

# ---- 7. Defending 0.75 -------------------------------------------------------
s = slide(); title_bar(s, "If asked: 'is 0.75 good enough?'")
tf = box(s, 0.6, 1.4, 12.1, 5.6)
bullets(tf, [
    ("Yes — for this task. 'Any diagnosis vs none' lumps ADHD/ASD/anxiety/depression together: the hardest possible target.", 0),
    ("Leak-free transdiagnostic EEG AUCs in the field live around 0.65-0.80; 0.75 is solid.", 1),
    ("The original 0.91 was leakage — not a real number to defend.", 0),
    ("Reviewers asked for RIGOR, not a bigger number. A model that jumps back to 0.9 after we add complexity would look suspicious.", 1),
    ("The contribution is the rigour + foundation-model EEG + external test + calibration — not a leaderboard score.", 0),
    ("Why not per-disorder? The public dataset only has a yes/no label — specific diagnoses aren't available (we won't fabricate them).", 1),
])

# ---- 8. Limitations & next steps --------------------------------------------
s = slide(); title_bar(s, "Limitations & next steps", GREEN)
tf = box(s, 0.6, 1.4, 12.1, 5.6)
bullets(tf, [
    ("Honest limitations (stated in the paper):", 0),
    ("Modest N (~118); ~20% prevalence; single primary cohort.", 1),
    ("Signal does not generalise out-of-distribution (HBN).", 1),
    ("Non-EEG modalities add nothing in this cohort.", 1),
    ("Pending (small):", 0),
    ("Two recent-baseline numbers (one cluster rerun).", 1),
    ("Model-consistent calibration on the best model.", 1),
    ("Final manuscript figures (architecture + pipeline — drafted).", 1),
    ("Future work: pretrain a gaze encoder + cross-modal alignment on a larger paired cohort.", 0),
])

# ---- 9. Takeaway -------------------------------------------------------------
s = slide()
r = s.shapes.add_shape(1, 0, 0, W, prs.slide_height); r.fill.solid()
r.fill.fore_color.rgb = NAVY; r.line.fill.background()
tf = box(s, 1.0, 2.6, 11.3, 2.5)
p = tf.paragraphs[0]
p.text = "Takeaway"
p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
q = tf.add_paragraph()
q.text = ("With a modern EEG foundation model, EEG goes from weakest to strongest "
          "modality (leak-free ~0.75). Honest multimodal and external tests bound the "
          "claims. A rigorously-evaluated proof-of-concept that answers every reviewer.")
q.font.size = Pt(20); q.font.color.rgb = RGBColor(0xD8, 0xE2, 0xF0)

out = Path("docs/resubmission/NeuroFormer-Supervisor-Meeting.pptx")
out.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(out))
print(f"[done] wrote {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
