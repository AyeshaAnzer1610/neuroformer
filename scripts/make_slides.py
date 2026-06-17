#!/usr/bin/env python
"""Generate the supervisor-meeting deck (NeuroFormer major revision), with
detailed speaker notes and the two figures embedded.

    pip install python-pptx
    # save the two diagrams first (see filenames in FIG_PIPELINE / FIG_ARCH):
    #   figures/neuroformer_pipeline.png      (the wide pipeline + bar chart)
    #   figures/neuroformer_architecture.png  (the tall stage architecture)
    python scripts/make_slides.py
-> docs/resubmission/NeuroFormer-Supervisor-Meeting.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

NAVY = RGBColor(0x1F, 0x33, 0x55)
BLUE = RGBColor(0x2E, 0x6F, 0xB5)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FIG_PIPELINE = "figures/neuroformer_pipeline.png"
FIG_ARCH = "figures/neuroformer_architecture.png"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    tf = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf


def title_bar(s, text, color=NAVY):
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = Inches(0.4)
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE


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
            par = c.text_frame.paragraphs[0]; par.font.size = Pt(fs)
            if i == 0:
                par.font.bold = True; par.font.color.rgb = WHITE
                c.fill.solid(); c.fill.fore_color.rgb = header_color
            else:
                par.font.color.rgb = NAVY
                c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else WHITE


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text.strip()


def figure_slide(title, path, caption, note, mode="wide"):
    s = slide(); title_bar(s, title)
    if Path(path).exists():
        if mode == "wide":
            s.shapes.add_picture(path, Inches(0.3), Inches(1.4), width=Inches(12.7))
        else:
            pic = s.shapes.add_picture(path, Inches(0), Inches(1.2), height=Inches(5.6))
            pic.left = int((W - pic.width) / 2)
    else:
        tf = box(s, 1.5, 3.2, 10.3, 1.2)
        p = tf.paragraphs[0]
        p.text = f"[ Save the figure to  {path}  and re-run scripts/make_slides.py ]"
        p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = RGBColor(0xB0, 0x30, 0x30)
    cap = box(s, 0.5, 6.85, 12.3, 0.55)
    cp = cap.paragraphs[0]; cp.text = caption
    cp.font.size = Pt(12); cp.font.italic = True; cp.font.color.rgb = GREY
    notes(s, note)
    return s


# ============================== 1. TITLE =====================================
s = slide()
r = s.shapes.add_shape(1, 0, 0, W, H); r.fill.solid()
r.fill.fore_color.rgb = NAVY; r.line.fill.background()
tf = box(s, 0.9, 2.2, 11.5, 3)
p = tf.paragraphs[0]; p.text = "NeuroFormer: Major Revision"
p.font.size = Pt(44); p.font.bold = True; p.font.color.rgb = WHITE
for line in ["Multimodal transformer for transdiagnostic mental-health screening",
             "From a leaky 0.91 to an honest 0.75 - with foundation-model EEG",
             "Supervisor meeting briefing"]:
    q = tf.add_paragraph(); q.text = line
    q.font.size = Pt(20); q.font.color.rgb = RGBColor(0xD8, 0xE2, 0xF0)
notes(s, """
Opening line you can say out loud:
"I want to walk you through how we turned the rejected NeuroFormer paper into a strong
major revision. The headline: our original 0.91 accuracy turned out to be inflated by a
data-leakage bug. After fixing the evaluation properly and swapping in a pretrained EEG
foundation model, we get an honest 0.75 - and importantly, that change flips EEG from our
weakest signal to our strongest. I'll cover what we did, the results, how we answered each
reviewer, and where we sit in the literature."
Keep this to ~30 seconds; it frames the whole meeting.
""")

# ============================== 2. WHAT WE DID ================================
s = slide(); title_bar(s, "What we did and why")
tf = box(s, 0.6, 1.3, 12.1, 5.6)
bullets(tf, [
    ("The model (unchanged): NeuroFormer fuses EEG + eye-tracking + behaviour to screen 'any DSM diagnosis vs none'. It's a REVISION, so we kept the architecture.", 0),
    ("Problem we found: the original 0.91 was inflated by DATA LEAKAGE - the same person's recordings were in both train and test, so the model memorised people, not the condition.", 0),
    ("Fix 1 - honest evaluation: keep all of one person's data on one side; repeated cross-validation; confidence intervals.", 1),
    ("Fix 2 - the big upgrade: replace weak hand-crafted EEG features with a PRETRAINED EEG FOUNDATION MODEL (LaBraM).", 1),
    ("Result: EEG flips from WEAKEST to STRONGEST modality - honest leak-free AUC ~ 0.75.", 0),
])
notes(s, """
Explain leakage in everyday terms: "Imagine testing a student on the exact questions they
studied - they look brilliant but haven't really learned. Our original split let the same
person appear in training and testing, so the model recognised individuals rather than the
disorder. That's why 0.91 was not real."

Then the fix: "We rebuilt the evaluation so every recording from one person sits entirely
in training OR testing, never both. And we repeat the whole cross-validation 10 times and
report confidence intervals, so it's not luck of one split."

Then the upgrade: "Separately, our EEG features were hand-designed and barely worked. We
replaced them with LaBraM - a model already trained on thousands of hours of EEG, like a
pretrained language model. We don't retrain it; we just use its learned representation."

Punchline: "Those two changes together flip EEG from our weakest modality to our strongest."
""")

# ============================== 3. PIPELINE FIGURE ===========================
figure_slide(
    "The pipeline at a glance", FIG_PIPELINE,
    "NeuroFormer pipeline: acquisition -> preprocessing/QC -> frozen LaBraM EEG -> "
    "multimodal transformer -> leak-free evaluation + calibrated referral -> external test.",
    """
Walk left to right - one sentence per box:
1. ACQUISITION: each participant does cognitive tasks while we record EEG (brain activity),
   eye-tracking (gaze + pupil), and behaviour (reaction time, accuracy).
2. PREPROCESSING + QC: clean the EEG - band-pass 0.1-75 Hz, remove 60 Hz mains noise,
   downsample to 200 Hz, and a quality gate drops bad recordings. Cut into 15-second windows.
3. FROZEN EEG FOUNDATION MODEL (LaBraM): each 15 s window becomes a 200-number summary
   ('embedding'). 'Frozen' (the snowflake) means we don't retrain it - we reuse what it learned.
4. MULTIMODAL TRANSFORMER: combines the three streams via attention and a [CLS] summary token.
   It's compact (~2.9M parameters) to suit our small sample.
5. LEAK-FREE EVALUATION + CALIBRATED REFERRAL: 5x10 cross-validation; and a calibration step
   so the model can REFER uncertain cases to a clinician instead of guessing.
6. EXTERNAL VALIDATION: we test on a completely separate cohort, HBN (~2,000 people).

Then read the BAR CHART at the bottom: "hand-crafted EEG is basically chance (0.46); a simple
classifier on the foundation-model EEG is 0.66; training it window-by-window gives our best,
0.748; pushing harder with full fine-tuning actually overfits to 0.69; and on the external
cohort it drops to chance (0.50). The dashed line at 0.5 is a coin flip."
If the box says 'save the figure' - I still need to drop the PNG into figures/.
""")

# ============================== 4. CONTRIBUTIONS =============================
s = slide(); title_bar(s, "Contributions: original vs. now")
add_table(s, [
    ["Original submission", "This revision (same model, fixed everything around it)"],
    ["Hierarchical cross-modal transformer", "SAME architecture - compact 2.9M (vs 18.9M originally submitted)"],
    ["Hand-crafted 5-channel EEG features", "Foundation-model EEG (frozen LaBraM, 200-D)"],
    ["Reported 0.91 (single split)", "Leak-free 0.75 (subject-level, 5x10 CV, CIs)"],
    ["Single cohort, no external test", "External validation on HBN (~2,000 subjects)"],
    ["Strong 3-modality claim", "Honest: fusion does not beat EEG-alone here"],
    ["Plain classifier output", "Calibrated, refer-to-clinician (triage) output"],
    ["-", "Missing-modality design + recent baselines + ablations"],
], 0.5, 1.25, 12.3, 5.7, fs=13)
notes(s, """
The message of this slide: "We did NOT invent a new architecture - that would be a different
paper. We kept NeuroFormer and improved its inputs, training, and evaluation - exactly what a
revision should do."

Go row by row if asked, but the three that matter most:
- Inputs: hand-crafted EEG -> foundation-model EEG (this is the scientific upgrade).
- Evaluation: leaky single split -> leak-free repeated CV + an external dataset (this is the
  rigour the reviewers wanted).
- Honesty: we dropped the overclaim that all three modalities help, and added a calibrated
  'refer to clinician' output so it's framed as decision-support, not a diagnosis machine.

If asked 'so what's genuinely new?': the COMBINATION - foundation-model EEG + missing-modality
+ transdiagnostic + external validation + calibration, all under leak-free evaluation.
""")

# ============================== 5. KEY RESULTS ===============================
s = slide(); title_bar(s, "Key results (honest, leak-free)")
add_table(s, [
    ["Approach", "ROC-AUC", "Read"],
    ["Hand-crafted EEG", "0.43-0.47", "~ chance (old features)"],
    ["Foundation-model EEG (window-level)", "0.748 +/- 0.029", "THE WIN"],
    ["Eye-tracking / behaviour alone", "0.51 / 0.40", "no signal here"],
    ["Multimodal fusion (best variant)", "<= 0.66", "fusion doesn't help"],
    ["Full end-to-end fine-tuning", "0.69 (+/-0.12)", "overfits -> less is more"],
    ["External (HBN, N~2,000)", "~ 0.50", "does not transfer"],
], 0.7, 1.35, 9.2, 4.7, fs=15)
tf = box(s, 10.2, 1.45, 2.8, 4.6)
bullets(tf, [("Plain takeaway:", 0), ("EEG (done right) carries the signal", 1),
             ("Other modalities add nothing here", 1),
             ("Doesn't generalise to a very different cohort", 1),
             ("We report all of it straight", 1)], size=15)
notes(s, """
Quick reminder: ROC-AUC of 0.5 = coin flip, 1.0 = perfect. ~0.75 is a meaningful, useful signal.

Read it as a story, not a list:
"Our old EEG features were basically chance. With the foundation model, EEG jumps to 0.75 -
that's the result. Eye and behaviour on their own are at chance in this cohort, and combining
all three never beats EEG alone - so we report that honestly rather than claiming fusion helps.
We also tried pushing harder by fine-tuning the whole foundation model; on only ~118 people it
OVERFITS - the average drops and the variance triples. That's actually a useful finding: it
directly answers the reviewer who said our model was too big. And on a separate cohort the
signal doesn't transfer - which we present as an honest boundary, not a failure."

If asked why fusion fails: with ~23 positive cases, adding noisy modalities injects variance
that outweighs any tiny complementary signal.
""")

# ============================== 6. ARCHITECTURE FIGURE =======================
figure_slide(
    "The model architecture", FIG_ARCH,
    "NeuroFormer (compact): per-modality encoders -> concatenate with [CLS] -> cross-modal "
    "fusion -> classification head. EEG enters as frozen LaBraM embeddings.",
    """
Walk BOTTOM to TOP:
- STAGE 1 INPUTS: EEG enters as the 200-number LaBraM embedding; eye is 11 features; behaviour
  is 6 features.
- STAGE 2 EMBEDDING: each is projected to a common size (128) and given a position code.
- STAGE 3 ENCODERS: three SEPARATE encoders (one per modality) - each looks only within its own
  modality first. (We fixed an earlier version that wrongly merged them too early.)
- STAGE 4 CONCATENATE: stack the three encoded streams and add a [CLS] 'summary' token.
- STAGE 5 FUSION: a cross-modal transformer lets [CLS] attend across all modalities.
- STAGE 6-7 HEAD: two fully-connected layers -> softmax -> 'any diagnosis vs none'.

Two things to say out loud:
1. "It's the COMPACT version - ~2.9M parameters, deliberately small for our sample size."
2. "Important honesty point: this diagram is the multimodal FUSION model. Our headline 0.75 is
   actually the EEG-only path. Fusion doesn't beat EEG alone here - the caption on the figure
   says exactly that, so we're not implying the full pipeline produced the number."
""", mode="tall")

# ============================== 7. REVIEWER RESPONSES ========================
s = slide(); title_bar(s, "Reviewer comments -> how we addressed them")
add_table(s, [
    ["Reviewer said", "We did"],
    ["Is EEG worth including? Eye dominates.", "Foundation-model EEG is now STRONGEST (0.75 vs 0.43)"],
    ["18.9M params too big for ~90 subjects.", "Compact 2.9M model; showed full fine-tuning overfits"],
    ["Small N - sensitive to folds.", "Repeated CV (5x10) + confidence intervals"],
    ["No external/real-world validation.", "Added HBN (~2,000); reported it doesn't transfer"],
    ["Soften the 3-modality claim.", "Stated plainly: fusion does not beat EEG-alone"],
    ["Add recent baselines.", "Single-stage transformer + gated fusion, same protocol"],
    ["Clinical claim overstated.", "Calibration + refer-to-clinician; proof-of-concept framing"],
    ["Fusion premise weak (diff. tasks).", "Reframed as participant-level (not synchronized) fusion"],
], 0.5, 1.25, 12.3, 5.7, fs=12)
notes(s, """
This is the slide the supervisor will care most about - it shows we engaged seriously.

Frame it as: "Every major comment is answered with a concrete change, and in two cases we
turned a criticism INTO a result."
- The 'EEG is weak' comment: we agree about the OLD features, and the foundation model resolves
  it - EEG is now the strongest modality.
- The 'too many parameters' comment: instead of just shrinking, we SHOWED that full fine-tuning
  overfits while the compact frozen model wins - so the data prove their point and ours.
- 'No external validation' was the biggest ask; we added a ~2,000-subject cohort and reported
  the honest negative.
- We softened the three-modality claim, added recent baselines under one fair protocol, added
  calibration/triage for the clinical framing, and reframed fusion as participant-level (because
  the modalities come from different tasks, not synchronized events).
""")

# ============================== 8. LITERATURE GAP ============================
s = slide(); title_bar(s, "Where we sit in the literature (the gap)")
tf = box(s, 0.6, 1.25, 12.1, 5.8)
bullets(tf, [
    ("EEG foundation models exist - but not used for multimodal psychiatric screening:", 0),
    ("LaBraM (ICLR'24), EEGPT (NeurIPS'24), CBraMod (ICLR'25). We adopt LaBraM.", 1),
    ("EEG+eye fusion exists - but disorder-specific, single-cohort, no external test:", 0),
    ("MTNet (BSPC'24), MSNet (ETRA'25) fuse EEG+eye for DEPRESSION. We do NOT claim 'first EEG+eye'.", 1),
    ("Missing-modality clinical fusion exists - in other medical domains:", 0),
    ("DrFuse (AAAI'24), M3Care (KDD'22). We bring it to psychiatric screening.", 1),
    ("External cohort + calibrated triage are largely absent in this niche:", 0),
    ("HBN-EEG (NeurIPS'25 challenge) external test; conformal clinical triage (Sci. Rep.'25).", 1),
    ("OUR GAP: first to COMBINE foundation-model EEG + missing-modality + transdiagnostic + external validation + calibrated referral, leak-free.", 0),
])
notes(s, """
This answers 'what paper are we filling the gap of'. Say it as four honest boundaries:
1. "EEG foundation models are new and powerful, but nobody has plugged them into a multimodal
   psychiatric screening model - we do, using LaBraM."
2. "People HAVE fused EEG and eye-tracking - but for a single disorder (depression), on one
   dataset, with no external test. So we are careful NOT to claim we're the first to fuse
   EEG+eye - our novelty is the transdiagnostic, externally-validated, foundation-model version."
3. "Missing-modality fusion is established in other medical AI (DrFuse, M3Care) but not here;
   we adapt it so one model works even when a modality is missing."
4. "External validation and calibrated 'refer to clinician' are basically absent in this niche."

Close with: "So the contribution is not one new trick - it's the COMBINATION, done rigorously.
That's a defensible, gap-filling story for IEEE Access."
""")

# ============================== 9. DEFENDING 0.75 ============================
s = slide(); title_bar(s, "If asked: 'is 0.75 good enough?'")
tf = box(s, 0.6, 1.3, 12.1, 5.6)
bullets(tf, [
    ("Yes - for THIS task. 'Any diagnosis vs none' lumps ADHD/ASD/anxiety/depression: the hardest target.", 0),
    ("Leak-free transdiagnostic EEG AUCs in the field sit ~0.65-0.80; 0.75 is solid.", 1),
    ("The original 0.91 was leakage - not a real number to defend.", 0),
    ("Reviewers asked for RIGOUR, not a bigger number; a jump back to 0.9 would look suspicious.", 1),
    ("Contribution = rigour + foundation-model EEG + external test + calibration, not a leaderboard score.", 0),
    ("Why not per-disorder? The public dataset only has a yes/no label - we won't fabricate specifics.", 1),
])
notes(s, """
This is the hard-question rehearsal. If the supervisor says "0.75 seems low":

"It's the right number for the hardest version of the task. We're not detecting one disorder -
we're asking 'any psychiatric diagnosis vs none', which mixes very different conditions. In the
EEG literature, honest leak-free numbers for this kind of task sit around 0.65-0.80, so 0.75 is
respectable. The 0.91 was an artifact of leakage; defending it would be defending a bug. And the
reviewers explicitly asked for rigour and generalisation - if our number jumped back up after we
added complexity, that's exactly the red flag they'd pounce on. Our contribution is the rigour
and the method, not a leaderboard score."

If asked about per-disorder breakdown: "The public dataset only gives a yes/no diagnosis label;
the specific disorders aren't released, so we can't do it without fabricating - and we won't."
""")

# ============================== 10. LIMITATIONS ==============================
s = slide(); title_bar(s, "Limitations & next steps", GREEN)
tf = box(s, 0.6, 1.3, 12.1, 5.6)
bullets(tf, [
    ("Honest limitations (stated in the paper):", 0),
    ("Modest N (~118); ~20% have a diagnosis; single primary cohort.", 1),
    ("Signal does not generalise out-of-distribution (HBN ~ 0.50).", 1),
    ("Non-EEG modalities add nothing in this cohort.", 1),
    ("Pending (small):", 0),
    ("Two recent-baseline numbers (one cluster rerun).", 1),
    ("Model-consistent calibration on the best model.", 1),
    ("Final manuscript figures (drafted).", 1),
    ("Future work: pretrain a gaze encoder + cross-modal alignment on a larger paired cohort.", 0),
])
notes(s, """
Be the first to name the weaknesses - it builds trust:
"We're upfront that the sample is small, only about a fifth have a diagnosis, and it's mainly
one cohort. The signal doesn't transfer to the external cohort, and the non-EEG modalities don't
help here. We say all of this in the paper."

Then what's left: "A couple of baseline numbers are one cluster job away; we'll add a cleaner
calibration result on the best model; and the manuscript figures are drafted. The natural next
project is to pretrain a dedicated eye-tracking encoder and learn a shared EEG-gaze
representation on a bigger paired dataset - that's where the multimodal idea could actually pay
off."
""")

# ============================== 11. TAKEAWAY =================================
s = slide()
r = s.shapes.add_shape(1, 0, 0, W, H); r.fill.solid()
r.fill.fore_color.rgb = NAVY; r.line.fill.background()
tf = box(s, 1.0, 2.5, 11.3, 2.7)
p = tf.paragraphs[0]; p.text = "Takeaway"
p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = WHITE
q = tf.add_paragraph()
q.text = ("With a modern EEG foundation model, EEG goes from weakest to strongest modality "
          "(leak-free ~0.75). Honest multimodal and external tests bound the claims. "
          "A rigorously-evaluated proof-of-concept that answers every reviewer.")
q.font.size = Pt(20); q.font.color.rgb = RGBColor(0xD8, 0xE2, 0xF0)
notes(s, """
Closing line: "So in one sentence - a modern EEG foundation model turns EEG from our weakest
into our strongest signal, we evaluate it honestly and test it externally, and the result is a
rigorous proof-of-concept that answers every reviewer concern. The headline number is lower than
before, but it's real - and the rigour is the contribution."
Then hand over: "Happy to dig into any part - the leakage fix, the foundation model, or the
reviewer responses."
""")

out = Path("docs/resubmission/NeuroFormer-Supervisor-Meeting.pptx")
out.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(out))
print(f"[done] wrote {out}  ({len(prs.slides._sldIdLst)} slides, with speaker notes)")
print(f"[fig] pipeline embedded: {Path(FIG_PIPELINE).exists()}  | architecture embedded: {Path(FIG_ARCH).exists()}")
