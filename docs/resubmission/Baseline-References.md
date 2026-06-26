# Baseline models & references (corrected Table 2)

All compared under one identical leak-free, subject-level, repeated (5×5) stratified CV
protocol with class-weighted training and ROC-AUC (answers R2.2: unified inputs/protocol,
representative recent multimodal architectures, significance testing). BibTeX keys are
placeholders — match to your `.bib`.

| Model (Table 2) | Family | Reference |
|---|---|---|
| Logistic Regression (concat) | classical | — (standard) |
| SVM-RBF (concat) | classical | Cortes & Vapnik, *Mach. Learn.* 1995 |
| Random Forest (concat) | classical ensemble | Breiman, *Mach. Learn.* 2001 |
| MLP early-fusion | early fusion | Ngiam et al., *Multimodal Deep Learning*, ICML 2011 |
| Late-fusion ensemble | late fusion | Poria et al., *Information Fusion* 2017 (review) |
| 1D-CNN (temporal) | temporal/conv | Schirrmeister et al., *Human Brain Mapping* 2017 (EEG deep ConvNets) |
| LSTM (temporal) | temporal/recurrent | Bashivan et al., ICLR 2016 (EEG recurrent-conv) |
| **TCN (temporal-fusion net)** | temporal-conv | Bai et al., arXiv 2018; Ingolfsson et al., *EEG-TCNet*, IEEE SMC 2020 |
| **GMU (gated fusion)** | gated | Arevalo et al., ICLR-W 2017 |
| **Cross-modal Transformer** | multimodal transformer | Tsai et al., *Multimodal Transformer (MulT)*, ACL 2019 |
| **Tensor Fusion Network** | tensor fusion | Zadeh et al., *TFN*, EMNLP 2017 |
| **Low-rank Multimodal Fusion** | low-rank tensor | Liu et al., *LMF*, ACL 2018 |
| **NeuroFormer (proposed)** | hierarchical cross-modal | this work |

**Taxonomy / survey citations** to frame the comparison (use in Related Work):
- Baltrušaitis, Ahuja & Morency, *Multimodal Machine Learning: A Survey and Taxonomy*,
  IEEE TPAMI 2019.
- Recent multimodal-fusion reviews (2023–2024) for the "latest methods" framing — add 1–2
  current surveys from your field (e.g., multimodal learning in healthcare / affective
  computing) so the bold rows above are positioned as representatives of current families.

**Statistical comparison.** Per-fold AUCs are saved (`perfold_table2.csv`) → paired Wilcoxon
vs NeuroFormer with Holm correction (`experiments/analyze_results.py`). Report the corrected
margin + significance (answers R2.2's "clarify extent of improvement and significance").

> Note (R2.3): the two indoor-object-detection / gesture-recognition papers the reviewer
> suggested are out of scope and are not added (per the EIC's guidance); the comparison above
> uses genuinely-relevant multimodal-fusion architectures instead.
