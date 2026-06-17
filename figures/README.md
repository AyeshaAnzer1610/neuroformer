# Figures

## Manuscript figures (export from the approved SciSpace diagrams)
These are referenced by the `.tex` and must be added as image files here:

| Filename | What it is | Source |
|---|---|---|
| `feature_extraction_pipelines.png` | Per-modality preprocessing + feature/embedding pipeline (EEG→LaBraM 200-D, eye 11-D, behaviour 6-D) | SciSpace pipeline diagram |
| `neuroformer_architecture_diagram.png` | NeuroFormer architecture (compact: d_model 128, 2+2 layers; EEG = frozen LaBraM embeddings) | SciSpace architecture diagram |

Export the final approved diagrams from SciSpace as PNG (≥300 dpi) under exactly
these names, place them in this folder, and commit.

> **Path note:** the LaTeX uses a `figures/` (lowercase) path to match this tracked
> directory. Overleaf/Linux are case-sensitive, so keep the include paths lowercase.

## Auto-generated supplementary figures (`src/make_figures.py`)
`architecture.png`, `embedding_map.png`, `modality_showcase.png`, `results_auc.png`,
`screening_report.png` are produced by `make_figures.py` from the cached features.
They are supplementary/illustrative and are **not** the manuscript's main figures.
Regenerate with:

```bash
python src/make_figures.py --eeg-dir $SCRATCH/mh/data/processed --labels data/labels.csv
```
