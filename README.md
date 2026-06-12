# Canada Alliance Case — Code & Datasets

Repository for analysis code and datasets related to the legal matter.

> **Confidential.** This repository may contain material relevant to active litigation.
> Do not share publicly. If pushed to a remote, use a **private** repository.

## Structure

| Folder | What goes here |
|---|---|
| `data/raw/` | Original, untouched source data (never edit by hand). Large files are git-ignored. |
| `data/processed/` | Cleaned / derived datasets produced by code in `src/`. Git-ignored. |
| `src/` | Analysis and processing code. |
| `notebooks/` | Exploratory Jupyter notebooks. |
| `results/` | Generated outputs: figures, tables, reports. Git-ignored. |
| `docs/` | Documentation, data dictionaries, methodology notes. |

## Data handling notes

- **Raw data is treated as read-only.** All transformations are done in code so results are reproducible.
- **Large data files are not committed** to git (see `.gitignore`). To version large files,
  enable [Git LFS](https://git-lfs.com/): `git lfs install` then `git lfs track "*.csv"`.
- Keep a record in `docs/` of where each raw dataset came from and when it was obtained
  (important for evidentiary chain-of-custody).

## Getting started

```powershell
# Clone (if remote exists)
git clone <remote-url>

# Or start working in the existing local repo
cd canada-alliance-case
```
