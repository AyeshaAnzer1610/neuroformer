#!/usr/bin/env python
"""E1 — trainable parameter count for the production NeuroFormer config (answers R3.4).
    python scripts/count_params.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from neuroformer import create_neuroformer

for aux in (False, True):
    m = create_neuroformer({"eeg_input_dim": 516, "eyetracking_input_dim": 11,
        "behavioral_input_dim": 6, "d_model": 128, "n_encoder_layers": 2,
        "n_fusion_layers": 2, "n_heads": 8, "dropout": 0.3, "n_classes": 2,
        "use_auxiliary_loss": aux})
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"use_auxiliary_loss={aux!s:<5}  trainable params = {n:,} = {n/1e6:.2f}M")
