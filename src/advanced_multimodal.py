"""
Simple Confidence-Gated Multimodal Fusion - Fast & Working
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Lightweight confidence-gated multimodal fusion."""

    def __init__(self, eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # EEG encoder: alignment + projection
        self.eeg_align = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Eye encoder
        self.eye_proj = nn.Sequential(
            nn.Linear(eye_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Behavior encoder
        self.beh_proj = nn.Sequential(
            nn.Linear(beh_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Confidence scores (per modality)
        self.conf_eeg = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.conf_eye = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.conf_beh = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Cross-modal refinement
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        # Fusion and classification
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, eeg, eye, beh):
        """
        Args:
            eeg: [B, 516]
            eye: [B, 11]
            beh: [B, 6]

        Returns:
            logits: [B, 2]
            confidences: dict
        """
        # Encode
        eeg_feat = self.eeg_align(eeg)    # [B, hidden_dim]
        eye_feat = self.eye_proj(eye)     # [B, hidden_dim]
        beh_feat = self.beh_proj(beh)     # [B, hidden_dim]

        # Confidence scores
        conf_eeg = self.conf_eeg(eeg_feat)
        conf_eye = self.conf_eye(eye_feat)
        conf_beh = self.conf_beh(beh_feat)

        # Softmax gating
        confs = torch.cat([conf_eeg, conf_eye, conf_beh], dim=1)  # [B, 3]
        gates = torch.softmax(confs, dim=1)

        # Weighted fusion
        fused = (gates[:, 0:1] * eeg_feat +
                 gates[:, 1:2] * eye_feat +
                 gates[:, 2:3] * beh_feat)

        # Cross-modal refinement: EEG guides others
        refine_gate = self.refine(eeg_feat)
        fused = fused * refine_gate

        # Classify
        fused = torch.relu(self.fusion(fused))
        logits = self.classifier(fused)

        return logits, {
            'eeg': conf_eeg,
            'eye': conf_eye,
            'beh': conf_beh
        }


def create_model(eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64):
    return SimpleModel(eeg_dim, eye_dim, beh_dim, hidden_dim)
