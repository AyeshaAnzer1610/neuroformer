"""
Confidence-Gated Multimodal Fusion (Simplified, no BatchNorm)

Simplified version without BatchNorm to handle batch size 1 training.
"""

import torch
import torch.nn as nn


class ConfidenceGatedFusionSimple(nn.Module):
    """Confidence-Gated Multimodal Fusion (SimplifiedVersion)."""

    def __init__(self, dims, h=64, dropout=0.3):
        """
        Args:
            dims: dict of {modality_name: embedding_dim}
            h: hidden dimension (default 64, reduced from 128 to avoid overfitting)
            dropout: dropout rate (default 0.3)
        """
        super().__init__()
        self.mods = sorted(dims.keys())
        self.h = h
        self.dims = dims

        # Modality projections (no BatchNorm to support batch size 1)
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for m in self.mods
        })

        # Confidence heads
        self.confidence_head = nn.ModuleDict()
        for m in self.mods:
            self.confidence_head[m] = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h // 2, 1),
                nn.Sigmoid()
            )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Linear(h, 2)

    def forward(self, X_dict):
        """Forward pass.

        Args:
            X_dict: dict of {modality: [B, embedding_dim]}

        Returns:
            logits: [B, 2]
            confidence_dict: per-modality confidence scores
        """
        batch_size = X_dict[self.mods[0]].shape[0]
        device = X_dict[self.mods[0]].device

        # Project each modality
        projected = {}
        for m in self.mods:
            projected[m] = self.proj[m](X_dict[m])

        # Compute confidence for each modality
        confidences = {}
        for m in self.mods:
            conf = self.confidence_head[m](projected[m])  # [B, 1]
            confidences[m] = conf

        # Normalize confidences with softmax
        conf_stack = torch.cat([confidences[m] for m in self.mods], dim=1)  # [B, n_mods]
        gates = torch.softmax(conf_stack, dim=1)

        # Weight each modality by its confidence
        fused = torch.zeros(batch_size, self.h, device=device)
        for i, m in enumerate(self.mods):
            fused = fused + gates[:, i:i+1] * projected[m]

        # Process fused representation
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)

        return logits, {m: confidences[m] for m in self.mods}


class CrossModalRefinedSimple(nn.Module):
    """Cross-Modal Refined Fusion (Simplified, no BatchNorm)."""

    def __init__(self, dims, h=64, dropout=0.3):
        """
        Args:
            dims: dict of {modality_name: embedding_dim}
            h: hidden dimension (default 64)
            dropout: dropout rate (default 0.3)
        """
        super().__init__()
        self.mods = sorted(dims.keys())
        self.h = h
        self.dims = dims

        # Modality projections (no BatchNorm)
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for m in self.mods
        })

        # Confidence heads
        self.confidence_head = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Linear(h // 2, 1),
                nn.Sigmoid()
            ) for m in self.mods
        })

        # Diagnostic attention (learns from EEG)
        self.diagnostic_attention = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h),
            nn.Sigmoid()  # Element-wise gating
        )

        # Fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(h, 2)

    def forward(self, X_dict):
        """Forward pass with cross-modal refinement.

        Args:
            X_dict: dict of {modality: [B, embedding_dim]}

        Returns:
            logits: [B, 2]
            confidence_dict: per-modality confidence scores
        """
        batch_size = X_dict[self.mods[0]].shape[0]
        device = X_dict[self.mods[0]].device

        # Project modalities
        projected = {}
        for m in self.mods:
            projected[m] = self.proj[m](X_dict[m])

        # Learn diagnostic attention from EEG
        if "eeg" in self.mods:
            diag_attention = self.diagnostic_attention(projected["eeg"])  # [B, h]

            # Apply to weak modalities
            for weak_mod in ["eye", "beh"]:
                if weak_mod in projected:
                    projected[weak_mod] = projected[weak_mod] * diag_attention

        # Compute confidence for each modality
        confidences = {}
        for m in self.mods:
            conf = self.confidence_head[m](projected[m])
            confidences[m] = conf

        # Normalize and fuse
        conf_stack = torch.cat([confidences[m] for m in self.mods], dim=1)
        gates = torch.softmax(conf_stack, dim=1)

        fused = torch.zeros(batch_size, self.h, device=device)
        for i, m in enumerate(self.mods):
            fused = fused + gates[:, i:i+1] * projected[m]

        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return logits, {m: confidences[m] for m in self.mods}
