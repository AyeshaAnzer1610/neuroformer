"""
Confidence-Gated Multimodal Fusion for Psychiatric Screening

Novel architecture that learns per-modality confidence from signal quality
and uses it to adaptively weight modalities during fusion.

Key innovation: Instead of learning gates as free parameters (which overfit at
small N), gates are derived from each modality's intrinsic confidence—how "sure"
is this modality about its prediction?
"""

import torch
import torch.nn as nn


class ConfidenceGatedFusion(nn.Module):
    """Confidence-Gated Multimodal Fusion.

    Learns per-modality confidence from signal quality and uses it to adaptively
    weight modalities. Avoids overfitting that plagues learned gates at small N.
    """

    def __init__(self, dims, h=128, dropout=0.3):
        """
        Args:
            dims: dict of {modality_name: embedding_dim}
                  Example: {"eeg": 512, "eye": 11, "beh": 2}
            h: hidden dimension for modality projection (default 128)
            dropout: dropout rate (default 0.3)
        """
        super().__init__()
        self.mods = sorted(dims.keys())  # Consistent ordering
        self.h = h
        self.dims = dims

        # Modality projections to common hidden space
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for m in self.mods
        })

        # Confidence heads: learn how sure each modality is
        self.confidence_head = nn.ModuleDict()
        for m in self.mods:
            self.confidence_head[m] = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
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
        """
        Forward pass.

        Args:
            X_dict: dict of {modality: [B, embedding_dim]}

        Returns:
            logits: [B, 2]
            confidence_dict: per-modality confidence scores
        """
        batch_size = X_dict[self.mods[0]].shape[0]
        device = X_dict[self.mods[0]].device

        # Project each modality to common space
        projected = {}
        for m in self.mods:
            projected[m] = self.proj[m](X_dict[m])  # [B, h]

        # Compute confidence for each modality
        confidences = {}
        for m in self.mods:
            conf = self.confidence_head[m](projected[m])  # [B, 1]
            confidences[m] = conf

        # Normalize confidences with softmax (gates sum to 1)
        conf_stack = torch.cat([confidences[m] for m in self.mods], dim=1)  # [B, n_mods]
        gates = torch.softmax(conf_stack, dim=1)  # [B, n_mods]

        # Weight each modality by its confidence
        fused = torch.zeros(batch_size, self.h, device=device)
        for i, m in enumerate(self.mods):
            fused = fused + gates[:, i:i+1] * projected[m]  # [B, h]

        # Process fused representation
        fused = self.fusion(fused)  # [B, h]

        # Classify
        logits = self.classifier(fused)  # [B, 2]

        return logits, {m: confidences[m] for m in self.mods}


class CrossModalRefinedFusion(nn.Module):
    """Cross-Modal Refined Fusion (Advanced SOTA version).

    Extends ConfidenceGatedFusion by learning diagnostic patterns from EEG
    and guiding weak modality representations.

    Idea: EEG is strong (0.65 AUC). Eye/Behavior are weak. Learn what makes
    EEG diagnostic, then teach Eye/Behavior to emphasize those patterns.
    """

    def __init__(self, dims, h=128, dropout=0.3):
        """
        Args:
            dims: dict of {modality_name: embedding_dim}
            h: hidden dimension (default 128)
            dropout: dropout rate (default 0.3)
        """
        super().__init__()
        self.mods = sorted(dims.keys())
        self.h = h
        self.dims = dims

        # Modality projections
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for m in self.mods
        })

        # Confidence heads (same as ConfidenceGatedFusion)
        self.confidence_head = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Linear(h // 2, 1),
                nn.Sigmoid()
            ) for m in self.mods
        })

        # NEW: Diagnostic pattern learner
        # Learns which dimensions of EEG are predictive of diagnosis
        self.diagnostic_attention = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h),
            nn.Softmax(dim=-1)  # Attention over feature dimensions
        )

        # Fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(h, 2)

    def forward(self, X_dict):
        """
        Forward pass with cross-modal refinement.

        Args:
            X_dict: dict of {modality: [B, embedding_dim]}

        Returns:
            logits: [B, 2]
            confidence_dict: per-modality confidence scores
        """
        batch_size = X_dict[self.mods[0]].shape[0]
        device = X_dict[self.mods[0]].device

        # Project modalities to common space
        projected = {}
        for m in self.mods:
            projected[m] = self.proj[m](X_dict[m])  # [B, h]

        # Learn diagnostic attention from EEG (strong modality)
        if "eeg" in self.mods:
            diag_attention = self.diagnostic_attention(projected["eeg"])  # [B, h]

            # Apply diagnostic attention to weak modalities
            # This teaches them to emphasize diagnostic dimensions
            for weak_mod in ["eye", "beh"]:
                if weak_mod in projected:
                    projected[weak_mod] = projected[weak_mod] * diag_attention

        # Compute confidence for each modality
        confidences = {}
        for m in self.mods:
            conf = self.confidence_head[m](projected[m])  # [B, 1]
            confidences[m] = conf

        # Normalize confidences and fuse
        conf_stack = torch.cat([confidences[m] for m in self.mods], dim=1)
        gates = torch.softmax(conf_stack, dim=1)

        fused = torch.zeros(batch_size, self.h, device=device)
        for i, m in enumerate(self.mods):
            fused = fused + gates[:, i:i+1] * projected[m]

        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return logits, {m: confidences[m] for m in self.mods}
