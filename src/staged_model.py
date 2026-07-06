"""
Stage 1-3 Implementation: Proper Architecture for EEG Psychiatric Screening

Stage 1:
  - Euclidean alignment (z-score normalization)
  - EEGNet depthwise-separable tokenizer

Stage 2:
  - Criss-cross attention (spatial + temporal)

Stage 3:
  - Hierarchical processing (residual MLP)

Confidence-gated multimodal fusion combines modalities adaptively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ STAGE 1: ALIGNMENT & TOKENIZATION ============

class EuclideanAlignment(nn.Module):
    """Stage 1A: Per-sample z-score normalization (no learnable params)."""

    def forward(self, x):
        """x: [B, D]"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise convolution followed by pointwise (1x1) convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNetTokenizer(nn.Module):
    """Stage 1B: EEGNet-style tokenizer for EEG feature encoding.

    Temporal convolution → Depthwise spatial convolution → Separable output.
    Reduces 516-dim input to hidden_dim with minimal parameters.
    """

    def __init__(self, input_dim=516, hidden_dim=64, F1=16, F2=32):
        super().__init__()
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, F1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(F1),
            nn.ReLU(inplace=True),
        )

        # Depthwise-separable spatial convolution
        self.spatial_conv = nn.Sequential(
            DepthwiseSeparableConv1d(F1, F2, kernel_size=3, padding=1),
            nn.BatchNorm1d(F2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # Reduce temporal dimension
        )

        # Separable output convolution
        self.output_conv = nn.Sequential(
            DepthwiseSeparableConv1d(F2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: [B, 516] → [B, hidden_dim]"""
        # Reshape to [B, 1, 516] for conv1d
        x = x.unsqueeze(1)  # [B, 1, 516]

        # Temporal convolution
        x = self.temporal_conv(x)  # [B, F1, 516]

        # Spatial convolution with pooling
        x = self.spatial_conv(x)  # [B, F2, 256]

        # Output convolution
        x = self.output_conv(x)  # [B, hidden_dim, 256]

        # Global average pooling
        x = x.mean(dim=-1)  # [B, hidden_dim]

        return x


# ============ STAGE 2: ATTENTION ============

class CrissCrossAttention(nn.Module):
    """Stage 2: Criss-cross attention (spatial + temporal simultaneously).

    Parallel attention mechanisms that capture feature interactions
    without the full quadratic cost of standard self-attention.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Spatial branch
        self.q_spatial = nn.Linear(dim, dim, bias=False)
        self.k_spatial = nn.Linear(dim, dim, bias=False)
        self.v_spatial = nn.Linear(dim, dim, bias=False)

        # Temporal branch
        self.q_temporal = nn.Linear(dim, dim, bias=False)
        self.k_temporal = nn.Linear(dim, dim, bias=False)
        self.v_temporal = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        """x: [B, D] → [B, D]"""
        B, D = x.shape

        # Add sequence dimension for attention computation
        x_seq = x.unsqueeze(1)  # [B, 1, D]

        # Spatial attention
        q_s = self.q_spatial(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]
        k_s = self.k_spatial(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]
        v_s = self.v_spatial(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]

        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.scale  # [B, H, 1, 1]
        attn_s = F.softmax(attn_s, dim=-1)
        attn_s = self.drop(attn_s)
        spatial_out = (attn_s @ v_s).transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]

        # Temporal attention
        q_t = self.q_temporal(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_temporal(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_temporal(x_seq).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_t = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn_t = F.softmax(attn_t, dim=-1)
        attn_t = self.drop(attn_t)
        temporal_out = (attn_t @ v_t).transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]

        # Combine
        out = spatial_out + temporal_out  # [B, 1, D]
        out = self.out_proj(out)
        out = out.squeeze(1)  # [B, D]

        return out


# ============ STAGE 3: HIERARCHICAL PROCESSING ============

class HierarchicalMLPBlock(nn.Module):
    """Stage 3: Hierarchical processing via residual MLP."""

    def __init__(self, dim, hidden_mult=2, dropout=0.1):
        super().__init__()
        hidden = int(dim * hidden_mult)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, D] → [B, D] (with residual connection)"""
        return x + self.net(x)


# ============ COMPLETE EEG ENCODER ============

class Stage123EEGEncoder(nn.Module):
    """Complete Stages 1-3 EEG encoder."""

    def __init__(self, eeg_dim=516, hidden_dim=64):
        super().__init__()

        # Stage 1
        self.alignment = EuclideanAlignment()
        self.tokenizer = EEGNetTokenizer(eeg_dim, hidden_dim)

        # Stage 2
        self.attention = CrissCrossAttention(hidden_dim, num_heads=4)

        # Stage 3
        self.hierarchical = HierarchicalMLPBlock(hidden_dim, hidden_mult=1.5, dropout=0.1)

    def forward(self, x):
        """x: [B, 516] → [B, hidden_dim]"""
        # Stage 1: Alignment → Tokenization
        x = self.alignment(x)
        x = self.tokenizer(x)  # [B, hidden_dim]

        # Stage 2: Criss-cross attention
        x = self.attention(x)  # [B, hidden_dim]

        # Stage 3: Hierarchical processing
        x = self.hierarchical(x)  # [B, hidden_dim]

        return x


# ============ MULTIMODAL FUSION ============

class Stage123Multimodal(nn.Module):
    """Stages 1-3 architecture with confidence-gated multimodal fusion."""

    def __init__(self, eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64):
        super().__init__()

        # EEG: Advanced encoder (all stages)
        self.eeg_encoder = Stage123EEGEncoder(eeg_dim, hidden_dim)

        # Eye & Behavior: Simple projections
        self.eye_proj = nn.Sequential(
            nn.Linear(eye_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.beh_proj = nn.Sequential(
            nn.Linear(beh_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Confidence heads
        self.conf_eeg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.conf_eye = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.conf_beh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Cross-modal refinement: EEG teaches others
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        # Fusion and classification
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, eeg, eye, beh):
        """
        Args:
            eeg: [B, 516]
            eye: [B, 11]
            beh: [B, 6]

        Returns:
            logits: [B, 2]
        """
        # Encode each modality
        eeg_feat = self.eeg_encoder(eeg)  # [B, hidden_dim]
        eye_feat = self.eye_proj(eye)     # [B, hidden_dim]
        beh_feat = self.beh_proj(beh)     # [B, hidden_dim]

        # Compute per-modality confidence
        conf_eeg = self.conf_eeg(eeg_feat)
        conf_eye = self.conf_eye(eye_feat)
        conf_beh = self.conf_beh(beh_feat)

        # Softmax gating (gates sum to 1)
        confs = torch.cat([conf_eeg, conf_eye, conf_beh], dim=1)  # [B, 3]
        gates = torch.softmax(confs, dim=1)  # [B, 3]

        # Weighted fusion
        fused = (gates[:, 0:1] * eeg_feat +
                 gates[:, 1:2] * eye_feat +
                 gates[:, 2:3] * beh_feat)  # [B, hidden_dim]

        # Cross-modal refinement: EEG guides weak modalities
        refine_gate = self.refine(eeg_feat)  # [B, hidden_dim]
        fused = fused * refine_gate

        # Classification
        fused = torch.relu(self.fusion_proj(fused))
        logits = self.classifier(fused)  # [B, 2]

        return logits


def create_model(eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64):
    """Factory function."""
    return Stage123Multimodal(eeg_dim, eye_dim, beh_dim, hidden_dim)
