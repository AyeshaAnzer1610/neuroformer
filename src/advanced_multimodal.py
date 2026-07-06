import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class EuclideanAlignment(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, p=0, d=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_c, in_c, k, padding=p, dilation=d, groups=in_c)
        self.pointwise = nn.Conv1d(in_c, out_c, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class EEGNetTokenizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.temporal_conv = nn.Sequential(nn.Conv1d(1, hidden_dim, 3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.spatial_conv = DepthwiseSeparableConv1d(hidden_dim, hidden_dim, 3, p=1)
        self.spatial_norm = nn.BatchNorm1d(hidden_dim)
        self.sep_conv = DepthwiseSeparableConv1d(hidden_dim, output_dim, 1)
        self.sep_norm = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = F.relu(self.spatial_norm(x))
        x = F.relu(self.sep_norm(self.sep_conv(x)))
        return x.mean(dim=-1)

class CrissCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q_spatial, self.k_spatial, self.v_spatial = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q_temporal, self.k_temporal, self.v_temporal = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        is_1d = len(x.shape) == 2
        if is_1d: x = x.unsqueeze(1)
        B, L, D = x.shape
        x = x + self.pos_embed
        q_s = self.q_spatial(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_s = self.k_spatial(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = self.v_spatial(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_s = F.softmax((q_s @ k_s.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        spatial_out = (attn_s @ v_s).transpose(1, 2).reshape(B, L, D)
        q_t = self.q_temporal(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_temporal(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_temporal(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_t = F.softmax((q_t @ k_t.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        temporal_out = (attn_t @ v_t).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(spatial_out + temporal_out)
        return out.squeeze(1) if is_1d else out

class FrequencyAwareAttention(nn.Module):
    def __init__(self, dim, num_bands=4, num_heads=4):
        super().__init__()
        self.num_bands, self.band_dim = num_bands, dim // num_bands
        self.band_attentions = nn.ModuleList([CrissCrossAttention(self.band_dim, num_heads=2) for _ in range(num_bands)])
        self.band_fusion = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_bands), nn.Softmax(dim=-1))
    def forward(self, x):
        bands = x.reshape(x.shape[0], self.num_bands, self.band_dim)
        processed_bands = [self.band_attentions[i](bands[:, i, :]) for i in range(self.num_bands)]
        processed = torch.stack(processed_bands, dim=1)          # (B, num_bands, band_dim)
        fusion_weights = self.band_fusion(x).unsqueeze(-1)       # (B, num_bands, 1)
        # Recombine the weighted per-band representations back to the full hidden
        # dim by concatenation. (Previously .sum(dim=1) collapsed to band_dim and
        # broke the downstream hidden_dim Linear -> create_model crashed for any
        # input; fixed 2026-07-06 to make NeuroFormer runnable.)
        return (processed * fusion_weights).reshape(x.shape[0], -1)  # (B, num_bands*band_dim = dim)

class HierarchicalEEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.alignment = EuclideanAlignment()
        self.tokenizer = EEGNetTokenizer(input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.criss_cross = CrissCrossAttention(hidden_dim, num_heads=num_heads)
        self.freq_aware = FrequencyAwareAttention(hidden_dim, num_bands=4, num_heads=2)
        self.hierarchical = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.output_dim = hidden_dim
    def forward(self, x):
        x = self.alignment(x)
        x = self.tokenizer(x)
        x = self.criss_cross(x)
        x = self.freq_aware(x)
        residual = x
        x = self.hierarchical(x)
        return x + residual

class MultimodalFusionWithConfidence(nn.Module):
    def __init__(self, eeg_dim, eye_dim, beh_dim, hidden_dim=64):
        super().__init__()
        self.eeg_encoder = HierarchicalEEGEncoder(eeg_dim, hidden_dim=hidden_dim)
        self.eye_proj = nn.Sequential(nn.Linear(eye_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.beh_proj = nn.Sequential(nn.Linear(beh_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.confidence_heads = nn.ModuleDict({'eeg': nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1), nn.Sigmoid()), 'eye': nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1), nn.Sigmoid()), 'beh': nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())})
        self.cross_modal_refine = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, hidden_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, eeg, eye, beh):
        eeg_feat = self.eeg_encoder(eeg)
        eye_feat = self.eye_proj(eye)
        beh_feat = self.beh_proj(beh)
        diag_pattern = self.cross_modal_refine(eeg_feat)
        eye_feat, beh_feat = eye_feat * diag_pattern, beh_feat * diag_pattern
        confidences = {'eeg': self.confidence_heads['eeg'](eeg_feat), 'eye': self.confidence_heads['eye'](eye_feat), 'beh': self.confidence_heads['beh'](beh_feat)}
        conf_stack = torch.cat([confidences[m] for m in ['eeg', 'eye', 'beh']], dim=1)
        gates = torch.softmax(conf_stack, dim=1)
        fused = gates[:, 0:1] * eeg_feat + gates[:, 1:2] * eye_feat + gates[:, 2:3] * beh_feat
        return self.classifier(self.fusion(fused)), confidences

def create_model(eeg_dim=516, eye_dim=11, beh_dim=6, hidden_dim=64):
    return MultimodalFusionWithConfidence(eeg_dim, eye_dim, beh_dim, hidden_dim)
