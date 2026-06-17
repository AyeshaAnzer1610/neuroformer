"""
NeuroFormer: Transformer-Based Multimodal Integration for Mental Health Diagnosis

Main model architecture implementing:
- Modality-specific transformer encoders
- Cross-modal fusion transformer
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ModalityEncoder(nn.Module):
    """Modality-specific transformer encoder."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        max_len: int = 5000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # Create attention mask if padding mask provided
        if mask is not None:
            # Convert padding mask to attention mask
            # mask: True for padded positions
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return x


class CrossModalFusionTransformer(nn.Module):
    """Cross-modal fusion transformer for integrating multiple modalities."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        n_modalities: int = 2
    ):
        super().__init__()

        self.d_model = d_model
        self.n_modalities = n_modalities
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Modality type embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(n_modalities, d_model)
        )
        
        # Positional encoding for concatenated sequence
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        encoded_modalities: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded_modalities: Dictionary of encoded modality tensors
                Each tensor has shape (batch_size, seq_len, d_model)
            modality_masks: Optional dictionary of padding masks
        
        Returns:
            cls_output: CLS token representation (batch_size, d_model)
            full_output: Full sequence output (batch_size, total_seq_len, d_model)
        """
        batch_size = next(iter(encoded_modalities.values())).size(0)
        device = next(iter(encoded_modalities.values())).device
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate all modalities with modality type embeddings
        sequences = [cls_tokens]
        full_mask = []
        
        modality_names = ['eeg', 'eyetracking']
        for idx, modality_name in enumerate(modality_names):
            if modality_name in encoded_modalities:
                modality_seq = encoded_modalities[modality_name]
                seq_len = modality_seq.size(1)
                
                # Add modality type embedding
                modality_emb = self.modality_embeddings[idx].view(1, 1, -1)
                modality_emb = modality_emb.expand(batch_size, seq_len, -1)
                modality_seq = modality_seq + modality_emb
                
                sequences.append(modality_seq)
                
                # Handle mask
                if modality_masks is not None and modality_name in modality_masks:
                    full_mask.append(modality_masks[modality_name])
                else:
                    full_mask.append(
                        torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                    )
        
        # Concatenate along sequence dimension
        x = torch.cat(sequences, dim=1)  # (batch_size, total_seq_len, d_model)
        
        # Concatenate masks (CLS token is never masked)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        if modality_masks is not None:
            combined_mask = torch.cat([cls_mask] + full_mask, dim=1)
        else:
            combined_mask = None
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply fusion transformer
        x = self.fusion_transformer(
            x,
            src_key_padding_mask=combined_mask
        )
        
        # Extract CLS token
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        
        return cls_output, x


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool over the sequence dim, ignoring padded positions.

    Args:
        x: (batch, seq, d)
        mask: (batch, seq) with True for *padded* positions, or None.
    Returns:
        (batch, d)
    """
    if mask is None:
        return x.mean(dim=1)
    keep = (~mask).float().unsqueeze(-1)          # (batch, seq, 1)
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (x * keep).sum(dim=1) / denom


class CrossModalAlignment(nn.Module):
    """Optional representation-space alignment across modalities.

    Projects each modality's pooled encoding into a shared, L2-normalized space
    and exposes a symmetric InfoNCE contrastive loss that pulls together the
    different modalities of the *same* subject and pushes apart different subjects.

    This is **off by default** in NeuroFormer. It is provided as a robustness /
    completeness mechanism (and a basis for cross-cohort transfer); on the small
    MIPDB complete-case cohort (N~104) it is not expected to raise accuracy and is
    not used for the headline leak-free results. See docs/resubmission/Methodology.md
    (section E) for the rationale.
    """

    def __init__(self, d_model: int, d_align: int = 64, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        # One projection head per modality into the shared space.
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_align),
            )
            for name in ('eeg', 'eyetracking')
        })

    def embed(self, pooled: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map pooled per-modality encodings to L2-normalized shared embeddings."""
        return {
            name: F.normalize(self.proj[name](vec), dim=-1)
            for name, vec in pooled.items()
        }

    def contrastive_loss(
        self,
        embeddings: Dict[str, torch.Tensor],
        availability: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Symmetric InfoNCE over every available modality pair in the batch.

        Positives = same row (subject) across two modalities; negatives = other
        rows. Pairs are only counted for rows where *both* modalities are present
        (per `availability`, a dict of (batch,) bool tensors; None => all present).
        Returns a scalar (0 if no valid pair has >=2 usable rows).
        """
        names = [n for n in ('eeg', 'eyetracking') if n in embeddings]
        device = next(iter(embeddings.values())).device
        total = torch.zeros((), device=device)
        n_pairs = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                za, zb = embeddings[a], embeddings[b]
                if availability is not None:
                    valid = availability.get(a, None)
                    vb = availability.get(b, None)
                    if valid is not None and vb is not None:
                        valid = valid & vb
                    elif vb is not None:
                        valid = vb
                    if valid is not None:
                        za, zb = za[valid], zb[valid]
                if za.size(0) < 2:
                    continue
                logits = za @ zb.t() / self.temperature   # (n, n)
                targets = torch.arange(za.size(0), device=device)
                loss = 0.5 * (F.cross_entropy(logits, targets)
                              + F.cross_entropy(logits.t(), targets))
                total = total + loss
                n_pairs += 1
        return total / n_pairs if n_pairs > 0 else total


class NeuroFormer(nn.Module):
    """
    NeuroFormer: Transformer-based multimodal framework for mental health diagnosis.

    Integrates EEG and eye-tracking data through:
    1. Modality-specific transformer encoders
    2. Cross-modal fusion transformer
    3. Classification head
    4. (optional) cross-modal representation-space alignment
    """
    
    def __init__(
        self,
        eeg_input_dim: int = 128,
        eyetracking_input_dim: int = 96,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_fusion_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        n_classes: int = 2,
        use_auxiliary_loss: bool = False,
        use_alignment: bool = False,
        d_align: int = 64,
        align_temperature: float = 0.07
    ):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_alignment = use_alignment
        
        # Modality-specific encoders
        self.eeg_encoder = ModalityEncoder(
            input_dim=eeg_input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.eyetracking_encoder = ModalityEncoder(
            input_dim=eyetracking_input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Cross-modal fusion transformer
        self.fusion_transformer = CrossModalFusionTransformer(
            d_model=d_model,
            n_layers=n_fusion_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_modalities=2
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        # Auxiliary reconstruction decoders (for NeuroFormer++)
        if use_auxiliary_loss:
            self.eeg_decoder = nn.Linear(d_model, eeg_input_dim)
            self.eyetracking_decoder = nn.Linear(d_model, eyetracking_input_dim)

        # Optional cross-modal representation-space alignment (off by default).
        if use_alignment:
            self.alignment = CrossModalAlignment(
                d_model=d_model, d_align=d_align, temperature=align_temperature
            )
    
    def forward(
        self,
        eeg: torch.Tensor,
        eyetracking: torch.Tensor,
        eeg_mask: Optional[torch.Tensor] = None,
        eyetracking_mask: Optional[torch.Tensor] = None,
        availability: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NeuroFormer.

        Args:
            eeg: EEG features (batch_size, seq_len, eeg_dim)
            eyetracking: Eye-tracking features (batch_size, seq_len, eye_dim)
            *_mask: Optional padding masks for each modality
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - logits: Classification logits (batch_size, n_classes)
                - probs: Class probabilities (batch_size, n_classes)
                - reconstructions: (optional) Reconstructed inputs for auxiliary loss
        """
        # Encode each modality
        eeg_encoded = self.eeg_encoder(eeg, eeg_mask)
        eyetracking_encoded = self.eyetracking_encoder(eyetracking, eyetracking_mask)

        # Prepare encoded modalities dictionary
        encoded_modalities = {
            'eeg': eeg_encoded,
            'eyetracking': eyetracking_encoded
        }

        # Prepare masks dictionary
        modality_masks = {}
        if eeg_mask is not None:
            modality_masks['eeg'] = eeg_mask
        if eyetracking_mask is not None:
            modality_masks['eyetracking'] = eyetracking_mask
        
        # Cross-modal fusion
        cls_output, full_output = self.fusion_transformer(
            encoded_modalities,
            modality_masks if modality_masks else None
        )
        
        # Classification
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=-1)
        
        # Prepare output dictionary
        output = {
            'logits': logits,
            'probs': probs,
            'cls_embedding': cls_output
        }
        
        # Auxiliary reconstructions if enabled
        if self.use_auxiliary_loss:
            output['reconstructions'] = {
                'eeg': self.eeg_decoder(eeg_encoded),
                'eyetracking': self.eyetracking_decoder(eyetracking_encoded)
            }

        # Optional cross-modal alignment: shared-space embeddings + contrastive loss.
        if self.use_alignment:
            pooled = {
                'eeg': _masked_mean(eeg_encoded, eeg_mask),
                'eyetracking': _masked_mean(eyetracking_encoded, eyetracking_mask),
            }
            align_emb = self.alignment.embed(pooled)
            output['alignment_embeddings'] = align_emb
            output['alignment_loss'] = self.alignment.contrastive_loss(
                align_emb, availability
            )

        return output
    
    def get_attention_weights(
        self,
        eeg: torch.Tensor,
        eyetracking: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability analysis."""
        # This requires modifying the forward pass to store attention weights
        # Implementation depends on PyTorch version and specific requirements
        raise NotImplementedError("Attention weight extraction requires custom hooks")


def create_neuroformer(config: dict) -> NeuroFormer:
    """Factory function to create NeuroFormer model from config."""
    return NeuroFormer(
        eeg_input_dim=config.get('eeg_input_dim', 128),
        eyetracking_input_dim=config.get('eyetracking_input_dim', 96),
        d_model=config.get('d_model', 256),
        n_encoder_layers=config.get('n_encoder_layers', 4),
        n_fusion_layers=config.get('n_fusion_layers', 6),
        n_heads=config.get('n_heads', 8),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.3),
        n_classes=config.get('n_classes', 2),
        use_auxiliary_loss=config.get('use_auxiliary_loss', False),
        use_alignment=config.get('use_alignment', False),
        d_align=config.get('d_align', 64),
        align_temperature=config.get('align_temperature', 0.07)
    )


if __name__ == "__main__":
    batch_size = 4
    seq_len = 300
    eeg = torch.randn(batch_size, seq_len, 128)
    eyetracking = torch.randn(batch_size, seq_len, 96)

    # Baseline (alignment off — headline configuration)
    model = NeuroFormer()
    output = model(eeg, eyetracking)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Alignment-enabled variant (optional capability)
    model_a = NeuroFormer(use_alignment=True)
    out_a = model_a(eeg, eyetracking)
    print(f"Alignment loss: {out_a['alignment_loss'].item():.4f}")
    print(f"Shared-embedding dim: {out_a['alignment_embeddings']['eeg'].shape[-1]}")
    # With an availability mask (some modalities missing per subject)
    avail = {
        'eeg': torch.tensor([True, True, True, True]),
        'eyetracking': torch.tensor([True, True, False, True]),
    }
    out_b = model_a(eeg, eyetracking, availability=avail)
    print(f"Alignment loss (masked): {out_b['alignment_loss'].item():.4f}")
