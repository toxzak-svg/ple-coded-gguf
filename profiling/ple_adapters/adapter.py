# Phase 3: PLE Adapters — Low-rank adapters mapping PLE vectors → residuals
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for PLE adapters."""
    rank: int = 16
    ple_dim: int = 256  # PLE embedding dimension
    hidden_dim: int = 2048  # Main stream hidden dimension
    dropout: float = 0.0
    ple_dominant_lr_multiplier: float = 2.0  # Higher LR for PLE-dominant layers


class LowRankAdapter(nn.Module):
    """Low-rank adapter mapping PLE vectors → main stream residuals.
    
    Architecture: ple_input → Linear(rank) → Activation → Linear(rank→hidden) → residual
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(config.ple_dim, config.rank, bias=False)
        self.up_proj = nn.Linear(config.rank, config.hidden_dim, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
    
    def forward(self, ple_vector: torch.Tensor) -> torch.Tensor:
        """Apply adapter: ple_vector → residual correction."""
        hidden = self.down_proj(ple_vector)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        residual = self.up_proj(hidden)
        return residual


class PLEAdapter(nn.Module):
    """Container for per-layer PLE adapters."""
    
    def __init__(self, config: AdapterConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict()
        
        for layer_idx in range(num_layers):
            self.adapters[str(layer_idx)] = LowRankAdapter(config)
    
    def forward(self, ple_vectors: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Apply adapters to PLE vectors.
        
        Args:
            ple_vectors: dict mapping layer_idx → ple_vector (batch, ple_dim)
            
        Returns:
            dict mapping layer_idx → residual_correction (batch, hidden_dim)
        """
        corrections = {}
        for layer_idx, ple_vec in ple_vectors.items():
            corrections[layer_idx] = self.adapters[str(layer_idx)](ple_vec)
        return corrections
    
    def get_adapter(self, layer_idx: int) -> LowRankAdapter:
        return self.adapters[str(layer_idx)]


class AdapterTuner:
    """Fine-tunes PLE adapters on calibration data."""
    
    def __init__(
        self,
        model: nn.Module,
        adapter: PLEAdapter,
        ple_dominance_scores: dict[int, float],
        config: AdapterConfig,
    ):
        self.model = model
        self.adapter = adapter
        self.ple_dominance_scores = ple_dominance_scores
        self.config = config
        self.device = next(model.parameters()).device
    
    def compute_adapter_loss(
        self,
        ple_vector: torch.Tensor,
        target_residual: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute loss for adapter at a given layer."""
        pred_residual = self.adapter.adapters[str(layer_idx)](ple_vector)
        loss = nn.functional.mse_loss(pred_residual, target_residual)
        return loss
    
    def fine_tune_adapters(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
        lr: float = 1e-4,
        ple_dominant_weight: float = 2.0,
    ) -> dict[int, float]:
        """Fine-tune all adapters with higher weight on PLE-dominant layers."""
        optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        
        layer_losses = {}
        
        for epoch in range(num_epochs):
            epoch_losses = {}
            
            self.model.eval()
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
                
                if hidden_states is None:
                    continue
                
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    if layer_idx == 0:
                        continue
                    
                    is_ple_dominant = self.ple_dominance_scores.get(layer_idx, 0.0) >= 0.5
                    weight = ple_dominant_weight if is_ple_dominant else 1.0
                    
                    if layer_idx not in epoch_losses:
                        epoch_losses[layer_idx] = []
                    
                    # Simplified: use hidden state as proxy for residual
                    target = layer_hidden
                    ple_vec = torch.randn(layer_hidden.shape[0], self.config.ple_dim).to(self.device)
                    
                    loss = self.compute_adapter_loss(ple_vec, target, layer_idx)
                    weighted_loss = weight * loss
                    
                    weighted_loss.backward()
                    epoch_losses[layer_idx].append(loss.item())
            
            optimizer.step()
            optimizer.zero_grad()
            
            for layer_idx in epoch_losses:
                avg_loss = sum(epoch_losses[layer_idx]) / len(epoch_losses[layer_idx])
                if layer_idx not in layer_losses:
                    layer_losses[layer_idx] = []
                layer_losses[layer_idx].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} complete")
        
        return layer_losses


def create_ple_adapters(
    num_layers: int,
    ple_dim: int = 256,
    hidden_dim: int = 2048,
    rank: int = 16,
) -> tuple[PLEAdapter, AdapterConfig]:
    """Factory to create PLE adapters for a model."""
    config = AdapterConfig(
        rank=rank,
        ple_dim=ple_dim,
        hidden_dim=hidden_dim,
    )
    adapter = PLEAdapter(config, num_layers)
    return adapter, config


def fine_tune_adapters(
    model: nn.Module,
    adapter: PLEAdapter,
    ple_dominance_scores: dict[int, float],
    dataloader: DataLoader,
    config: Optional[AdapterConfig] = None,
    num_epochs: int = 3,
    lr: float = 1e-4,
) -> dict[int, float]:
    """Convenience function to fine-tune adapters."""
    if config is None:
        config = AdapterConfig()
    
    tuner = AdapterTuner(model, adapter, ple_dominance_scores, config)
    return tuner.fine_tune_adapters(dataloader, num_epochs, lr)


def save_adapters(adapter: PLEAdapter, output_path: Path):
    """Save adapter weights to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), output_path)
    logger.info(f"Adapters saved to {output_path}")


def load_adapters(adapter: PLEAdapter, input_path: Path) -> PLEAdapter:
    """Load adapter weights from disk."""
    adapter.load_state_dict(torch.load(input_path))
    logger.info(f"Adapters loaded from {input_path}")
    return adapter
