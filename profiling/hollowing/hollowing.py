# Phase 2: Hollowing — Structured pruning + quantization for PLE-dominant blocks
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class HollowingConfig:
    """Configuration for backbone hollowing."""
    # Pruning
    prune_block_size: int = 64
    prune_threshold: float = 0.5  # PLE dominance threshold for aggressive pruning
    
    # Quantization
    q2_bits: int = 2
    q3_bits: int = 3
    q4_bits: int = 4
    q5_bits: int = 5
    
    # Which quant type per PLE dominance level
    ple_dominant_quant: str = "Q2"  # Q2 or Q3 for PLE-subsidized blocks
    backbone_quant: str = "Q4"       # Q4 for non-PLE-dominant blocks
    
    # Target compression ratio in PLE-dominant zones
    target_compression: float = 0.4  # 40% of original size (2.5x smaller)


@dataclass
class BlockMask:
    """Mask for a pruned weight block."""
    block_idx: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    ple_subsidized: bool


class StructuredPruner:
    """Prunes weight blocks in PLE-dominant regions."""
    
    def __init__(self, config: HollowingConfig):
        self.config = config
        self.masks: dict[str, list[BlockMask]] = {}
    
    def compute_ple_subsidy_map(
        self,
        ple_dominance_scores: dict[int, float],
        num_layers: int,
    ) -> dict[int, bool]:
        """Map each layer to whether it's PLE-subsidized (aggressively prunable)."""
        subsidy_map = {}
        for layer_idx in range(num_layers):
            score = ple_dominance_scores.get(layer_idx, 0.0)
            subsidy_map[layer_idx] = score >= self.config.prune_threshold
        return subsidy_map
    
    def prune_weight_block(
        self,
        weight: torch.Tensor,
        block_size: int,
        ple_attribution: torch.Tensor,
    ) -> tuple[torch.Tensor, list[BlockMask]]:
        """Identify and mark prunable blocks based on PLE attribution."""
        rows, cols = weight.shape
        masks = []
        
        # For attention weights (4D: heads, seq, seq, hidden) vs MLP (2D)
        if weight.dim() == 4:
            # Attention: (num_heads, seq_len, seq_len, hidden_dim)
            # Prune entire head blocks
            num_heads = weight.shape[0]
            hidden_dim = weight.shape[3]
            
            for head_idx in range(num_heads):
                head_attr = ple_attribution[head_idx] if ple_attribution.numel() > head_idx else 0.0
                if head_attr >= self.config.prune_threshold:
                    mask = BlockMask(
                        block_idx=head_idx,
                        row_start=head_idx,
                        row_end=head_idx + 1,
                        col_start=0,
                        col_end=hidden_dim,
                        ple_subsidized=True,
                    )
                    masks.append(mask)
        else:
            # MLP or dense: (rows, cols)
            num_row_blocks = (rows + block_size - 1) // block_size
            num_col_blocks = (cols + block_size - 1) // block_size
            
            for r_block in range(num_row_blocks):
                r_start = r_block * block_size
                r_end = min(r_start + block_size, rows)
                
                for c_block in range(num_col_blocks):
                    c_start = c_block * block_size
                    c_end = min(c_start + block_size, cols)
                    
                    block_attr = ple_attribution[r_block, c_block] if ple_attribution.numel() > r_block * num_col_blocks + c_block else 0.0
                    if block_attr >= self.config.prune_threshold:
                        mask = BlockMask(
                            block_idx=r_block * num_col_blocks + c_block,
                            row_start=r_start,
                            row_end=r_end,
                            col_start=c_start,
                            col_end=c_end,
                            ple_subsidized=True,
                        )
                        masks.append(mask)
        
        return weight, masks
    
    def prune_ple_dominant_blocks(
        self,
        model: nn.Module,
        ple_dominance_scores: dict[int, float],
    ) -> dict[str, list[BlockMask]]:
        """Prune weight blocks in PLE-dominant layers."""
        subsidy_map = self.compute_ple_subsidy_map(
            ple_dominance_scores,
            num_layers=len(ple_dominance_scores),
        )
        
        masks = {}
        
        for name, module in model.named_modules():
            if not any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                continue
            
            layer_num = None
            for part in name.split("."):
                if part.isdigit():
                    layer_num = int(part)
                    break
            
            if layer_num is None or not subsidy_map.get(layer_num, False):
                continue
            
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight
                ple_attr = torch.rand(weight.shape[0]) * 0.3 + 0.5  # Simulated
                
                _, block_masks = self.prune_weight_block(
                    weight,
                    self.config.prune_block_size,
                    ple_attr,
                )
                
                if block_masks:
                    masks[name] = block_masks
                    logger.info(f"Layer {layer_num} ({name}): {len(block_masks)} prunable blocks")
        
        self.masks = masks
        return masks


class Quantizer:
    """Quantizes weight matrices to target bit widths."""
    
    def __init__(self, config: HollowingConfig):
        self.config = config
    
    def quantize_q2(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize to 2-bit with scale and zero-point."""
        max_val = 1.87
        scale = weight.abs().max() / max_val + 1e-8
        quantized = torch.clamp(torch.round(weight / scale), -2, 1)
        return quantized, scale, torch.zeros_like(scale)
    
    def quantize_q3(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize to 3-bit with scale and zero-point."""
        max_val = 3.5
        scale = weight.abs().max() / max_val + 1e-8
        quantized = torch.clamp(torch.round(weight / scale), -4, 3)
        return quantized, scale, torch.zeros_like(scale)
    
    def quantize_q4(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize to 4-bit with scale and zero-point."""
        max_val = 7.0
        scale = weight.abs().max() / max_val + 1e-8
        quantized = torch.clamp(torch.round(weight / scale), -8, 7)
        return quantized, scale, torch.zeros_like(scale)
    
    def quantize_q5(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize to 5-bit with scale and zero-point."""
        max_val = 15.0
        scale = weight.abs().max() / max_val + 1e-8
        quantized = torch.clamp(torch.round(weight / scale), -16, 15)
        return quantized, scale, torch.zeros_like(scale)
    
    def quantize_weight(
        self,
        weight: torch.Tensor,
        ple_subsidized: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Quantize a weight matrix based on PLE subsidy status."""
        if ple_subsidized:
            if self.config.ple_dominant_quant == "Q2":
                q, s, z = self.quantize_q2(weight)
            else:
                q, s, z = self.quantize_q3(weight)
            quant_type = self.config.ple_dominant_quant
        else:
            if self.config.backbone_quant == "Q4":
                q, s, z = self.quantize_q4(weight)
            else:
                q, s, z = self.quantize_q5(weight)
            quant_type = self.config.backbone_quant
        
        return q, s, z, quant_type
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize back to FP16."""
        return quantized.float() * scale.float() + zero_point.float()


@dataclass
class HollowedWeight:
    """Container for a hollowed (pruned + quantized) weight matrix."""
    name: str
    original_shape: tuple[int, ...]
    quantized_data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    quant_type: str
    block_masks: list[BlockMask]
    ple_subsidized: bool
    compression_ratio: float


class HollowingEngine:
    """Orchestrates the full hollowing pipeline."""
    
    def __init__(self, config: HollowingConfig):
        self.config = config
        self.pruner = StructuredPruner(config)
        self.quantizer = Quantizer(config)
        self.hollowed_weights: dict[str, HollowedWeight] = {}
    
    def hollow_model(
        self,
        model: nn.Module,
        ple_dominance_scores: dict[int, float],
    ) -> dict[str, HollowedWeight]:
        """Full hollowing: prune PLE-dominant blocks + quantize."""
        
        logger.info("Phase 2: Hollowing model backbone")
        
        # Step 1: Identify prunable blocks
        masks = self.pruner.prune_ple_dominant_blocks(model, ple_dominance_scores)
        
        # Step 2: Quantize weights
        for name, module in model.named_modules():
            if not any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                continue
            
            if not hasattr(module, "weight") or module.weight is None:
                continue
            
            layer_num = None
            for part in name.split("."):
                if part.isdigit():
                    layer_num = int(part)
                    break
            
            ple_subsidized = ple_dominance_scores.get(layer_num, 0.0) >= self.config.prune_threshold
            
            weight = module.weight.data
            quantized, scale, zero_point, quant_type = self.quantizer.quantize_weight(weight, ple_subsidized)
            
            # Compute compression ratio
            if ple_subsidized:
                orig_bits = weight.numel() * 16
                quant_bits = quantized.numel() * self.config.q2_bits + scale.numel() * 16
                compression = quant_bits / orig_bits
            else:
                orig_bits = weight.numel() * 16
                quant_bits = quantized.numel() * self.config.q4_bits + scale.numel() * 16
                compression = quant_bits / orig_bits
            
            self.hollowed_weights[name] = HollowedWeight(
                name=name,
                original_shape=weight.shape,
                quantized_data=quantized,
                scale=scale,
                zero_point=zero_point,
                quant_type=quant_type,
                block_masks=masks.get(name, []),
                ple_subsidized=ple_subsidized,
                compression_ratio=compression,
            )
            
            logger.info(f"  {name}: {quant_type}, compression={compression:.2%}")
        
        return self.hollowed_weights
    
    def save_hollowed_weights(self, output_path: Path):
        """Save hollowed weights to disk for GGUF encoding."""
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {}
        for name, hw in self.hollowed_weights.items():
            serializable[name] = {
                "original_shape": list(hw.original_shape),
                "quant_type": hw.quant_type,
                "ple_subsidized": hw.ple_subsidized,
                "compression_ratio": float(hw.compression_ratio),
                "block_masks": [
                    {
                        "block_idx": m.block_idx,
                        "row_start": m.row_start,
                        "row_end": m.row_end,
                        "col_start": m.col_start,
                        "col_end": m.col_end,
                        "ple_subsidized": m.ple_subsidized,
                    }
                    for m in hw.block_masks
                ],
            }
        
        results = {
            "config": {
                "prune_block_size": self.config.prune_block_size,
                "prune_threshold": self.config.prune_threshold,
                "ple_dominant_quant": self.config.ple_dominant_quant,
                "backbone_quant": self.config.backbone_quant,
                "target_compression": self.config.target_compression,
            },
            "weights": serializable,
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Hollowing results saved to {output_path}")


def run_hollowing(
    model: nn.Module,
    ple_dominance_scores: dict[int, float],
    config: Optional[HollowingConfig] = None,
) -> dict[str, HollowedWeight]:
    """Convenience function to run full hollowing pipeline."""
    if config is None:
        config = HollowingConfig()
    
    engine = HollowingEngine(config)
    return engine.hollow_model(model, ple_dominance_scores)
