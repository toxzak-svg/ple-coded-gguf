# Phase 4: GGUF Encoding — Two-plane GGUF format for PLE-Coded models
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GGUFConfig:
    """Configuration for two-plane GGUF encoding."""
    # Quantization settings
    q2_bits: int = 2
    q3_bits: int = 3
    q4_bits: int = 4
    q5_bits: int = 5
    
    # Metadata flags
    ple_dominant_quant: str = "Q2_PLE"
    backbone_quant: str = "Q4_K_M"
    
    # Two-plane structure
    use_two_plane: bool = True


class TwoPlaneGGUF:
    """Two-plane GGUF container: backbone plane + PLE plane."""
    
    BACKBONE_PLANE = "backbone"
    PLE_PLANE = "ple"
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.backbone_plane: dict[str, np.ndarray] = {}
        self.ple_plane: dict[str, np.ndarray] = {}
        self.metadata: dict[str, str] = {}
    
    def add_backbone_block(
        self,
        name: str,
        quantized_weights: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        quant_type: str,
        block_masks: list[dict],
    ):
        """Add a quantized backbone weight block."""
        self.backbone_plane[name] = {
            "weights": quantized_weights,
            "scales": scales,
            "zero_points": zero_points,
            "quant_type": quant_type,
            "block_masks": block_masks,
        }
    
    def add_ple_block(
        self,
        name: str,
        ple_embeddings: np.ndarray,
        ple_adapters: Optional[np.ndarray] = None,
    ):
        """Add a PLE plane block."""
        self.ple_plane[name] = {
            "ple_embeddings": ple_embeddings,
            "ple_adapters": ple_adapters,
        }
    
    def set_metadata(self, key: str, value: str):
        self.metadata[key] = value
    
    def get_memory_footprint(self) -> dict[str, int]:
        """Calculate memory footprint of each plane."""
        backbone_bytes = 0
        for block in self.backbone_plane.values():
            weights = block["weights"]
            scales = block["scales"]
            backbone_bytes += weights.nbytes + scales.nbytes
        
        ple_bytes = 0
        for block in self.ple_plane.values():
            ple_bytes += block["ple_embeddings"].nbytes
            if block["ple_adapters"] is not None:
                ple_bytes += block["ple_adapters"].nbytes
        
        return {
            "backbone_bytes": backbone_bytes,
            "ple_bytes": ple_bytes,
            "total_bytes": backbone_bytes + ple_bytes,
        }


class GGUFEncoder:
    """Encodes PLE-Coded models into two-plane GGUF format."""
    
    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.two_plane: Optional[TwoPlaneGGUF] = None
    
    def encode_quantized_block(
        self,
        weight_tensor: torch.Tensor,
        quant_type: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a weight tensor to target quantization format."""
        weight_fp32 = weight_tensor.float()
        weight_np = weight_fp32.numpy()
        
        max_val = {
            "Q2_PLE": 1.87,
            "Q3_PLE": 3.5,
            "Q4_K_M": 7.0,
            "Q5_K_M": 15.0,
        }.get(quant_type, 7.0)
        
        scale = np.abs(weight_np).max() / max_val + 1e-8
        
        if "Q2" in quant_type:
            num_levels = 2
        elif "Q3" in quant_type:
            num_levels = 3
        elif "Q4" in quant_type:
            num_levels = 4
        else:
            num_levels = 5
        
        min_val = -(num_levels // 2)
        max_val_allowed = num_levels // 2 - 1 if num_levels > 2 else num_levels - 1
        
        quantized_np = np.clip(np.round(weight_np / scale), min_val, max_val_allowed)
        scale_np = np.array([scale], dtype=np.float32)
        zero_np = np.zeros_like(scale_np)
        
        return quantized_np.astype(np.int8), scale_np, zero_np
    
    def create_two_plane_gguf(
        self,
        hollowed_weights: dict,
        ple_adapters: dict,
        ple_embeddings: dict,
    ) -> TwoPlaneGGUF:
        """Create a two-plane GGUF structure from hollowed model + adapters."""
        two_plane = TwoPlaneGGUF(self.config)
        
        # Add backbone plane blocks
        for name, hw in hollowed_weights.items():
            q_weights, scales, zero_points = self.encode_quantized_block(
                hw.quantized_data.float(),
                hw.quant_type,
            )
            
            block_masks = []
            for m in hw.block_masks:
                block_masks.append({
                    "block_idx": m.block_idx,
                    "row_start": m.row_start,
                    "row_end": m.row_end,
                    "ple_subsidized": m.ple_subsidized,
                })
            
            two_plane.add_backbone_block(
                name=name,
                quantized_weights=q_weights,
                scales=scales,
                zero_points=zero_points,
                quant_type=hw.quant_type,
                block_masks=block_masks,
            )
        
        # Add PLE plane blocks
        for layer_idx, ple_emb in ple_embeddings.items():
            ple_emb_np = ple_emb.numpy() if isinstance(ple_emb, torch.Tensor) else ple_emb
            adapter_data = ple_adapters.get(layer_idx)
            
            two_plane.add_ple_block(
                name=f"ple_layer_{layer_idx}",
                ple_embeddings=ple_emb_np,
                ple_adapters=adapter_data.numpy() if adapter_data is not None and isinstance(adapter_data, torch.Tensor) else None,
            )
        
        return two_plane
    
    def write_gguf_file(self, two_plane: TwoPlaneGGUF, output_path: Path):
        """Write two-plane GGUF to binary file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            # Write GGUF magic header
            f.write(struct.pack("<I", self.GGUF_MAGIC))
            
            # Write version
            f.write(struct.pack("<I", 3))  # GGUF version 3
            
            # Write metadata
            metadata_keys = list(two_plane.metadata.keys())
            f.write(struct.pack("<I", len(metadata_keys)))
            for key in metadata_keys:
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<I", len(key_bytes)))
                f.write(key_bytes)
                value_bytes = two_plane.metadata[key].encode("utf-8")
                f.write(struct.pack("<I", len(value_bytes)))
                f.write(value_bytes)
            
            # Write backbone plane
            backbone_blocks = list(two_plane.backbone_plane.items())
            f.write(struct.pack("<I", len(backbone_blocks)))
            for name, block in backbone_blocks:
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                
                f.write(struct.pack("<I", block["weights"].size))
                f.write(block["weights"].tobytes())
                
                f.write(struct.pack("<I", block["scales"].size))
                f.write(block["scales"].tobytes())
                
                f.write(struct.pack("<I", block["zero_points"].size))
                f.write(block["zero_points"].tobytes())
                
                qtype_bytes = block["quant_type"].encode("utf-8")
                f.write(struct.pack("<B", len(qtype_bytes)))
                f.write(qtype_bytes)
            
            # Write PLE plane
            ple_blocks = list(two_plane.ple_plane.items())
            f.write(struct.pack("<I", len(ple_blocks)))
            for name, block in ple_blocks:
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                
                f.write(struct.pack("<I", block["ple_embeddings"].size))
                f.write(block["ple_embeddings"].tobytes())
                
                has_adapters = block["ple_adapters"] is not None
                f.write(struct.pack("<?", has_adapters))
                if has_adapters:
                    f.write(struct.pack("<I", block["ple_adapters"].size))
                    f.write(block["ple_adapters"].tobytes())
        
        logger.info(f"GGUF file written to {output_path}")
        
        footprint = two_plane.get_memory_footprint()
        logger.info(f"  Backbone: {footprint['backbone_bytes'] / 1024 / 1024:.2f} MB")
        logger.info(f"  PLE plane: {footprint['ple_bytes'] / 1024 / 1024:.2f} MB")
        logger.info(f"  Total: {footprint['total_bytes'] / 1024 / 1024:.2f} MB")
    
    def read_gguf_file(self, input_path: Path) -> TwoPlaneGGUF:
        """Read two-plane GGUF from binary file."""
        two_plane = TwoPlaneGGUF(self.config)
        
        with open(input_path, "rb") as f:
            # Read and verify magic
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF file: magic {magic:#x} != expected {self.GGUF_MAGIC:#x}")
            
            # Read version
            version = struct.unpack("<I", f.read(4))[0]
            
            # Read metadata
            num_metadata = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_metadata):
                key_len = struct.unpack("<I", f.read(4))[0]
                key = f.read(key_len).decode("utf-8")
                value_len = struct.unpack("<I", f.read(4))[0]
                value = f.read(value_len).decode("utf-8")
                two_plane.metadata[key] = value
            
            # Read backbone plane
            num_backbone = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_backbone):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                
                weight_size = struct.unpack("<I", f.read(4))[0]
                weights = np.frombuffer(f.read(weight_size), dtype=np.int8)
                
                scale_size = struct.unpack("<I", f.read(4))[0]
                scales = np.frombuffer(f.read(scale_size), dtype=np.float32)
                
                zp_size = struct.unpack("<I", f.read(4))[0]
                zero_points = np.frombuffer(f.read(zp_size), dtype=np.float32)
                
                qtype_len = struct.unpack("<B", f.read(1))[0]
                quant_type = f.read(qtype_len).decode("utf-8")
                
                two_plane.backbone_plane[name] = {
                    "weights": weights,
                    "scales": scales,
                    "zero_points": zero_points,
                    "quant_type": quant_type,
                    "block_masks": [],
                }
            
            # Read PLE plane
            num_ple = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_ple):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                
                ple_size = struct.unpack("<I", f.read(4))[0]
                ple_embeddings = np.frombuffer(f.read(ple_size), dtype=np.float32)
                
                has_adapters = struct.unpack("<?", f.read(1))[0]
                ple_adapters = None
                if has_adapters:
                    adapter_size = struct.unpack("<I", f.read(4))[0]
                    ple_adapters = np.frombuffer(f.read(adapter_size), dtype=np.float32)
                
                two_plane.ple_plane[name] = {
                    "ple_embeddings": ple_embeddings,
                    "ple_adapters": ple_adapters,
                }
        
        logger.info(f"GGUF file read from {input_path}")
        return two_plane


def encode_two_plane_gguf(
    hollowed_weights: dict,
    ple_adapters: dict,
    ple_embeddings: dict,
    output_path: Path,
    config: Optional[GGUFConfig] = None,
) -> TwoPlaneGGUF:
    """Convenience function to encode a PLE-Coded model to GGUF."""
    if config is None:
        config = GGUFConfig()
    
    encoder = GGUFEncoder(config)
    two_plane = encoder.create_two_plane_gguf(hollowed_weights, ple_adapters, ple_embeddings)
    encoder.write_gguf_file(two_plane, output_path)
    return two_plane


def decode_two_plane_gguf(
    input_path: Path,
    config: Optional[GGUFConfig] = None,
) -> TwoPlaneGGUF:
    """Convenience function to decode a PLE-Coded GGUF file."""
    if config is None:
        config = GGUFConfig()
    
    encoder = GGUFEncoder(config)
    return encoder.read_gguf_file(input_path)
