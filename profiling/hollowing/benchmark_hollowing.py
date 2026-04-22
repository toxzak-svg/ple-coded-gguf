#!/usr/bin/env python3
"""
Phase 2 Hollowing Benchmark
Compares hollowed model quality vs naive quantization on calibration data.
"""
import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import ModelLoader, run_layer_profiling
from profiling.analysis.config import PLEDominanceConfig
from profiling.hollowing.hollowing import (
    HollowingEngine,
    HollowingConfig,
    run_hollowing,
)


def benchmark_hollowing():
    print("=" * 60)
    print("Phase 2: Hollowing Benchmark")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")
    print(f"      Loaded: {type(model).__name__}")

    print("\n[2/5] Loading PLE dominance results...")
    ple_results_path = Path("profiling/outputs/ple_dominance_results.json")
    if not ple_results_path.exists():
        ple_results_path = Path("profiling/outputs/quick_profile_results.json")
    
    if ple_results_path.exists():
        with open(ple_results_path) as f:
            ple_data = json.load(f)
        ple_scores = {
            k: v["ple_dominance"] 
            for k, v in ple_data.get("layer_results", {}).items()
        }
        print(f"      Loaded {len(ple_scores)} layer scores")
    else:
        print("      No PLE results found, computing...")
        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = torch.randint(0, 32000, (8, 64))
        dataloader = DataLoader(TensorDataset(dummy_data), batch_size=2)
        results = run_layer_profiling(model, dataloader, device=loader.device)
        ple_scores = {
            k: v["ple_dominance"]
            for k, v in results["layer_results"].items()
        }

    print("\n[3/5] Running hollowing pipeline...")
    config = HollowingConfig(
        prune_block_size=64,
        prune_threshold=0.5,
        ple_dominant_quant="Q2",
        backbone_quant="Q4",
        target_compression=0.4,
    )
    engine = HollowingEngine(config)
    hollowed = engine.hollow_model(model, ple_scores)
    print(f"      Hollowed {len(hollowed)} weight matrices")

    print("\n[4/5] Computing compression statistics...")
    total_orig = 0
    total_hollowed = 0
    ple_subsidized_count = 0

    for name, hw in hollowed.items():
        orig_bits = hw.original_shape[0] * hw.original_shape[1] * 16
        if hw.ple_subsidized:
            quant_bits = hw.quantized_data.numel() * 2 + hw.scale.numel() * 16
            ple_subsidized_count += 1
        else:
            quant_bits = hw.quantized_data.numel() * 4 + hw.scale.numel() * 16
        total_orig += orig_bits
        total_hollowed += quant_bits

    overall_ratio = total_hollowed / total_orig if total_orig > 0 else 1.0
    print(f"      PLE-subsidized matrices: {ple_subsidized_count}")
    print(f"      Overall compression: {overall_ratio:.2%}")
    print(f"      Size reduction: {(1 - overall_ratio):.2%}")

    print("\n[5/5] Saving results...")
    output_path = Path("profiling/outputs/hollowing_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_hollowed_weights(output_path)
    print(f"      Saved to {output_path}")

    print("\n" + "=" * 60)
    print("Hollowing Benchmark Complete")
    print("=" * 60)
    print(f"  Total matrices: {len(hollowed)}")
    print(f"  PLE-subsidized: {ple_subsidized_count}")
    print(f"  Overall compression: {overall_ratio:.2%}")
    
    return hollowed


if __name__ == "__main__":
    benchmark_hollowing()
