#!/usr/bin/env python3
"""Quick profiling run with minimal samples to verify concept."""
import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import (
    ModelLoader,
    run_layer_profiling,
    save_profiling_results,
)
from torch.utils.data import DataLoader, TensorDataset


def quick_profile():
    print("Quick Profile — Gemma E4B PLE Dominance")
    print("=" * 50)

    print("[1/4] Loading model...")
    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")
    print(f"      Loaded: {type(model).__name__}")

    print("[2/4] Creating minimal dataset (8 samples, seq=64)...")
    dummy_data = torch.randint(0, 32000, (8, 64))
    ds = TensorDataset(dummy_data)
    dataloader = DataLoader(ds, batch_size=2)

    print("[3/4] Running layer profiling...")
    results = run_layer_profiling(model, dataloader, device=loader.device)

    print("[4/4] Saving results...")
    output_path = Path("profiling/outputs/quick_profile_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_profiling_results(results, output_path)

    print()
    print("=" * 50)
    print("Quick Profile Complete")
    print("=" * 50)
    print(f"PLE-dominant layers: {results['ple_dominant_layers']}")
    print(f"Total layers: {results['total_layers']}")
    print(f"Results: {output_path}")

    for layer_num in sorted(results["layer_results"].keys()):
        r = results["layer_results"][layer_num]
        print(f"  Layer {layer_num}: PLE dominance={r['ple_dominance']:.4f} (var={r['ple_variance']:.4f})")


if __name__ == "__main__":
    quick_profile()
