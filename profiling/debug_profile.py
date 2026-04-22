#!/usr/bin/env python3
"""Debug script to verify PLE dominance computation."""
import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import ModelLoader, LayerActivationCollector, analyze_layer_ple_dominance
from torch.utils.data import DataLoader, TensorDataset


def main():
    print("=" * 60)
    print("PLE Dominance Debug")
    print("=" * 60)

    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")
    print(f"Loaded: {type(model).__name__}")

    dummy_data = torch.randint(0, 32000, (4, 32))
    dataloader = DataLoader(TensorDataset(dummy_data), batch_size=2)

    collector = LayerActivationCollector(model)
    print(f"Hooks registered: {len(collector.hooks)}")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(loader.device)
            _ = model(input_ids)
            break

    print(f"\nLayer inputs collected: {len(collector.layer_inputs)}")
    print(f"Layer outputs collected: {len(collector.layer_outputs)}")

    if collector.layer_inputs:
        first_name = list(collector.layer_inputs.keys())[0]
        print(f"\nFirst input name: {first_name}")
        print(f"First input shape: {collector.layer_inputs[first_name].shape}")

    layer_summaries = {}
    for name, layer_input in collector.layer_inputs.items():
        if "_out" in name:
            continue
        layer_output = collector.layer_outputs.get(f"{name}_out", None)
        if layer_output is None:
            continue

        parts = name.split(".")
        layer_num = int(parts[-1])

        analysis = analyze_layer_ple_dominance(
            layer_input,
            layer_output,
            layer_num,
            variance_threshold=0.5,
        )

        layer_summaries[layer_num] = analysis

    print("\nLayer PLE Dominance Summary:")
    print("-" * 40)
    for layer_num in sorted(layer_summaries.keys()):
        a = layer_summaries[layer_num]
        status = "PLE-DOM" if a["is_ple_dominant"] else "backbone"
        print(f"Layer {layer_num:2d}: PLE={a['ple_dominance']:.4f} ({status})")

    ple_dominant = [l for l, a in layer_summaries.items() if a["is_ple_dominant"]]
    print(f"\nPLE-dominant layers: {ple_dominant}")

    collector.remove_hooks()


if __name__ == "__main__":
    main()
