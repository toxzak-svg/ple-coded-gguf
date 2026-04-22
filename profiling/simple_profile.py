#!/usr/bin/env python3
"""Simplified PLE profiling that hooks into transformer layers."""
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import ModelLoader, save_profiling_results


def simple_profile():
    print("Simple PLE Profile")
    print("=" * 50)

    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")

    dummy_data = torch.randint(0, 32000, (4, 32))
    dataloader = DataLoader(TensorDataset(dummy_data), batch_size=2)

    hidden_states = {}
    residuals = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states[name] = input[0].detach()
            if isinstance(output, tuple) and len(output) > 0:
                residuals[name] = output[0].detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if "gemma4" in name.lower() and "layer" in name.lower() and any(c.isdigit() for c in name):
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)

    print(f"Registered {len(handles)} hooks")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(loader.device)
            _ = model(input_ids)
            break

    print(f"Captured {len(hidden_states)} hidden states")
    print(f"Captured {len(residuals)} residuals")

    for h in handles:
        h.remove()

    layer_results = {}
    for name in sorted(hidden_states.keys()):
        inp = hidden_states[name]
        out = residuals.get(name, None)
        if out is None:
            continue

        layer_num = int(name.split(".")[-1]) if name.split(".")[-1].isdigit() else 0

        residual = out - inp
        if inp.numel() == 0:
            continue
        residual_var = torch.var(residual).item()
        output_var = torch.var(out).item()

        if output_var > 0:
            ple_dominance = residual_var / output_var
        else:
            ple_dominance = 0.0

        layer_results[layer_num] = {
            "layer_idx": layer_num,
            "ple_dominance": ple_dominance,
            "ple_variance": residual_var,
            "output_variance": output_var,
            "is_ple_dominant": ple_dominance >= 0.5,
        }

    ple_dominant_layers = [l for l, v in layer_results.items() if v["is_ple_dominant"]]

    results = {
        "layer_results": layer_results,
        "ple_dominant_layers": sorted(ple_dominant_layers),
        "total_layers": len(layer_results),
        "batches_processed": len(dataloader),
    }

    output_path = Path("profiling/outputs/quick_profile_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_profiling_results(results, output_path)

    print(f"\nResults saved to {output_path}")
    print(f"Total layers: {len(layer_results)}")
    print(f"PLE-dominant layers: {sorted(ple_dominant_layers)}")

    for layer_num in sorted(layer_results.keys()):
        r = layer_results[layer_num]
        print(f"  Layer {layer_num}: PLE={r['ple_dominance']:.4f}")


if __name__ == "__main__":
    simple_profile()
