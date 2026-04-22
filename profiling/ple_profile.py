#!/usr/bin/env python3
"""
Minimal PLE profiler for Gemma E4B.
Hooks only transformer layers (35 total) and computes residual-based PLE dominance.
"""
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import ModelLoader


def run_minimal_profile():
    print("Minimal PLE Profile — Gemma E4B")
    print("=" * 50)

    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")
    print(f"Model: {type(model).__name__}")

    dummy_data = torch.randint(0, 32000, (2, 16))
    dataloader = DataLoader(TensorDataset(dummy_data), batch_size=1)

    layer_input = {}
    layer_output = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                layer_input[name] = input[0].detach()
            if isinstance(output, tuple) and len(output) > 0:
                layer_output[name] = output[0].detach()
            elif hasattr(output, 'last_hidden_state'):
                layer_output[name] = output.last_hidden_state.detach()
            elif not isinstance(output, tuple):
                layer_output[name] = output.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if "model.language_model.layers." in name:
            parts = name.split(".")
            if len(parts) == 5 and parts[-1].isdigit():
                handles.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered {len(handles)} layer hooks")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(loader.device)
            model(input_ids)
            break

    print(f"Captured inputs: {len(layer_input)}, outputs: {len(layer_output)}")

    for h in handles:
        h.remove()

    results = {}
    for name in sorted(layer_input.keys()):
        inp = layer_input[name]
        out = layer_output.get(name, None)
        if out is None:
            continue

        layer_num = int(name.split(".")[-1])
        residual = out - inp
        residual_var = torch.var(residual).item()
        output_var = torch.var(out).item()
        ple_dom = residual_var / output_var if output_var > 0 else 0.0

        results[layer_num] = {
            "layer_idx": layer_num,
            "ple_dominance": ple_dom,
            "ple_variance": residual_var,
            "output_variance": output_var,
            "is_ple_dominant": ple_dom >= 0.5,
        }

    ple_dominant = [l for l, v in results.items() if v["is_ple_dominant"]]

    output = {
        "layer_results": results,
        "ple_dominant_layers": sorted(ple_dominant),
        "total_layers": len(results),
        "batches_processed": 1,
    }

    output_path = Path("profiling/outputs/ple_dominance_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total layers: {len(results)}")
    print(f"PLE-dominant: {sorted(ple_dominant)}")

    print("\nPer-layer PLE dominance:")
    for ln in sorted(results.keys()):
        r = results[ln]
        bar = "#" * int(r["ple_dominance"] * 20)
        print(f"  Layer {ln:2d}: {r['ple_dominance']:.4f} |{bar:<20}| var={r['ple_variance']:.2f}")


if __name__ == "__main__":
    run_minimal_profile()
