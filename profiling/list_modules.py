#!/usr/bin/env python3
"""List all layer modules in Gemma E4B to understand the naming."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import ModelLoader


def main():
    loader = ModelLoader("huggingface")
    model, _ = loader.load_gemma_e2b(model_name="google/gemma-4-E2B-it")

    layer_modules = []
    for name, module in model.named_modules():
        if "layer" in name.lower():
            layer_modules.append((name, type(module).__name__))

    print(f"Total modules with 'layer' in name: {len(layer_modules)}")
    print("\nFirst 20 layer modules:")
    for name, mod_type in layer_modules[:20]:
        print(f"  {name}: {mod_type}")

    print("\nLooking for decoder layers specifically:")
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 3 and parts[-1].isdigit():
            print(f"  {name}")


if __name__ == "__main__":
    main()
