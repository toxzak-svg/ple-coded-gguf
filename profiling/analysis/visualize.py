#!/usr/bin/env python3
"""
Visualize PLE dominance scores from profiling results
Run: python -m profiling.analysis.visualize
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_ple_dominance_bar(results: dict, output_path: Optional[Path] = None):
    layer_results = results.get("layer_results", {})
    layers = []
    scores = []

    for k, v in sorted(layer_results.items(), key=lambda x: x[1].get("layer_idx", 0)):
        layers.append(f"Layer {v['layer_idx']}")
        scores.append(v["ple_dominance"])

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#2ecc71" if s >= 0.5 else "#e74c3c" for s in scores]
        ax.bar(layers, scores, color=colors)
        ax.axhline(y=0.5, color="orange", linestyle="--", label="Threshold (0.5)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("PLE Dominance Score")
        ax.set_title("PLE Dominance by Layer (Gemma E4B)")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Chart saved to {output_path}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available — printing text summary")
        print_table(results)


def plot_channel_heatmap(layer_data: dict, output_path: Optional[Path] = None):
    ple_attrs = layer_data.get("channel_ple_attribution", [])

    if not ple_attrs or len(ple_attrs) == 0:
        print("No channel attribution data available")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.array(ple_attrs)
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn")
        ax.set_xlabel("Channel")
        ax.set_ylabel("PLE Attribution")
        ax.set_title(f"Channel-level PLE Attribution — Layer {layer_data.get('layer_idx', '?')}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available")


def print_table(results: dict):
    layer_results = results.get("layer_results", {})

    print(f"\n{'Layer':<10} {'PLE Dominance':<18} {'Status':<15}")
    print("-" * 45)

    for k, v in sorted(layer_results.items(), key=lambda x: x[1].get("layer_idx", 0)):
        layer_idx = v.get("layer_idx", 0)
        score = v.get("ple_dominance", 0)
        status = "PLE-DOMINANT" if v.get("is_ple_dominant", False) else "backbone"
        print(f"Layer {layer_idx:<4} {score:>12.4f}     {status:<15}")

    print()
    print(f"PLE-dominant layers: {results.get('ple_dominant_layers', [])}")
    print(f"Total layers: {results.get('total_layers', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PLE dominance results")
    parser.add_argument("results", nargs="?", type=str, default=None,
                       help="Path to profiling_results.json")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output path for chart")
    parser.add_argument("--layer", "-l", type=int, default=None,
                       help="Specific layer to visualize")

    args = parser.parse_args()

    if not args.results:
        default_path = Path("profiling/outputs/profiling_results.json")
        if default_path.exists():
            args.results = str(default_path)
        else:
            print("No results file specified and none found at default path")
            return

    results = load_results(Path(args.results))
    print(f"Loaded results from {args.results}")
    print(f"PLE-dominant layers: {results.get('ple_dominant_layers', [])}")

    if args.layer is not None:
        layer_data = None
        for k, v in results["layer_results"].items():
            if v.get("layer_idx") == args.layer:
                layer_data = v
                break
        if layer_data:
            output = Path(args.output) if args.output else None
            plot_channel_heatmap(layer_data, output)
        else:
            print(f"Layer {args.layer} not found in results")
    else:
        output = Path(args.output) if args.output else None
        plot_ple_dominance_bar(results, output)


if __name__ == "__main__":
    main()
