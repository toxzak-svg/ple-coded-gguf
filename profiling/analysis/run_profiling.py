#!/usr/bin/env python3
"""
Phase 1 Profiling Entry Point
Run: python -m profiling.analysis.run_profiling
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.analysis.profiler import (
    ModelLoader,
    ActivationCollector,
    compute_ple_dominance_score,
    compute_channel_attribution,
    run_profiling,
    save_profiling_results,
)
from profiling.analysis.config import PLEDominanceConfig


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Profile Gemma E4B PLE dominance")
    parser.add_argument("--source", choices=["lmstudio", "huggingface", "local"], default="huggingface",
                       help="Model source")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Local path or LM Studio model ID")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E2B-it",
                       help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Calibration dataset")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                       help="Dataset config")
    parser.add_argument("--num-samples", type=int, default=256,
                       help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for results")
    parser.add_argument("--variance-threshold", type=float, default=0.5,
                       help="PLE dominance variance threshold")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test setup without running profiling")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path("profiling/outputs/profiling_results.json")

    print("=" * 60)
    print("PLE-Coded GGUF — Phase 1: Profiling")
    print("=" * 60)
    print(f"Model source: {args.source}")
    print(f"Dataset: {args.dataset}/{args.dataset_config}")
    print(f"Samples: {args.num_samples}, Seq len: {args.seq_len}")
    print(f"Variance threshold: {args.variance_threshold}")
    print()

    config = PLEDominanceConfig(
        variance_threshold=args.variance_threshold,
        calibration_seq_len=args.seq_len,
    )

    print("[1/5] Loading model...")
    loader = ModelLoader(model_source=args.source)
    model, tokenizer = loader.load_gemma_e2b(
        model_path=args.model_path,
        model_name=args.model_name,
    )
    print(f"      Model loaded: {type(model).__name__}")

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    print("[2/5] Loading calibration dataset...")
    try:
        from profiling.analysis.profiler import get_calibration_dataset
        dataloader = get_calibration_dataset(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
        )
        print(f"      Dataset loaded: {len(dataloader)} batches")
    except Exception as e:
        print(f"      ERROR loading dataset: {e}")
        print("      Creating dummy dataloader for testing...")
        dummy_data = torch.randint(0, 32000, (args.num_samples, args.seq_len))
        from torch.utils.data import DataLoader, TensorDataset
        dataloader = DataLoader(TensorDataset(dummy_data), batch_size=4)

    print("[3/5] Registering hooks for activation collection...")
    collector = ActivationCollector(model)
    print(f"      Hooks registered: {len(collector.hooks)}")

    print("[4/5] Running profiling pass...")
    results = run_profiling(
        model=model,
        dataloader=dataloader,
        device=loader.device,
    )

    print("[5/5] Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_profiling_results(results, output_path)

    print()
    print("=" * 60)
    print("Profiling Complete")
    print("=" * 60)
    print(f"PLE-dominant layers: {results['ple_dominant_layers']}")
    print(f"Total layers analyzed: {results['total_layers']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
