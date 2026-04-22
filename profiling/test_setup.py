#!/usr/bin/env python3
"""Quick sanity check for Phase 1 setup"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from profiling.analysis.profiler import (
    compute_ple_dominance_score,
    compute_channel_attribution,
    ModelLoader,
)

def test_ple_dominance():
    print("Testing PLE dominance computation...")

    ple = torch.randn(32, 512, 2048)
    backbone = torch.randn(32, 512, 2048) + ple * 0.8

    score = compute_ple_dominance_score(ple, backbone)
    print(f"  PLE dominance score (high PLE): {score:.4f}")

    ple_attr, backbone_attr = compute_channel_attribution(ple, backbone)
    print(f"  Channel attribution shape: {ple_attr.shape}")
    print(f"  Mean PLE attribution: {ple_attr.mean():.4f}")

    ple_low = torch.randn(32, 512, 2048) * 0.1
    backbone_high = torch.randn(32, 512, 2048)
    score_low = compute_ple_dominance_score(ple_low, backbone_high)
    print(f"  PLE dominance score (low PLE): {score_low:.4f}")

    print("  PASS: PLE dominance computation works")
    return True

def test_model_loader():
    print("\nTesting ModelLoader (dry run — no actual loading)...")

    for source in ["lmstudio", "huggingface", "local"]:
        loader = ModelLoader(model_source=source)
        print(f"  {source}: device={loader.device}")

    print("  PASS: ModelLoader initializes correctly")
    return True

def main():
    print("=" * 50)
    print("Phase 1 Setup Sanity Check")
    print("=" * 50)

    tests = [
        ("PLE Dominance Computation", test_ple_dominance),
        ("Model Loader Initialization", test_model_loader),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                print(f"  FAIL: {name}")
                failed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
