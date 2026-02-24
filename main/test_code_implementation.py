"""
Quick test to verify the implementation works correctly
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from config import Code2NLConfig, Code2CodeConfig, NL2CodeConfig
        from models import create_salora_model, save_lora_config_for_peft
        from search import SALoraArchitecture, SpectralIntrusionMetric
        from data_loaders import get_data_loader
        from metrics import compute_code2nl_metrics
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation with SALora"""
    print("\nTesting model creation...")
    try:
        from models import inject_salora_to_qwen
        from transformers import AutoModelForCausalLM

        # Use a small model for testing
        model_name = "Qwen/Qwen2.5-Coder-0.5B"
        print(f"  Loading model: {model_name} (this may take a while...)")

        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"  Model loaded, injecting SALora...")

        model, lora_modules = inject_salora_to_qwen(model, r=4, lora_alpha=8)

        print(f"  ✓ Model created with {len(lora_modules)} LoRA modules")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        print("  Note: This test requires downloading Qwen model, which may fail if not available")
        return False


def test_architecture():
    """Test architecture module"""
    print("\nTesting architecture module...")
    try:
        from search import SALoraArchitecture

        arch = SALoraArchitecture(n_layers=4, r_max=8)
        alphas = arch()

        assert alphas.shape == (4, 7, 8), f"Expected shape (4, 7, 8), got {alphas.shape}"

        alpha_dict = arch.get_alpha_dict()
        assert len(alpha_dict) == 4 * 7, f"Expected 28 entries, got {len(alpha_dict)}"

        ranks = arch.get_rank_summary()
        assert len(ranks) == 7, f"Expected 7 module types, got {len(ranks)}"

        print(f"  ✓ Architecture module working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Architecture test failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    try:
        from metrics import compute_code2nl_metrics

        predictions = [
            "this is a test function",
            "returns the sum of two numbers",
        ]
        references = [
            "this is a test method",
            "returns sum of two integers",
        ]

        metrics = compute_code2nl_metrics(predictions, references)

        assert 'bleu' in metrics, "BLEU not computed"
        assert 'meteor' in metrics, "METEOR not computed"
        assert 'rouge-l-f' in metrics, "ROUGE-L not computed"

        print(f"  ✓ Metrics computed successfully")
        print(f"    BLEU: {metrics['bleu']:.4f}")
        print(f"    METEOR: {metrics['meteor']:.4f}")
        print(f"    ROUGE-L: {metrics['rouge-l-f']:.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        print(f"  Note: This requires nltk and rouge-score packages")
        return False


def test_config():
    """Test configuration classes"""
    print("\nTesting configuration classes...")
    try:
        from config import Code2NLConfig, Code2CodeConfig, NL2CodeConfig

        config = Code2NLConfig()
        assert config.model_name == "Qwen/Qwen2.5-Coder-1.5B"
        assert config.lora_r == 8
        assert config.task_type == "causal_lm"

        print(f"  ✓ Configuration classes working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def main():
    print("=" * 80)
    print("SALora Code Tasks Implementation Tests")
    print("=" * 80 + "\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Architecture", test_architecture()))
    results.append(("Metrics", test_metrics()))
    results.append(("Model Creation", test_model_creation()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s} {status}")

    print("\n" + "=" * 80)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. Please check the errors above.")

    print("=" * 80)


if __name__ == "__main__":
    main()
