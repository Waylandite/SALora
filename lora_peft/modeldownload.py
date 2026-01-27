import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {model_name} to {output_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(output_dir)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model locally")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B", help="Model id on Hugging Face Hub")
    parser.add_argument("--output", type=str, default="./qwen2.5-coder-3b", help="Local output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_model(args.model, args.output)
