"""
Interactive story generation with the trained CS336 model.

Usage:
    .venv\\Scripts\\python.exe generate_story.py
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch

import train
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def find_checkpoint(out_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    final_path = out_dir / "final.pt"
    if final_path.exists():
        return final_path

    ckpts = sorted(out_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        print(f"final.pt not found; using latest checkpoint: {ckpts[0]}")
        return ckpts[0]

    raise FileNotFoundError(f"No final.pt or ckpt_*.pt found in {out_dir}")


def load_cached_tokenizer(cfg: dict) -> Tokenizer:
    train_path, _ = train.load_data(cfg["data_dir"], cfg["data_mode"])
    payload = {
        "kind": "tokenizer",
        "version": 1,
        "train_file": train._file_signature(train_path),
        "vocab_size": cfg["vocab_size"],
        "special_tokens": cfg["special_tokens"],
        "sample_bytes": cfg["bpe_sample_bytes"],
    }
    tokenizer_stem = f"tokenizer_{train._cache_key(payload)}"
    tokenizer_path = Path(cfg["out_dir"]) / "token_cache" / f"{tokenizer_stem}.pkl"

    if not tokenizer_path.exists():
        candidates = sorted((Path(cfg["out_dir"]) / "token_cache").glob("tokenizer_*.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"Tokenizer cache not found: {tokenizer_path}\n"
                "Run train.py at least once so it can create out/token_cache/."
            )
        tokenizer_path = candidates[-1]
        print(f"Expected tokenizer cache not found; using fallback: {tokenizer_path}")

    with open(tokenizer_path, "rb") as f:
        vocab, merges = pickle.load(f)
    print(f"Loaded tokenizer: {tokenizer_path}")
    return Tokenizer(vocab, merges, cfg["special_tokens"])


def build_model(cfg: dict, device: torch.device) -> TransformerLM:
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["rope_theta"],
        device=device,
    )
    return model.to(device)


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    device: torch.device,
) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    eot_id = tokenizer.special_to_id.get("<|endoftext|>")

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.context_length :]
        logits = model(idx_cond)[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= top_p
                keep[..., 0] = True
                filtered = torch.zeros_like(probs)
                filtered.scatter_(-1, sorted_idx, sorted_probs * keep.float())
                probs = filtered / filtered.sum(dim=-1, keepdim=True)

            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=-1)
        if eot_id is not None and int(next_id.item()) == eot_id:
            break

    return tokenizer.decode(ids[0].tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stories with the trained CS336 model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to final.pt or ckpt_*.pt.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 uses greedy decoding; 0.7-1.0 is typical.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold; set 1.0 to disable.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dict(train.CFG)
    out_dir = Path(cfg["out_dir"])
    checkpoint_path = find_checkpoint(out_dir, args.checkpoint)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    tokenizer = load_cached_tokenizer(cfg)
    model = build_model(cfg, device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Ready on {device}. Type /quit to exit.\n")

    while True:
        prompt = input("Story opening> ").strip()
        if prompt.lower() in {"/q", "/quit", "quit", "exit"}:
            break
        if not prompt:
            continue

        print("\nGenerating...\n")
        text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=None if args.top_p >= 1 else args.top_p,
            device=device,
        )
        print(text)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
