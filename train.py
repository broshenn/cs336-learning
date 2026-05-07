"""
CS336 Assignment 1 training script.

Usage:
    uv run python train.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
)


CFG = {
    # Data
    "data_mode": "tinystories",  # "tinystories" or "owt"
    "data_dir": "data",
    "vocab_size": 8192,
    "special_tokens": ["<|endoftext|>"],
    "bpe_sample_bytes": 50 * 1024 * 1024,
    "valid_max_tokens": 5_000_000,
    # Model
    "context_length": 256,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10000.0,
    # Training
    "batch_size": 16,
    "micro_batch_size": 8,
    "max_iters": 12000,
    "lr_max": 3e-4,
    "lr_min": 3e-5,
    "warmup_iters": 1000,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "cuda_memory_target_gb": 6.0,
    # Logging and checkpointing
    "log_every": 20,
    "eval_every": 1000,
    "save_every": 3000,
    "sample_every": 2000,
    "out_dir": "out"
}


def load_data(data_dir, mode):
    if mode == "tinystories":
        train_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
        valid_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    else:
        train_path = os.path.join(data_dir, "owt_train.txt")
        valid_path = os.path.join(data_dir, "owt_valid.txt")

    for path in (train_path, valid_path):
        if not os.path.exists(path):
            sys.exit(f"Data file does not exist: {path}")

    return train_path, valid_path


def _file_signature(path: str | os.PathLike) -> dict[str, int | str]:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.int32)


def _load_token_cache(cache_dir: Path, stem: str):
    meta_path = cache_dir / f"{stem}.json"
    bin_path = cache_dir / f"{stem}.bin"
    if not meta_path.exists() or not bin_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    expected_bytes = int(meta["count"]) * dtype.itemsize
    if bin_path.stat().st_size != expected_bytes:
        return None
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(int(meta["count"]),))


def _encode_file_to_cache(
    input_path: str | os.PathLike,
    tokenizer: Tokenizer,
    cache_dir: Path,
    stem: str,
    dtype: np.dtype,
    max_tokens: int | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir / f"{stem}.bin"
    meta_path = cache_dir / f"{stem}.json"

    count = 0
    last_log = time.time()
    t0 = last_log
    with open(input_path, "r", encoding="utf-8", errors="replace") as f_in, open(bin_path, "wb") as f_out:
        for line in f_in:
            ids = tokenizer.encode(line)
            if max_tokens is not None:
                remaining = max_tokens - count
                if remaining <= 0:
                    break
                ids = ids[:remaining]
            if ids:
                arr = np.asarray(ids, dtype=dtype)
                arr.tofile(f_out)
                count += int(arr.size)

            now = time.time()
            if now - last_log >= 30:
                print(f"   encoded {count:,} tokens so far ({(now - t0) / 60:.1f} min)")
                last_log = now

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"count": count, "dtype": dtype.name}, f)

    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(count,))


def build_tokenizer(train_path, cfg):
    vocab_size = cfg["vocab_size"]
    special_tokens = cfg["special_tokens"]
    sample_bytes = cfg["bpe_sample_bytes"]
    cache_dir = Path(cfg["out_dir"]) / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_payload = {
        "kind": "tokenizer",
        "version": 1,
        "train_file": _file_signature(train_path),
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "sample_bytes": sample_bytes,
    }
    tokenizer_stem = f"tokenizer_{_cache_key(tokenizer_payload)}"
    tokenizer_path = cache_dir / f"{tokenizer_stem}.pkl"

    print("=" * 50)
    if tokenizer_path.exists():
        t0 = time.time()
        print(f"1. Loading cached BPE tokenizer: {tokenizer_path}")
        with open(tokenizer_path, "rb") as f:
            vocab, merges = pickle.load(f)
        print(f"   loaded in {time.time() - t0:.1f}s")
    else:
        print("1. Training BPE tokenizer")
        print(f"   data: {train_path}")
        print(f"   vocab size: {vocab_size}")
        print(f"   sample: {sample_bytes / 1024 / 1024:.0f} MB")

        t0 = time.time()
        sample_path = cache_dir / f"{tokenizer_stem}.sample.txt"
        with open(train_path, "rb") as f_in:
            sample_text = f_in.read(sample_bytes).decode("utf-8", errors="replace")
        with open(sample_path, "w", encoding="utf-8") as f_out:
            f_out.write(sample_text)
        try:
            vocab, merges = train_bpe(sample_path, vocab_size, special_tokens)
        finally:
            try:
                sample_path.unlink(missing_ok=True)
            except PermissionError:
                pass

        with open(tokenizer_path, "wb") as f:
            pickle.dump((vocab, merges), f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.time() - t0
        print(f"   BPE done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print(f"   actual vocab size: {len(vocab)}, merges: {len(merges)}")

    token_payload = {
        "kind": "encoded_train",
        "version": 1,
        "train_file": _file_signature(train_path),
        "tokenizer": tokenizer_stem,
    }
    token_stem = f"train_ids_{_cache_key(token_payload)}"
    train_ids = _load_token_cache(cache_dir, token_stem)
    if train_ids is not None:
        print(f"2. Loading cached train ids: {len(train_ids):,} tokens")
    else:
        print("2. Encoding train file to cache...")
        t1 = time.time()
        train_ids = _encode_file_to_cache(train_path, tokenizer, cache_dir, token_stem, _token_dtype(vocab_size))
        elapsed = time.time() - t1
        print(f"   encoded {len(train_ids):,} tokens in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print("=" * 50)
    return tokenizer, train_ids, tokenizer_stem


def load_valid_ids(valid_path, tokenizer, cfg, tokenizer_stem):
    cache_dir = Path(cfg["out_dir"]) / "token_cache"
    payload = {
        "kind": "encoded_valid",
        "version": 1,
        "valid_file": _file_signature(valid_path),
        "tokenizer": tokenizer_stem,
        "max_tokens": cfg["valid_max_tokens"],
    }
    stem = f"valid_ids_{_cache_key(payload)}"

    ids = _load_token_cache(cache_dir, stem)
    if ids is not None:
        print(f"Loaded cached valid ids: {len(ids):,} tokens")
        return ids

    print("Encoding valid file to cache...")
    t0 = time.time()
    ids = _encode_file_to_cache(
        valid_path,
        tokenizer,
        cache_dir,
        stem,
        _token_dtype(cfg["vocab_size"]),
        max_tokens=cfg["valid_max_tokens"],
    )
    print(f"  valid ids: {len(ids):,} tokens, {time.time() - t0:.1f}s")
    return ids


def _gradient_accumulation(effective_batch_size: int, micro_batch_size: int) -> int:
    return (effective_batch_size + micro_batch_size - 1) // micro_batch_size


def _cuda_memory_summary(device, micro_batch_size: int, effective_batch_size: int):
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    accum = _gradient_accumulation(effective_batch_size, micro_batch_size)
    print(
        f"CUDA memory: allocated {allocated:.2f}GB, reserved {reserved:.2f}GB / {total:.2f}GB "
        f"| micro_batch={micro_batch_size}, grad_accum={accum}, effective_batch={effective_batch_size}"
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=1.0, top_p=None, device="cpu"):
    from cs336_basics.training import softmax

    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.context_length :]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / temperature
        probs = softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= top_p
            keep[..., 0] = True
            filtered = torch.zeros_like(probs)
            filtered.scatter_(-1, sorted_idx, sorted_probs * keep.float())
            probs = filtered / filtered.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=-1)

    model.train()
    return tokenizer.decode(ids[0].tolist())


def train():
    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(cfg["out_dir"], exist_ok=True)

    train_path, valid_path = load_data(cfg["data_dir"], cfg["data_mode"])
    tokenizer, train_ids, tokenizer_stem = build_tokenizer(train_path, cfg)
    valid_ids = load_valid_ids(valid_path, tokenizer, cfg, tokenizer_stem)

    print("Initializing model...")
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
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr_max"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg["weight_decay"],
    )

    effective_batch_size = cfg["batch_size"]
    micro_batch_size = min(cfg.get("micro_batch_size") or effective_batch_size, effective_batch_size)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _cuda_memory_summary(device, micro_batch_size, effective_batch_size)
    print(f"\nStarting training for {cfg['max_iters']} steps\n")
    t_start = time.time()
    last_log_time = t_start
    last_log_iter = 0

    for it in range(cfg["max_iters"]):
        lr = get_lr_cosine_schedule(
            it,
            cfg["lr_max"],
            cfg["lr_min"],
            cfg["warmup_iters"],
            cfg["max_iters"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        while True:
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad()
                remaining = effective_batch_size
                accum_loss = 0.0
                while remaining > 0:
                    current_batch_size = min(micro_batch_size, remaining)
                    x, y = get_batch(train_ids, current_batch_size, cfg["context_length"], device)
                    logits = model(x)
                    loss = cross_entropy(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
                    loss_weight = current_batch_size / effective_batch_size
                    (loss * loss_weight).backward()
                    accum_loss += loss.item() * loss_weight
                    remaining -= current_batch_size

                gradient_clipping(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                loss_value = accum_loss
                if device.type == "cuda" and cfg.get("cuda_memory_target_gb") is not None:
                    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    target_gb = float(cfg["cuda_memory_target_gb"])
                    if peak_gb > target_gb and micro_batch_size > 1:
                        micro_batch_size = max(1, micro_batch_size // 2)
                        torch.cuda.empty_cache()
                        accum = _gradient_accumulation(effective_batch_size, micro_batch_size)
                        print(
                            f"  [memory] peak {peak_gb:.2f}GB > target {target_gb:.2f}GB; "
                            f"micro_batch_size -> {micro_batch_size}, grad_accum={accum}"
                        )
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if micro_batch_size <= 1:
                    raise
                micro_batch_size = max(1, micro_batch_size // 2)
                accum = _gradient_accumulation(effective_batch_size, micro_batch_size)
                print(
                    f"  [OOM] reducing micro_batch_size to {micro_batch_size}; "
                    f"grad_accum={accum}, effective_batch={effective_batch_size}"
                )

        if it % cfg["log_every"] == 0:
            now = time.time()
            elapsed = now - t_start
            steps_done = it + 1
            total_steps = cfg["max_iters"]
            overall_steps_per_sec = steps_done / max(elapsed, 1e-9)
            recent_steps = max(1, steps_done - last_log_iter)
            recent_steps_per_sec = recent_steps / max(now - last_log_time, 1e-9)
            eta_seconds = (total_steps - steps_done) / max(overall_steps_per_sec, 1e-9)
            tokens_per_sec = recent_steps_per_sec * effective_batch_size * cfg["context_length"]
            progress = 100 * steps_done / total_steps
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                mem = f" | mem {allocated:.2f}/{reserved:.2f}GB"
            else:
                mem = ""
            accum = _gradient_accumulation(effective_batch_size, micro_batch_size)
            print(
                f"  it {it:5d}/{total_steps} ({progress:5.1f}%)"
                f" | loss {loss_value:.4f}"
                f" | lr {lr:.2e}"
                f" | {recent_steps_per_sec:.2f} step/s"
                f" | {tokens_per_sec:,.0f} tok/s"
                f" | elapsed {_format_duration(elapsed)}"
                f" | eta {_format_duration(eta_seconds)}"
                f" | micro {micro_batch_size}x{accum}"
                f"{mem}"
            )
            last_log_time = now
            last_log_iter = steps_done

        if it % cfg["eval_every"] == 0 and it > 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(valid_ids, micro_batch_size, cfg["context_length"], device)
                v_logits = model(xv)
                v_loss = cross_entropy(v_logits.reshape(-1, cfg["vocab_size"]), yv.reshape(-1))
            model.train()
            print(f"  >>> it {it:5d} | valid_loss {v_loss.item():.4f}")

        if it % cfg["sample_every"] == 0 and it > 0:
            sample = generate(
                model,
                tokenizer,
                prompt="Once upon a time",
                max_new_tokens=128,
                temperature=0.8,
                device=device,
            )
            print(f"  --- sample (it={it}) ---")
            print(sample[:300])
            print("  ---")

        if it % cfg["save_every"] == 0 and it > 0:
            ckpt_path = os.path.join(cfg["out_dir"], f"ckpt_{it:05d}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"  [saved] {ckpt_path}")

    final_path = os.path.join(cfg["out_dir"], "final.pt")
    save_checkpoint(model, optimizer, cfg["max_iters"], final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total training time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    train()
