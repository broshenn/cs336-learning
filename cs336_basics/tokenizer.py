# cs336_basics/tokenizer.py
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import json
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


def _special_pattern(special_tokens: list[str]) -> str | None:
    if not special_tokens:
        return None
    return "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    pattern = _special_pattern(special_tokens)
    pieces = re.split(pattern, text) if pattern else [text]
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            bs = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in bs)] += 1
    return counts


def _count_pairs(word_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, count in word_counts.items():
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += count
    return dict(pair_counts)


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    out = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        tok_b = tok.encode("utf-8")
        if tok_b not in vocab.values():
            vocab[len(vocab)] = tok_b

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    word_counts = dict(_pretoken_counts(text, special_tokens))

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(word_counts)
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        new_counts: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for word, count in word_counts.items():
            new_counts[_merge_word(word, best)] += count
        word_counts = dict(new_counts)
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        existing = {v: k for k, v in self.vocab.items()}
        for tok in self.special_tokens:
            tok_b = tok.encode("utf-8")
            if tok_b not in existing:
                self.vocab[len(self.vocab)] = tok_b
                existing[tok_b] = len(self.vocab) - 1

        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.special_to_id = {tok: self.token_to_id[tok.encode("utf-8")] for tok in self.special_tokens}
        self._special_split_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}
        self._encode_cache_max_size = 200_000

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): bytes(v) if isinstance(v, list) else v.encode("latin1") for k, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, b = line.split("\t") if "\t" in line else line.split(" ")
                merges.append((a.encode("latin1"), b.encode("latin1")))
        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, bs: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in bs]
        if len(tokens) < 2:
            return tokens
        while True:
            best_i = None
            best_rank = None
            for i, pair in enumerate(zip(tokens, tokens[1:])):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_i = i
                    best_rank = rank
            if best_i is None:
                break
            i = best_i
            tokens = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2 :]
        return tokens

    def _encode_pretoken(self, bs: bytes) -> tuple[int, ...]:
        cached = self._encode_cache.get(bs)
        if cached is not None:
            return cached

        ids = tuple(self.token_to_id[tok] for tok in self._apply_bpe(bs))
        if len(self._encode_cache) >= self._encode_cache_max_size:
            self._encode_cache.clear()
        self._encode_cache[bs] = ids
        return ids

    def _encode_regular(self, text: str) -> list[int]:
        ids = []
        for match in _PAT_RE.finditer(text):
            ids.extend(self._encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_regular(text)
        ids = []
        for part in self._special_split_re.split(text):
            if part == "":
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(self._encode_regular(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
