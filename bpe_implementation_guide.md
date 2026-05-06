# CS336 Assignment 1 BPE 实现专题文档

本文只讲 Assignment 1 的 BPE 部分：`run_train_bpe`、`get_tokenizer`、`Tokenizer.encode`、`Tokenizer.decode`、`Tokenizer.encode_iterable`，以及相关测试如何一步一步通过。

本地当前状态备注：`tests/adapters.py` 的 BPE 相关说明已经改成中文，并且当前 `cs336_basics/tokenizer.py` 已经有一个可工作的朴素实现。用 `.venv` 直接跑测试时，`tests/test_tokenizer.py` 全部通过，`tests/test_train_bpe.py` 的正确性和 special token 测试通过，但 `test_train_bpe_speed` 失败，当前耗时约 2.2s，测试要求小于 1.5s。所以如果你的目标是全绿，最后还需要优化 BPE training 的合并阶段。

## 1. BPE 测试架构

### 问题背景

CS336 的测试不会直接假设你的内部文件结构。测试只认识 `tests/adapters.py` 里的两个 BPE 入口：

```python
get_tokenizer(vocab, merges, special_tokens=None)
run_train_bpe(input_path, vocab_size, special_tokens, **kwargs)
```

也就是说，测试文件会准备好输入，然后调用 adapter；adapter 再调用你在 `cs336_basics/` 里写的真实实现。

调用链是：

```text
tests/test_train_bpe.py
  -> tests/adapters.py::run_train_bpe
    -> cs336_basics/tokenizer.py::train_bpe

tests/test_tokenizer.py
  -> tests/adapters.py::get_tokenizer
    -> cs336_basics/tokenizer.py::Tokenizer
      -> tokenizer.encode / decode / encode_iterable
```

### 学生要做的事情

你不需要改测试文件。你只需要：

1. 在 `cs336_basics/tokenizer.py` 写 `train_bpe` 和 `Tokenizer`。
2. 在 `tests/adapters.py` 里把两个 BPE adapter 接到你的实现。
3. 分别跑 `test_train_bpe.py` 和 `test_tokenizer.py`。

### 相关答案

如果 adapter 还没有填写，应填成这样：

```python
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.tokenizer import train_bpe
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
```

当前仓库中这两个函数已经接好了，而且 docstring 已经中文化。注意：adapter 里不应该写 substantive logic，也就是不要把 BPE 训练或编码逻辑写在这里；它只是胶水层。

## 2. 相关测试文件讲解

### 问题背景

BPE 部分主要有四类文件：

```text
tests/adapters.py                 # 测试入口胶水
tests/test_train_bpe.py            # BPE 训练测试
tests/test_tokenizer.py            # Tokenizer 编码/解码测试
tests/common.py                    # GPT-2 bytes-to-unicode 辅助函数
tests/fixtures/*                   # 小语料、GPT-2 词表、参考 merges/vocab
```

### 学生要做的事情

阅读测试时优先看断言，不必被 fixture 转换代码吓到。你要搞清楚每个测试到底检查什么：

```bash
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -q
```

如果 `uv` 可用，也可以用：

```bash
uv run pytest tests/test_train_bpe.py -q
uv run pytest tests/test_tokenizer.py -q
```

### 相关答案

`tests/test_train_bpe.py` 有三个测试：

1. `test_train_bpe_speed`

检查 `run_train_bpe(tests/fixtures/corpus.en, vocab_size=500, special_tokens=["<|endoftext|>"])` 是否在 1.5 秒内完成。这个测试不是检查算法思想，而是逼你别写太慢的 toy implementation。

2. `test_train_bpe`

同样在 `corpus.en` 上训练 500 vocab，然后检查：

```python
assert merges == reference_merges
assert set(vocab.keys()) == set(reference_vocab.keys())
assert set(vocab.values()) == set(reference_vocab.values())
```

这说明 merges 的顺序必须完全一致；vocab 的 key/value 集合必须一致，但不要求 dict 插入顺序一致。

3. `test_train_bpe_special_tokens`

在 `tinystories_sample_5M.txt` 上训练 1000 vocab，检查普通 vocab 里不能出现包含 `b"<|"` 的 token。这是在确认 special token 是硬边界，没有参与普通合并。

`tests/test_tokenizer.py` 重点更多：

1. `get_tokenizer_from_vocab_merges_path`

这是测试内部辅助函数。它读取 GPT-2 的 `gpt2_vocab.json` 和 `gpt2_merges.txt`，用 `common.py::gpt2_bytes_to_unicode()` 转回原始 bytes，然后调用你的 `get_tokenizer`。

2. roundtrip 测试

例如：

```python
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
```

这要求你的 encode 和 decode 至少互逆。

3. matches tiktoken 测试

例如：

```python
reference_ids = tiktoken.get_encoding("gpt2").encode(text)
ids = tokenizer.encode(text)
assert ids == reference_ids
```

这要求你的 regex 预分词和 BPE merge 优先级与 GPT-2 一致。

4. special token 测试

`<|endoftext|>` 必须作为单个 token。重叠 special token 时，例如同时存在 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，必须优先匹配更长的那个。

5. `encode_iterable` 测试

它会对文件 handle 逐段迭代，要求 lazy yield token id。Linux 下还有内存限制测试；Windows 上通常 skip。

## 3. 文件和函数规划

### 问题背景

BPE 部分可以全部放在一个文件：

```text
cs336_basics/tokenizer.py
```

建议先写清楚小函数，再组合成大函数。不要一开始就写 multiprocessing 或复杂优化。

### 学生要做的事情

推荐函数顺序：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _special_pattern(special_tokens): ...
def _pretoken_counts(text, special_tokens): ...
def _count_pairs(word_counts): ...
def _merge_word(word, pair): ...
def train_bpe(input_path, vocab_size, special_tokens, **kwargs): ...

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None): ...
    def _apply_bpe(self, bs): ...
    def _encode_regular(self, text): ...
    def encode(self, text): ...
    def encode_iterable(self, iterable): ...
    def decode(self, ids): ...
```

### 相关答案

依赖关系如下：

```text
train_bpe
  -> _special_pattern
  -> _pretoken_counts
  -> _count_pairs
  -> _merge_word

Tokenizer.encode
  -> special token split
  -> _encode_regular
  -> _apply_bpe
  -> token_to_id lookup

Tokenizer.decode
  -> vocab id lookup
  -> bytes join
  -> utf-8 decode(errors="replace")
```

## 4. 第一步：初始化词表

### 问题背景

byte-level BPE 的初始词表是所有 256 个单字节 token。`vocab` 类型必须是 `dict[int, bytes]`。

### 学生要做的事情

先实现：

```python
vocab = {i: bytes([i]) for i in range(256)}
```

然后把 special tokens 加进去：

```python
for token in special_tokens:
    token_bytes = token.encode("utf-8")
    if token_bytes not in vocab.values():
        vocab[len(vocab)] = token_bytes
```

### 相关答案

注意 special token 占用词表大小，所以如果 `vocab_size=500` 且有 1 个 special token，那么 BPE merge 最多做：

```text
500 - 256 - 1 = 243 次
```

常见错误：

```python
# 错误：value 是 int，不是 bytes
vocab = {i: i for i in range(256)}

# 错误：特殊 token 不 encode
vocab[len(vocab)] = "<|endoftext|>"
```

正确 value 必须全是 bytes。

## 5. 第二步：特殊 token 分割与预分词

### 问题背景

训练 BPE 时，special token 是硬边界。比如：

```text
Doc1<|endoftext|>Doc2
```

不能允许 `Doc1` 末尾和 `Doc2` 开头跨 special token 合并；special token 本身也不参与普通 pair 统计。

### 学生要做的事情

实现 special token regex：

```python
import regex as re

def _special_pattern(special_tokens):
    if not special_tokens:
        return None
    return "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
```

实现预分词计数：

```python
from collections import Counter

def _pretoken_counts(text: str, special_tokens: list[str]):
    counts = Counter()
    pattern = _special_pattern(special_tokens)
    pieces = re.split(pattern, text) if pattern else [text]
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            bs = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in bs)] += 1
    return counts
```

### 相关答案

这里的返回类型建议是：

```python
Counter[tuple[bytes, ...]]
```

例如 `"low low"` 预分词后，可能统计成：

```python
{
    (b"l", b"o", b"w"): 1,
    (b" ", b"l", b"o", b"w"): 1,
}
```

使用 GPT-2 regex 的原因是测试会对齐 `tiktoken`。必须用第三方 `regex` 包，因为 Python 内置 `re` 不支持 `\p{L}`、`\p{N}`。

## 6. 第三步：pair 统计和 tie-break

### 问题背景

BPE 每一轮都找最高频相邻 pair。如果频率相同，作业要求选字典序更大的 pair。

### 学生要做的事情

实现 pair 统计：

```python
from collections import defaultdict

def _count_pairs(word_counts):
    pair_counts = defaultdict(int)
    for word, count in word_counts.items():
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += count
    return dict(pair_counts)
```

选择 best pair：

```python
best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
```

### 相关答案

不要写成：

```python
best = max(pair_counts, key=pair_counts.get)
```

这样只按频率选，频率相同的 tie-break 不受控，`test_train_bpe` 的 `merges == reference_merges` 会失败。

## 7. 第四步：merge 一个 word

### 问题背景

如果 best pair 是 `(b"a", b"b")`，那么 word：

```python
(b"a", b"b", b"b")
```

应该变成：

```python
(b"ab", b"b")
```

合并时不能重叠合并。

### 学生要做的事情

实现：

```python
def _merge_word(word, pair):
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
```

### 相关答案

对于 `(b"a", b"a", b"a")` 合并 `(b"a", b"a")`，结果应是：

```python
(b"aa", b"a")
```

不是 `(b"aa", b"aa")`。

## 8. 第五步：朴素 train_bpe

### 问题背景

朴素版本每轮都重新统计所有 pair，然后对所有 pre-token 应用 merge。它容易写对，但速度可能不够。

### 学生要做的事情

先实现这个版本，目标是通过正确性测试：

```python
def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        tok_b = tok.encode("utf-8")
        if tok_b not in vocab.values():
            vocab[len(vocab)] = tok_b

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_counts = dict(_pretoken_counts(text, special_tokens))
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(word_counts)
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        new_counts = defaultdict(int)
        for word, count in word_counts.items():
            new_counts[_merge_word(word, best)] += count
        word_counts = dict(new_counts)

    return vocab, merges
```

### 相关答案

这个版本应该可以通过：

```bash
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py::test_train_bpe -q
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -q
```

但可能无法通过：

```bash
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py::test_train_bpe_speed -q
```

本地当前朴素实现耗时约 2.2s，测试要求小于 1.5s。

## 9. 第六步：优化 train_bpe 以通过 speed

### 问题背景

朴素版本慢在每一轮都全量重算 pair。实际上每次合并后，只有包含 best pair 的 word 发生变化；其他 word 的 pair 计数不变。

### 学生要做的事情

优化方向：

1. 维护 `pair_counts: dict[pair, int]`。
2. 维护 `pair_to_words: dict[pair, set[word]]`，记录每个 pair 出现在哪些 word 里。
3. 每轮选 `best` 后，只取 `affected_words = pair_to_words[best]`。
4. 对 affected words：
   - 先从 `pair_counts` 中减掉旧 word 的所有 pair 贡献。
   - merge 成 new word。
   - 再把 new word 的所有 pair 加回 `pair_counts`。
   - 更新 `word_counts` 和 `pair_to_words`。

### 相关答案

伪代码：

```python
word_counts = _pretoken_counts(...)
pair_counts, pair_to_words = build_indexes(word_counts)

while len(vocab) < vocab_size:
    best = max(pair_counts, key=lambda p: (pair_counts[p], p))
    affected = list(pair_to_words[best])

    for old_word in affected:
        count = word_counts.pop(old_word, 0)
        if count == 0:
            continue

        # 旧 word 的 pair 从索引中移除
        for pair in pairs(old_word):
            pair_counts[pair] -= count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            pair_to_words[pair].discard(old_word)

        new_word = _merge_word(old_word, best)
        word_counts[new_word] += count

        # 新 word 的 pair 加入索引
        for pair in pairs(new_word):
            pair_counts[pair] += count
            pair_to_words[pair].add(new_word)
```

实现时有一个小坑：多个 old word 可能 merge 成同一个 new word，所以 `word_counts[new_word] += count` 后，`pair_to_words[pair].add(new_word)` 用 set 是合适的。但如果你想严格维护旧索引，要确保 discard 不报错，并且 `pair_counts` 为 0 时删除。

对于本作业的小测试，通常只要避免每轮重建全部 `word_counts` 和全量 pair 统计，就能压到 1.5s 内。

## 10. 第七步：Tokenizer 初始化

### 问题背景

`Tokenizer` 接收已有 `vocab` 和 `merges`，不是自己训练。测试会传入 GPT-2 的真实 vocab/merges 转换后的 bytes 版本。

### 学生要做的事情

实现：

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
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
        self.special_to_id = {
            tok: self.token_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens
        }
```

### 相关答案

`special_tokens` 必须按长度降序排序，因为测试有重叠 special token：

```python
["<|endoftext|>", "<|endoftext|><|endoftext|>"]
```

如果不按长的优先，`<|endoftext|><|endoftext|>` 会被拆成两个 `<|endoftext|>`，`test_overlapping_special_tokens` 会失败。

## 11. 第八步：对一个 pre-token 应用 BPE

### 问题背景

编码时不是重新统计频率，而是按训练得到的 merges 优先级应用。`merge_rank` 越小，优先级越高。

### 学生要做的事情

实现：

```python
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
        tokens = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]

    return tokens
```

### 相关答案

这个写法虽然不是最高效，但对测试语料足够，并且最不容易错。关键是“每次找当前序列里 rank 最小的可合并 pair”，而不是盲目遍历 merges 并替换一次就结束。

## 12. 第九步：encode 普通文本

### 问题背景

普通文本编码流程：

```text
text -> regex pre-token -> UTF-8 bytes -> BPE merges -> token id
```

### 学生要做的事情

实现：

```python
def _encode_regular(self, text: str) -> list[int]:
    ids = []
    for match in re.finditer(PAT, text):
        bs = match.group(0).encode("utf-8")
        for tok in self._apply_bpe(bs):
            ids.append(self.token_to_id[tok])
    return ids
```

### 相关答案

跑这些测试：

```bash
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_empty_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_single_character_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_unicode_string_matches_tiktoken -q
```

如果失败，优先检查：

1. 是否用了 `regex` 包。
2. `PAT` 是否完全一致。
3. `merge_rank` 是否按 merges 顺序建立。
4. `vocab` 是否是 bytes 到 id 的映射。

## 13. 第十步：encode special token

### 问题背景

如果构造 Tokenizer 时传了 special tokens，编码时必须先切出 special token，再对普通片段调用 `_encode_regular`。

### 学生要做的事情

实现：

```python
def encode(self, text: str) -> list[int]:
    if not self.special_tokens:
        return self._encode_regular(text)

    pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
    ids = []
    for part in re.split(pattern, text):
        if part == "":
            continue
        if part in self.special_to_id:
            ids.append(self.special_to_id[part])
        else:
            ids.extend(self._encode_regular(part))
    return ids
```

### 相关答案

跑：

```bash
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_overlapping_special_tokens -q
```

如果重叠 special token 测试失败，通常是 `self.special_tokens` 没有按长度降序排序。

## 14. 第十一步：decode

### 问题背景

decode 比 encode 简单：id 查 vocab，拼接 bytes，用 UTF-8 解码。

### 学生要做的事情

实现：

```python
def decode(self, ids: list[int]) -> str:
    return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

### 相关答案

`errors="replace"` 是作业要求：如果 token id 序列拼出来的 bytes 不是合法 UTF-8，应该用 Unicode replacement character 替换坏字节，而不是报错。

跑：

```bash
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -k roundtrip -q
```

## 15. 第十二步：encode_iterable

### 问题背景

`encode_iterable` 是为了大文件流式编码，不能先把整个文件读入内存。

### 学生要做的事情

最小实现：

```python
def encode_iterable(self, iterable):
    for chunk in iterable:
        yield from self.encode(chunk)
```

### 相关答案

这个版本能通过当前 Windows 上可运行的测试：

```bash
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken -q
```

更严谨的实现要注意 chunk 边界不能切断一个 regex pre-token，否则流式编码会和整段编码不同。测试目前用文件逐行迭代，行边界足够通过这些 fixture；Linux 内存测试主要检查你没有一次性读完整 5MB 文件。

## 16. 推荐实现和测试顺序

### 问题背景

BPE 很容易因为一次写太多而不知道错在哪里。最稳的方式是每写一个函数跑一小组测试或手写检查。

### 学生要做的事情

按下面顺序推进：

1. 写 vocab 初始化、special token 加入。
2. 写 `_pretoken_counts`，手动 print 小字符串结果。
3. 写 `_count_pairs` 和 `_merge_word`。
4. 写朴素 `train_bpe`。
5. 跑 BPE 正确性测试。
6. 写 `Tokenizer.__init__` 和 `decode`。
7. 写 `_apply_bpe` 和 `_encode_regular`。
8. 写 `encode` 的 special token 分支。
9. 写 `encode_iterable`。
10. 最后优化 `train_bpe` speed。

### 相关答案

命令顺序：

```bash
# 训练正确性
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py::test_train_bpe -q
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -q

# 编码/解码基础
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_roundtrip_empty -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_single_character_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_unicode_string_matches_tiktoken -q

# special tokens
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_overlapping_special_tokens -q

# 文件级样本
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_address_matches_tiktoken -q
.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py::test_tinystories_matches_tiktoken -q

# 全部 BPE/tokenizer
.venv\Scripts\python.exe -m pytest tests/test_train_bpe.py tests/test_tokenizer.py -q
```

如果用 `uv` 遇到缓存权限错误，可以先用 `.venv\Scripts\python.exe -m pytest ...`。本机曾出现 `.pytest_cache` 无法写入 warning，不影响测试断言。

## 17. 常见失败点对照表

### 问题背景

多数 BPE 失败都能从测试名定位。

### 学生要做的事情

看到失败后先按表排查。

### 相关答案

```text
test_train_bpe 失败：
  - tie-break 没有用 max(pair_counts, key=lambda p: (count, p))
  - special token 没有在训练前 split 掉
  - vocab value 不是 bytes
  - merge 时发生了重叠合并

test_train_bpe_special_tokens 失败：
  - special token 参与了 pair 统计
  - 没有把 special token 当硬边界
  - re.split 没有 re.escape，导致 <|endoftext|> 中的 | 被当正则或

test_train_bpe_speed 失败：
  - 每一轮全量重算 pair_counts
  - 每一轮全量 merge 所有 word
  - 没有维护 pair_to_words 或局部更新索引

matches_tiktoken 失败：
  - PAT 不一致
  - 没有用 regex 包
  - BPE encode 没有按 merge rank 优先级
  - GPT-2 vocab/merges 已经由测试转换成 bytes，不要再做 bytes_to_unicode 映射

overlapping_special_tokens 失败：
  - special token 没有按长度降序匹配

roundtrip 失败：
  - decode 没有 bytes join
  - decode 没有 errors="replace"
  - encode 对 Unicode 多字节字符逐 byte 解码了，而不是处理 bytes
```

## 18. 当前仓库的下一步

### 问题背景

当前 `cs336_basics/tokenizer.py` 已经是一个朴素但正确的实现。它通过了 tokenizer 相关测试，也通过了 BPE 的正确性和 special token 测试。

### 学生要做的事情

如果你要继续让 BPE 全绿，优先改 `train_bpe` 的合并阶段，不要动已经通过的 `Tokenizer` 编码逻辑。

### 相关答案

当前最小目标：

```text
保持 test_train_bpe 和 test_train_bpe_special_tokens 通过；
把 test_train_bpe_speed 从约 2.2s 降到 < 1.5s。
```

建议只优化这些内部函数或新增辅助索引函数：

```python
def _pairs(word): ...
def _build_pair_indexes(word_counts): ...
def _remove_word_from_indexes(word, count, pair_counts, pair_to_words): ...
def _add_word_to_indexes(word, count, pair_counts, pair_to_words): ...
```

不要改 adapter 的接口；测试依赖的外部返回格式必须保持：

```python
tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

