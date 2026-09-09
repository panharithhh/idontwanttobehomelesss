"""
Diagnostic harness for a byte-level BPE tokenizer on Khmer.

Usage from the notebook, once you have a `merges` dict:

    from eval_tokenizer import report
    report(merges)

Reports chars/token per category, so you can see *where* the tokenizer is weak
rather than only that it averages some number. Also verifies the one thing that
actually defines correctness: decode(encode(x)) == x.
"""

import json
from pathlib import Path

TESTSET = Path(__file__).with_name("tokenizer_testset.json")

CATEGORY_NOTES = {
    "pure_prose": "clean Khmer, no digits or Latin",
    "khmer_numerals": "Khmer digits ០-៩",
    "arabic_numerals": "Western digits in Khmer text",
    "latin_mixed": "code-switching, Latin inside Khmer",
    "zwsp_present": "contains U+200B word markers",
    "stacked_coeng": "multiple subscript consonants",
    "long_compounds": "words of 22+ characters",
    "probes": "hand-picked hard cases",
}


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def report(merges, testset=TESTSET):
    data = json.loads(Path(testset).read_text(encoding="utf-8"))
    vocab = build_vocab(merges)

    fragments = 0
    for idx in range(256, 256 + len(merges)):
        try:
            vocab[idx].decode("utf-8")
        except UnicodeDecodeError:
            fragments += 1

    print(f"vocab {256 + len(merges):,}   merges {len(merges):,}   "
          f"fragments {fragments}/{len(merges)} ({fragments / max(len(merges), 1) * 100:.0f}%)")
    print()
    print(f"{'category':18} {'lines':>5} {'chars':>7} {'tokens':>7} {'ch/tok':>7}  {'':2} note")
    print("-" * 88)

    all_chars = all_tokens = 0
    failures = []
    for name, lines in data.items():
        if not lines:
            continue
        chars = tokens = 0
        for line in lines:
            ids = encode(line, merges)
            if decode(ids, vocab) != line:
                failures.append((name, line))
            chars += len(line)
            tokens += len(ids)
        all_chars += chars
        all_tokens += tokens
        flag = "  " if chars / tokens >= 2.0 else "!!"
        print(f"{name:18} {len(lines):>5} {chars:>7,} {tokens:>7,} {chars / tokens:>7.3f}  {flag} "
              f"{CATEGORY_NOTES.get(name, '')}")

    print("-" * 88)
    print(f"{'OVERALL':18} {'':>5} {all_chars:>7,} {all_tokens:>7,} {all_chars / all_tokens:>7.3f}")
    print()
    if failures:
        print(f"ROUND TRIP: FAIL on {len(failures)} lines")
        for name, line in failures[:3]:
            print(f"  [{name}] {line[:70]}")
    else:
        print("ROUND TRIP: PASS on every line — encoding is lossless")

    return all_chars / all_tokens
