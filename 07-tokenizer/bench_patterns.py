"""
bench_patterns.py - which regex pre-split pattern is best for Khmer BPE?

Holds everything constant (corpus, vocab size, BPE algorithm) and varies ONLY the
pre-split pattern. Reports fertility on HELD-OUT text, so a pattern cannot win by
memorising the training split.

Key metric: characters per token. Higher is better - it means the same sentence
costs fewer tokens, which is directly what a Khmer speaker pays for in context
length, latency and API cost.
"""
import time, regex as re
from minbpe.regex import RegexTokenizer, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, KHMER_SPLIT_PATTERN

VOCAB = 512
TRAIN_CHARS, TEST_CHARS = 120_000, 20_000
NO_SPLIT = r"[\s\S]+"    # one chunk = plain byte-level BPE, the baseline

full = open('khmer_only.txt', encoding='utf-8').read()
train_text, test_text = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
eng = open('../06-gpt/input.txt', encoding='utf-8').read()
eng_train, eng_test = eng[:TRAIN_CHARS], eng[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]

PATTERNS = [
    ("none (byte BPE)", NO_SPLIT,             train_text, test_text),
    ("GPT-2",           GPT2_SPLIT_PATTERN,   train_text, test_text),
    ("GPT-4",           GPT4_SPLIT_PATTERN,   train_text, test_text),
    ("Khmer syllable",  KHMER_SPLIT_PATTERN,  train_text, test_text),
    ("GPT-4 on ENGLISH", GPT4_SPLIT_PATTERN,  eng_train,  eng_test),
]

print(f"vocab={VOCAB}  train={TRAIN_CHARS:,} chars  test={TEST_CHARS:,} chars (held out)\n")
print(f"{'pattern':<18} {'chunks':>8} {'ch/chunk':>9} {'maxchunk':>9} "
      f"{'tokens':>8} {'chars/tok':>10} {'bytes/tok':>10} {'train s':>8}  rt")
print("-" * 96)

results = []
for name, pat, tr, te in PATTERNS:
    chunks = re.findall(re.compile(pat), tr)
    n_chunks = len(chunks)
    mean_c = sum(len(c) for c in chunks) / n_chunks
    max_c = max(len(c) for c in chunks)

    t0 = time.time()
    tok = RegexTokenizer(pattern=pat)
    tok.train(tr, VOCAB, verbose=False)
    dt = time.time() - t0

    ids = tok.encode(te)
    rt = "ok" if tok.decode(ids) == te else "FAIL"
    cpt = len(te) / len(ids)
    bpt = len(te.encode('utf-8')) / len(ids)
    results.append((name, cpt))
    print(f"{name:<18} {n_chunks:>8,} {mean_c:>9.2f} {max_c:>9,} "
          f"{len(ids):>8,} {cpt:>10.3f} {bpt:>10.3f} {dt:>8.1f}  {rt}")

print("\n--- fertility vs the English baseline ---")
eng_cpt = dict(results)["GPT-4 on ENGLISH"]
for name, cpt in results:
    if name == "GPT-4 on ENGLISH":
        continue
    print(f"{name:<18} {eng_cpt/cpt:>5.2f}x more tokens than English for the same character count")

print("\n--- what the Khmer pattern actually learned (sample merges) ---")
tok = RegexTokenizer(pattern=KHMER_SPLIT_PATTERN)
tok.train(train_text, VOCAB, verbose=False)
for idx in list(range(256, 276)):
    print(f"  {idx}  {tok.vocab[idx].decode('utf-8', errors='replace')!r}")
sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញ"
print(f"\nsample: {sample}  ({len(sample)} chars)")
for name, pat in [("GPT-4", GPT4_SPLIT_PATTERN), ("Khmer", KHMER_SPLIT_PATTERN)]:
    t = RegexTokenizer(pattern=pat); t.train(train_text, VOCAB, verbose=False)
    ids = t.encode(sample)
    print(f"  {name:<6} {len(ids):>3} tokens: {[t.vocab[i].decode('utf-8',errors='replace') for i in ids]}")
