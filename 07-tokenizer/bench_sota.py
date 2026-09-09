"""
bench_sota.py - your 3.5k Khmer SentencePiece vs production multilingual tokenizers.

TWO METRICS, BOTH NEEDED:

  chars/token (absolute)  - what a Khmer speaker actually pays in context window,
      latency and API cost. Unfair to a 3.5k vocab in one sense, but it is the real
      number: nobody gets to spend a 200k vocab on Khmer alone.

  Khmer-capable vocab     - how many of the tokenizer's pieces contain Khmer script
      at all. This is where the unfairness inverts: a 200k multilingual vocab may
      dedicate a fraction of one percent to Khmer.
"""
import os, tempfile, sentencepiece as spm

TRAIN_CHARS, TEST_CHARS = 400_000, 20_000
full = open('khmer_only.txt', encoding='utf-8').read()
km_train, km_test = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
is_khmer = lambda ch: 0x1780 <= ord(ch) <= 0x17FF or 0x19E0 <= ord(ch) <= 0x19FF

rows = []

# ---- ours ----
tmp = tempfile.mkdtemp()
f = os.path.join(tmp, 'km.txt'); open(f, 'w', encoding='utf-8').write(km_train)
for v in [4000, 8000, 16000]:
    pre = os.path.join(tmp, f'ours{v}')
    spm.SentencePieceTrainer.train(
        input=f, model_prefix=pre, vocab_size=v, model_type='bpe',
        character_coverage=1.0, split_by_whitespace=False,
        normalization_rule_name='identity', remove_extra_whitespaces=False,
        add_dummy_prefix=False, byte_fallback=True, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=pre + '.model')
    n = len(sp.encode(km_test))
    kv = sum(1 for i in range(sp.get_piece_size())
             if any(is_khmer(c) for c in sp.id_to_piece(i)))
    rows.append((f"OURS SP-bpe (Khmer only)", v, n, kv, kv / v * 100))

# ---- tiktoken (OpenAI) ----
try:
    import tiktoken
    for name, label in [('cl100k_base', 'GPT-4 / 3.5-turbo'), ('o200k_base', 'GPT-4o / o-series')]:
        try:
            enc = tiktoken.get_encoding(name)
            n = len(enc.encode(km_test))
            kv = 0
            for i in range(enc.n_vocab):
                try:
                    s = enc.decode_single_token_bytes(i).decode('utf-8')
                except Exception:
                    continue
                if any(is_khmer(c) for c in s):
                    kv += 1
            rows.append((f"{label} ({name})", enc.n_vocab, n, kv, kv / enc.n_vocab * 100))
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}")
except ImportError:
    pass

# ---- HuggingFace tokenizer.json ----
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HF = [
    ("XLM-RoBERTa",        "FacebookAI/xlm-roberta-base"),
    ("Qwen2.5 (Sailor2 base)", "Qwen/Qwen2.5-0.5B"),
    ("NLLB-200",           "facebook/nllb-200-distilled-600M"),
    ("mT5",                "google/mt5-base"),
    ("Gemma 2 (SEA-LION v4 family)", "google/gemma-2-2b"),
    ("SEA-LION v4",        "aisingapore/Gemma-SEA-LION-v4-27B-IT"),
    ("Sailor2",            "sail/Sailor2-1B"),
]
for label, repo in HF:
    try:
        p = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        tk = Tokenizer.from_file(p)
        n = len(tk.encode(km_test).ids)
        vocab = tk.get_vocab()
        V = len(vocab)
        kv = 0
        for i in range(V):
            s = tk.decode([i], skip_special_tokens=False)
            if s and any(is_khmer(c) for c in s):
                kv += 1
        rows.append((f"{label}", V, n, kv, kv / V * 100))
        print(f"  loaded {label}")
    except Exception as e:
        print(f"  skip {label} ({repo}): {type(e).__name__}: {str(e)[:80]}")

print(f"\ntest = {TEST_CHARS:,} Khmer characters (held out)\n")
print(f"{'tokenizer':<34}{'vocab':>9}{'tokens':>9}{'chars/tok':>11}{'Khmer pieces':>14}{'%':>7}")
print("-" * 84)
for label, V, n, kv, pct in sorted(rows, key=lambda r: -TEST_CHARS / r[2]):
    print(f"{label:<34}{V:>9,}{n:>9,}{TEST_CHARS/n:>11.3f}{kv:>14,}{pct:>6.2f}%")

best_ours = max((r for r in rows if r[0].startswith('OURS')), key=lambda r: TEST_CHARS / r[2])
print(f"\nbest ours: {best_ours[1]:,} vocab -> {TEST_CHARS/best_ours[2]:.3f} chars/token")
for label, V, n, kv, pct in rows:
    if not label.startswith('OURS'):
        r = (TEST_CHARS / best_ours[2]) / (TEST_CHARS / n)
        verdict = "we WIN" if r > 1 else "they win"
        print(f"  vs {label:<32} {r:>5.2f}x   {verdict}   "
              f"(they use {V/best_ours[1]:.0f}x our vocab)")
