"""
bench_patterns2.py - pattern choice as a function of vocab size.

bench_patterns.py showed no-split winning at vocab=512, but that regime is dominated
by UTF-8 reassembly: Khmer is 3 bytes/char, so ~2 of every merge budget go to
rebuilding single characters before any subword learning starts. Sweep vocab to see
where the ordering settles, and add the missing candidate: word-level splitting on
the zero-width space, which is Khmer's actual word boundary.
"""
import time, regex as re
from minbpe.regex import RegexTokenizer, GPT4_SPLIT_PATTERN, KHMER_SPLIT_PATTERN

ZWSP = '​'
NO_SPLIT = r"[\s\S]+"
# maximal runs of Khmer script = one word, since ZWSP/space delimit them
KHMER_WORD_PATTERN = (
    r"[ក-៿]+"
    rf"|{ZWSP}"
    r"|\p{N}+"
    r"|\s+"
    rf"|[^\p{{Khmer}}\s{ZWSP}]+"
)

TRAIN_CHARS, TEST_CHARS = 120_000, 20_000
VOCABS = [512, 2048, 4096]

full = open('khmer_only.txt', encoding='utf-8').read()
train_text, test_text = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
eng = open('../06-gpt/input.txt', encoding='utf-8').read()
eng_train, eng_test = eng[:TRAIN_CHARS], eng[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]

PATTERNS = [
    ("none",           NO_SPLIT),
    ("GPT-4",          GPT4_SPLIT_PATTERN),
    ("Khmer syllable", KHMER_SPLIT_PATTERN),
    ("Khmer word",     KHMER_WORD_PATTERN),
]

print(f"train={TRAIN_CHARS:,} chars  test={TEST_CHARS:,} chars (held out)")
print(f"metric = characters per token on held-out text, higher is better\n")
print(f"{'pattern':<16} " + "".join(f"{'v='+str(v):>12}" for v in VOCABS) + f"{'ch/chunk':>10}")
print("-" * 66)

grid, reached = {}, {}
for name, pat in PATTERNS:
    chunks = re.findall(re.compile(pat), train_text)
    mean_c = sum(len(c) for c in chunks) / len(chunks)
    row = []
    for v in VOCABS:
        tok = RegexTokenizer(pattern=pat)
        tok.train(train_text, v, verbose=False)
        ids = tok.encode(test_text)
        assert tok.decode(ids) == test_text, f"roundtrip failed {name} v={v}"
        cpt = len(test_text) / len(ids)
        grid[(name, v)] = cpt
        reached[(name, v)] = len(tok.vocab)
        row.append(cpt)
    print(f"{name:<16} " + "".join(
        f"{c:>11.3f}{'*' if reached[(name,v)] < v else ' '}"
        for c, v in zip(row, VOCABS)) + f"{mean_c:>10.2f}")

print(f"\n{'ENGLISH (GPT-4)':<16} ", end="")
eng_row = []
for v in VOCABS:
    tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    tok.train(eng_train, v, verbose=False)
    ids = tok.encode(eng_test)
    eng_row.append(len(eng_test) / len(ids))
print("".join(f"{c:>12.3f}" for c in eng_row))

print("\n--- Khmer token cost relative to English, same character count ---")
print(f"{'pattern':<16} " + "".join(f"{'v='+str(v):>12}" for v in VOCABS))
for name, _ in PATTERNS:
    print(f"{name:<16} " + "".join(f"{eng_row[i]/grid[(name,v)]:>11.2f}x" for i, v in enumerate(VOCABS)))

best = max(PATTERNS, key=lambda p: grid[(p[0], VOCABS[-1])])[0]
print(f"\nbest at vocab={VOCABS[-1]}: {best}  ({grid[(best, VOCABS[-1])]:.3f} chars/token)")

print("\n* = merges exhausted before reaching that vocab size:")
for (n, v), r in sorted(reached.items()):
    if r < v: print(f"    {n:<16} v={v:<5} reached only {r}")
print("\n--- sample encoding at the largest vocab ---")
sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញ"
for name, pat in PATTERNS:
    t = RegexTokenizer(pattern=pat); t.train(train_text, VOCABS[-1], verbose=False)
    ids = t.encode(sample)
    toks = [t.vocab[i].decode('utf-8', errors='replace') for i in ids]
    print(f"  {name:<16} {len(ids):>3} tokens: {toks}")
