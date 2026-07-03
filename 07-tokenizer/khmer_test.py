"""
Quick demo of minbpe on Khmer text.

Trains a byte-level BPE tokenizer (BasicTokenizer) on khmer.txt, then tests:
  - encode/decode roundtrip is lossless
  - compression ratio (utf-8 bytes -> tokens)
  - what Khmer subword tokens it learned

Run:  python khmer_test.py
(For a bigger/better vocab, point CORPUS at "khmer_corpus.txt".)
"""
from minbpe import BasicTokenizer

CORPUS = "khmer.txt"      # swap to "khmer_corpus.txt" for the 3.6MB corpus
VOCAB_SIZE = 512          # 256 base bytes + 256 learned merges

# 1) load the Khmer text
text = open(CORPUS, encoding="utf-8").read()
nbytes = len(text.encode("utf-8"))
print(f"loaded {CORPUS}: {len(text)} chars, {nbytes} utf-8 bytes\n")

# 2) train a byte-level BPE tokenizer
tok = BasicTokenizer()
tok.train(text, VOCAB_SIZE, verbose=True)

# 3) test roundtrip: decode(encode(x)) == x
ids = tok.encode(text)
assert tok.decode(ids) == text, "roundtrip failed!"
print("\nroundtrip OK (decode(encode(text)) == text)")

# 4) compression ratio (bytes -> tokens)
print(f"bytes: {nbytes}  tokens: {len(ids)}  compression: {nbytes/len(ids):.2f}X")

# 5) peek at some learned Khmer tokens
print("\nsome learned tokens:")
for idx in range(256, 256 + min(20, len(tok.merges))):
    print(idx, repr(tok.vocab[idx].decode("utf-8", errors="replace")))

# 6) try encoding a fresh sentence
sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញ"
enc = tok.encode(sample)
print(f"\nsample: {sample}")
print(f"encoded ({len(enc)} tokens): {enc}")
print(f"decoded: {tok.decode(enc)}")

# 7) save model + human-readable vocab
tok.save("khmer_basic")
print("\nsaved khmer_basic.model and khmer_basic.vocab")

# -----------------------------------------------------------------------------
# Want the Khmer-aware RegexTokenizer instead? Install regex (pip install regex),
# then:
#
#   from minbpe import RegexTokenizer
#   from minbpe.regex import KHMER_SPLIT_PATTERN
#   tok = RegexTokenizer(pattern=KHMER_SPLIT_PATTERN)
#   tok.train(text, VOCAB_SIZE, verbose=True)
#
# This splits text into Khmer orthographic syllables first, so merges never
# cross a syllable boundary.
