"""
make_khmer_only.py - distil khmer_corpus.txt down to Khmer script only.

Two stages, in this order for a reason:

  1. LINE filter. Keep a line only if it is already overwhelmingly Khmer. Doing this
     first means we never cut through the middle of a sentence - an English sentence
     is dropped whole rather than leaving its punctuation and spaces behind.
  2. CHARACTER filter. Remove whatever non-Khmer residue survives inside kept lines.
     Because those lines were already >=90% Khmer, this removes stray characters
     rather than mangling text.

Khmer punctuation (។ ៕ ៗ) and Khmer digits (០-៩) live inside the Khmer block, so
they are kept automatically. ZWSP is not in the block but is functionally part of
Khmer text, so it gets an explicit flag.
"""
from collections import Counter
from pathlib import Path

SRC = Path('khmer_corpus.txt')
DST = Path('khmer_only.txt')

MIN_KHMER_RATIO = 0.90   # a line must be at least this fraction Khmer to survive
MIN_KHMER_CHARS = 10     # ...and this long, to drop stubs and headings
KEEP_ZWSP = True         # U+200B word-boundary marks
KEEP_ZWNJ = False        # U+200C, much rarer and less consistently used

ZWSP, ZWNJ = '​', '‌'


def is_khmer(ch):
    o = ord(ch)
    return 0x1780 <= o <= 0x17FF or 0x19E0 <= o <= 0x19FF


allowed_extra = {' ', '\n'}
if KEEP_ZWSP:
    allowed_extra.add(ZWSP)
if KEEP_ZWNJ:
    allowed_extra.add(ZWNJ)

raw = SRC.read_text(encoding='utf-8')
lines = raw.split('\n')

kept, dropped_lines = [], 0
for line in lines:
    meaningful = [c for c in line if not c.isspace() and c not in (ZWSP, ZWNJ)]
    if not meaningful:
        continue
    nk = sum(1 for c in meaningful if is_khmer(c))
    if nk >= MIN_KHMER_CHARS and nk / len(meaningful) >= MIN_KHMER_RATIO:
        kept.append(line)
    else:
        dropped_lines += 1

stage1 = '\n'.join(kept)
stage2 = ''.join(c for c in stage1 if is_khmer(c) or c in allowed_extra)

# collapse runs of spaces left behind by removed characters
while '  ' in stage2:
    stage2 = stage2.replace('  ', ' ')
stage2 = '\n'.join(l.strip() for l in stage2.split('\n') if l.strip())

DST.write_text(stage2, encoding='utf-8')

c_in, c_1, c_out = len(raw), len(stage1), len(stage2)
kh = sum(1 for c in stage2 if is_khmer(c))
print(f"source          {c_in:>10,} chars   {len(set(raw)):>4} codepoints   {len(lines):>7,} lines")
print(f"after line cut  {c_1:>10,} chars   {len(set(stage1)):>4} codepoints   {len(kept):>7,} lines "
      f"({dropped_lines:,} dropped)")
print(f"after char cut  {c_out:>10,} chars   {len(set(stage2)):>4} codepoints")
print(f"\nKhmer script:   {kh/c_out*100:.2f}%   ZWSP kept: {stage2.count(ZWSP):,}")
print(f"retained {c_out/c_in*100:.1f}% of the original text")
print(f"\ntiny Shakespeare is 1,115,394 chars / 65 codepoints - for comparison")
print(f"\npreview:\n{stage2[:200]}")
