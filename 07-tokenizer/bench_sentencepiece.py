"""
bench_sentencepiece.py - the approach Thai and Indic NLP actually converged on.

WHY THIS SHOULD BEAT BYTE-BPE ON KHMER
Byte-level BPE (minbpe, GPT-2/4) starts from 256 byte values. Khmer codepoints are
3 UTF-8 bytes each, so roughly 2 merges are spent rebuilding every single CHARACTER
before any subword structure can be learned. English gets its characters free at one
byte apiece. SentencePiece works on the Unicode character stream directly and never
pays that tax.

WHY THAI IS THE RIGHT REFERENCE, NOT HINDI
Thai, like Khmer, has no spaces between words. Thai NLP standardised on SentencePiece
unigram (WangchanBERTa). Indic scripts share Khmer's combining marks and conjuncts
but DO use spaces, so whitespace pre-tokenization works there and cannot work here.

TWO GOTCHAS THAT SILENTLY CORRUPT KHMER
  normalization_rule_name='identity'  - the default (nmt_nfkc) applies Unicode
      normalization that can reorder or fold Khmer combining marks. Roundtrip breaks
      and you get a tokenizer that quietly alters the text.
  split_by_whitespace=False - Khmer words are not whitespace-delimited, so the
      default whitespace pre-split lumps whole phrases into single pieces.
"""
import os, tempfile, sentencepiece as spm

TRAIN_CHARS, TEST_CHARS = 120_000, 20_000
VOCABS = [512, 2048, 4096, 8000]
ZWSP = '​'

full = open('khmer_only.txt', encoding='utf-8').read()
train_text, test_text = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
eng = open('../06-gpt/input.txt', encoding='utf-8').read()
eng_train, eng_test = eng[:TRAIN_CHARS], eng[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]

tmp = tempfile.mkdtemp()
km_f, en_f = os.path.join(tmp, 'km.txt'), os.path.join(tmp, 'en.txt')
open(km_f, 'w', encoding='utf-8').write(train_text)
open(en_f, 'w', encoding='utf-8').write(eng_train)


def n_words(t):
    """Khmer 'words' = runs delimited by ZWSP or whitespace."""
    return len([w for w in t.replace(ZWSP, ' ').split() if w])


def run(train_file, test, vocab, model_type, split_ws, tag):
    prefix = os.path.join(tmp, f"{tag}_{model_type}_{vocab}_{split_ws}")
    spm.SentencePieceTrainer.train(
        input=train_file, model_prefix=prefix, vocab_size=vocab,
        model_type=model_type,
        character_coverage=1.0,
        split_by_whitespace=split_ws,
        normalization_rule_name='identity',   # do NOT fold Khmer combining marks
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        byte_fallback=True,
        minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=prefix + '.model')
    ids = sp.encode(test)
    ok = sp.decode(ids) == test
    return len(test) / len(ids), len(ids) / max(n_words(test), 1), ok


print(f"train={TRAIN_CHARS:,} chars  test={TEST_CHARS:,} chars (held out)")
print("chars/token - higher is better\n")
print(f"{'tokenizer':<34}" + "".join(f"{'v='+str(v):>10}" for v in VOCABS) + "   rt")
print("-" * 82)

rows = {}
CONFIGS = [
    ("SP unigram  (no ws split)", 'unigram', False, km_f, test_text, 'km'),
    ("SP unigram  (ws split)",    'unigram', True,  km_f, test_text, 'km'),
    ("SP bpe      (no ws split)", 'bpe',     False, km_f, test_text, 'km'),
    ("SP bpe      (ws split)",    'bpe',     True,  km_f, test_text, 'km'),
]
for label, mt, ws, f, te, tag in CONFIGS:
    cells, fert, allok = [], [], True
    for v in VOCABS:
        try:
            cpt, tpw, ok = run(f, te, v, mt, ws, tag)
        except Exception as e:
            cells.append(None); fert.append(None); continue
        cells.append(cpt); fert.append(tpw); allok &= ok
    rows[label] = (cells, fert)
    print(f"{label:<34}" + "".join(f"{c:>10.3f}" if c else f"{'-':>10}" for c in cells)
          + f"   {'ok' if allok else 'FAIL'}")

print()
eng_cells = []
for v in VOCABS:
    cpt, _, _ = run(en_f, eng_test, v, 'unigram', True, 'en')
    eng_cells.append(cpt)
print(f"{'ENGLISH SP unigram (ws split)':<34}" + "".join(f"{c:>10.3f}" for c in eng_cells))

print("\n--- tokens per Khmer word (fertility, lower is better) ---")
print(f"{'tokenizer':<34}" + "".join(f"{'v='+str(v):>10}" for v in VOCABS))
for label, (_, fert) in rows.items():
    print(f"{label:<34}" + "".join(f"{x:>10.3f}" if x else f"{'-':>10}" for x in fert))

print("\n--- Khmer cost relative to English at the same vocab ---")
for label, (cells, _) in rows.items():
    print(f"{label:<34}" + "".join(
        f"{eng_cells[i]/c:>9.2f}x" if c else f"{'-':>10}" for i, c in enumerate(cells)))

best = max(rows.items(), key=lambda kv: kv[1][0][-1] or 0)
print(f"\nbest SentencePiece config at v={VOCABS[-1]}: {best[0].strip()} "
      f"({best[1][0][-1]:.3f} chars/token)")

print("\n--- sample pieces (unigram, no ws split, v=4096) ---")
prefix = os.path.join(tmp, f"km_unigram_4096_False")
sp = spm.SentencePieceProcessor(model_file=prefix + '.model')
sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញ"
print(f"  {sample}  ({len(sample)} chars)")
print(f"  -> {len(sp.encode(sample))} tokens: {sp.encode(sample, out_type=str)}")
