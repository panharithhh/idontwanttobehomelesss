"""
bench_methods.py - every tokenization method that plausibly fits Khmer.

The untested candidate: SEGMENT INTO REAL WORDS FIRST, then run subword BPE inside
those boundaries. This is what Thai NLP does with newmm, and what Indic NLP does with
whitespace it already has. Khmer has no spaces, but khmercut/khmer-nltk supply the
boundaries a segmenter would.

Metric is characters of ORIGINAL text per token, so a pipeline that inserts word
delimiters is still charged for the text a user actually typed.
"""
import os, time, tempfile, warnings
warnings.filterwarnings('ignore')
import sentencepiece as spm

TRAIN_CHARS, TEST_CHARS = 400_000, 20_000
VOCABS = [4000, 8000, 16000]

full = open('khmer_only.txt', encoding='utf-8').read()
km_train, km_test = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
tmp = tempfile.mkdtemp()

from khmercut import tokenize as kcut
t0 = time.time()
seg_train = ' '.join(kcut(km_train))
seg_test = ' '.join(kcut(km_test))
seg_time = time.time() - t0
words = kcut(km_test)
print(f"khmercut segmented {TRAIN_CHARS+TEST_CHARS:,} chars in {seg_time:.1f}s "
      f"({(TRAIN_CHARS+TEST_CHARS)/seg_time/1000:.0f}k chars/s)")
print(f"test slice: {len(words):,} words, {len(km_test)/len(words):.2f} chars/word\n")

paths = {}
for tag, txt in [('raw', km_train), ('seg', seg_train)]:
    paths[tag] = os.path.join(tmp, f'{tag}.txt')
    open(paths[tag], 'w', encoding='utf-8').write(txt)


def sp_run(tag, test, vocab, model_type, split_ws):
    pre = os.path.join(tmp, f"{tag}_{model_type}_{vocab}_{split_ws}")
    spm.SentencePieceTrainer.train(
        input=paths[tag], model_prefix=pre, vocab_size=vocab, model_type=model_type,
        character_coverage=1.0, split_by_whitespace=split_ws,
        normalization_rule_name='identity', remove_extra_whitespaces=False,
        add_dummy_prefix=False, byte_fallback=True, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=pre + '.model')
    return len(sp.encode(test)), sp


def wordpiece_run(train_txt, test_txt, vocab, pretok):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=200))
    if pretok:
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tr = trainers.WordPieceTrainer(vocab_size=vocab, special_tokens=["[UNK]"],
                                   show_progress=False)
    tk.train_from_iterator([train_txt], tr)
    return len(tk.encode(test_txt).ids)


METHODS = [
    ("SP bpe          (raw)",      lambda v: sp_run('raw', km_test,  v, 'bpe',     False)[0]),
    ("SP unigram      (raw)",      lambda v: sp_run('raw', km_test,  v, 'unigram', False)[0]),
    ("SP bpe      + khmercut",     lambda v: sp_run('seg', seg_test, v, 'bpe',     True)[0]),
    ("SP unigram  + khmercut",     lambda v: sp_run('seg', seg_test, v, 'unigram', True)[0]),
    ("WordPiece   + khmercut",     lambda v: wordpiece_run(seg_train, seg_test, v, True)),
    ("SP char         (floor)",    lambda v: sp_run('raw', km_test,  v, 'char',    False)[0]),
]

print(f"train={TRAIN_CHARS:,} chars   test={TEST_CHARS:,} chars (held out)")
print("characters of ORIGINAL text per token - higher is better\n")
print(f"{'method':<26}" + "".join(f"{'v='+str(v):>11}" for v in VOCABS))
print("-" * 60)

best = (None, 0, 0)
for name, fn in METHODS:
    cells = []
    for v in VOCABS:
        try:
            n = fn(v)
            cpt = len(km_test) / n
            cells.append(cpt)
            if cpt > best[1]:
                best = (name, cpt, v)
        except Exception as e:
            cells.append(None)
    print(f"{name:<26}" + "".join(f"{c:>11.3f}" if c else f"{'-':>11}" for c in cells))

print(f"\nBEST: {best[0].strip()} at v={best[2]} -> {best[1]:.3f} chars/token")
print(f"(bench_sota.py reference: XLM-R 2.895, NLLB 2.423, SEA-LION v4 1.933, GPT-4o 1.599)")

print("\n--- sample at v=8000 ---")
sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញជារាជធានី"
print(f"  {sample}  ({len(sample)} chars)")
n, sp = sp_run('raw', km_test, 8000, 'bpe', False)
print(f"  SP bpe raw       {len(sp.encode(sample)):>2} tok: {sp.encode(sample, out_type=str)}")
n, sp2 = sp_run('seg', seg_test, 8000, 'bpe', True)
segd = ' '.join(kcut(sample))
print(f"  SP bpe +khmercut {len(sp2.encode(segd)):>2} tok: {sp2.encode(segd, out_type=str)}")
