"""
bench_sp2.py - SentencePiece vs byte-BPE on Khmer, with the vocab ceiling handled.

SentencePiece cannot mint more pieces than the training data supports; at 120k chars
that ceiling is ~3651 for English. Vocabs are capped per-config and the achieved size
is reported, so no cell silently compares a full vocab against a truncated one.
"""
import os, tempfile, sentencepiece as spm

TRAIN_CHARS, TEST_CHARS = 120_000, 20_000
VOCABS = [512, 1024, 2048, 3500]
ZWSP = '​'

full = open('khmer_only.txt', encoding='utf-8').read()
km_train, km_test = full[:TRAIN_CHARS], full[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]
eng = open('../06-gpt/input.txt', encoding='utf-8').read()
en_train, en_test = eng[:TRAIN_CHARS], eng[TRAIN_CHARS:TRAIN_CHARS+TEST_CHARS]

tmp = tempfile.mkdtemp()
paths = {}
for tag, txt in [('km', km_train), ('en', en_train)]:
    paths[tag] = os.path.join(tmp, f'{tag}.txt')
    open(paths[tag], 'w', encoding='utf-8').write(txt)

n_words = lambda t: len([w for w in t.replace(ZWSP, ' ').split() if w])


def run(tag, test, vocab, model_type, split_ws):
    prefix = os.path.join(tmp, f"{tag}_{model_type}_{vocab}_{split_ws}")
    spm.SentencePieceTrainer.train(
        input=paths[tag], model_prefix=prefix, vocab_size=vocab,
        model_type=model_type, character_coverage=1.0,
        split_by_whitespace=split_ws,
        normalization_rule_name='identity',
        remove_extra_whitespaces=False, add_dummy_prefix=False,
        byte_fallback=True, minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=prefix + '.model')
    ids = sp.encode(test)
    return len(test)/len(ids), len(ids)/n_words(test), sp.decode(ids) == test, sp


CONFIGS = [
    ("KHMER  SP unigram", 'km', km_test, 'unigram', False),
    ("KHMER  SP bpe",     'km', km_test, 'bpe',     False),
    ("ENG    SP unigram", 'en', en_test, 'unigram', True),
    ("ENG    SP bpe",     'en', en_test, 'bpe',     True),
]

print(f"train={TRAIN_CHARS:,} chars  test={TEST_CHARS:,} chars (held out)")
print("chars/token, higher is better\n")
print(f"{'tokenizer':<22}" + "".join(f"{'v='+str(v):>10}" for v in VOCABS) + "   rt")
print("-" * 68)

cpt_grid, fert_grid = {}, {}
for label, tag, test, mt, ws in CONFIGS:
    cells, ok_all = [], True
    for v in VOCABS:
        try:
            cpt, tpw, ok, _ = run(tag, test, v, mt, ws)
        except RuntimeError:
            cells.append(None); continue
        cpt_grid[(label, v)] = cpt; fert_grid[(label, v)] = tpw
        cells.append(cpt); ok_all &= ok
    print(f"{label:<22}" + "".join(f"{c:>10.3f}" if c else f"{'ceiling':>10}" for c in cells)
          + f"   {'ok' if ok_all else 'FAIL'}")

print("\n--- tokens per word (fertility, lower is better) ---")
print(f"{'tokenizer':<22}" + "".join(f"{'v='+str(v):>10}" for v in VOCABS))
for label, *_ in CONFIGS:
    print(f"{label:<22}" + "".join(
        f"{fert_grid.get((label,v),float('nan')):>10.3f}" for v in VOCABS))

print("\n--- Khmer token cost vs English, best config each, same vocab ---")
for v in VOCABS:
    km = max(cpt_grid.get(("KHMER  SP unigram", v), 0), cpt_grid.get(("KHMER  SP bpe", v), 0))
    en = max(cpt_grid.get(("ENG    SP unigram", v), 0), cpt_grid.get(("ENG    SP bpe", v), 0))
    if km and en:
        print(f"  v={v:<6} Khmer {km:.3f} ch/tok   English {en:.3f} ch/tok   "
              f"Khmer costs {en/km:.2f}x more tokens per character")

print("\n--- why unigram loses to bpe here: piece length distribution (v=2048) ---")
for mt in ['unigram', 'bpe']:
    _, _, _, sp = run('km', km_test, 2048, mt, False)
    lens = [len(sp.id_to_piece(i)) for i in range(sp.get_piece_size())]
    long = sum(1 for l in lens if l >= 4)
    print(f"  {mt:<8} mean piece {sum(lens)/len(lens):.2f} chars, "
          f"{long} pieces of >=4 chars ({long/len(lens)*100:.1f}%)")

sample = "ប្រទេសកម្ពុជាមានទីក្រុងភ្នំពេញ"
print(f"\n--- sample ({len(sample)} chars) at v=2048 ---")
for mt in ['unigram', 'bpe']:
    _, _, _, sp = run('km', km_test, 2048, mt, False)
    print(f"  {mt:<8} {len(sp.encode(sample)):>2} tokens: {sp.encode(sample, out_type=str)}")
