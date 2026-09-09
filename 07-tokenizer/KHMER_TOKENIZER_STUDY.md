# Khmer Tokenization: an Empirical Study

How badly do current tokenizers handle Khmer, and what should a Khmer tokenizer
actually be? Everything below is measured on a laptop and reproducible from this
directory.

**Headline:** a **4,000-token** Khmer-only tokenizer compresses Khmer better than
**every** production multilingual tokenizer tested, including ones with 250,000
tokens. No tested tokenizer allocates more than **0.79%** of its vocabulary to
Khmer script. GPT-4's `cl100k_base` contains exactly **one** such token.

---

## 1. Corpus

Khmer text is not ASCII, and the difference shows up immediately.

| | tiny Shakespeare | `khmer_corpus.txt` | `khmer_only.txt` |
|---|---|---|---|
| bytes | 1,115,394 | 3,688,921 | 3,166,810 |
| characters | 1,115,394 | 1,407,722 | 1,084,880 |
| unique codepoints | 65 | 884 | **95** |
| Khmer script | — | 78.5% | 93.2% |

`khmer_corpus.txt` is web-scraped and dirty: 545 of its 884 codepoints appear three
times or fewer (stray Korean, Chinese, Thai, Tamil, Devanagari). `make_khmer_only.py`
distils it in two stages — a **line** filter first (keep only lines already ≥90% Khmer,
so an English sentence is dropped whole rather than shredded), then a **character**
filter on the residue. Result: 95 codepoints, comparable to Shakespeare's 65, at 97%
of the character count.

**Same text, 3× the bytes.** Khmer codepoints are 3 UTF-8 bytes. This single fact
drives most of what follows.

## 2. Zero-width space is not a reliable word boundary

Khmer has no spaces between words; `U+200B` (ZWSP) is used to mark boundaries where
needed. The corpus contains **30,132** of them, which looked like free word-boundary
supervision.

It isn't. Density is wildly uneven — 1 per 36 characters corpus-wide, but only
**8 in a 20,000-character test slice**. Segmenting on ZWSP + whitespace yields
"words" averaging **25.6 characters**, the longest being 112:

```
'ដែលជាស្ថាប័នរដ្ឋបាលខ្ពស់បំផុត'   ← one "word", 29 characters
```

Those are clauses. **Consequence: `tokens/word` fertility is not computable on this
corpus**, and any result depending on ZWSP as a boundary signal is unsound here.
All fertility numbers below use `chars/token` instead.

## 3. Regex pre-split patterns (byte-level BPE)

`minbpe`, varying only the pre-split pattern. Held-out text, chars/token, higher better.

| pattern | v=512 | v=2048 | v=4096 |
|---|---|---|---|
| none (no pre-split) | 1.606 | 2.984 | **3.566** |
| Khmer word (ZWSP-delimited) | **1.630** | 2.937 | 3.437 |
| GPT-4 (`cl100k`) | 1.498 | 2.069 | 2.202 |
| Khmer orthographic syllable | 1.512 | 1.812 | 1.813 † |
| *English, GPT-4 pattern* | *2.103* | *3.243* | *3.502* |

† could not reach v=4096 — merges exhausted at **2057**. Chunks average 1.82
characters, so everything mergeable is merged before the vocabulary fills.

**The GPT patterns shred Khmer grapheme clusters.** `\p{L}` matches Khmer consonants
(`Lo`) but not combining marks (`Mn`/`Mc`), so vowel signs and the COENG joiner fall
into the punctuation branch and are torn from their bases:

```
text  : ប្រទេសកម្ពុជា។ ២០២៥ hello
khmer : ['ប្រ', 'ទេ', 'ស', 'ក', 'ម្ពុ', 'ជា', '។', ' ', '២០២៥', ' ', 'hello']
gpt2  : ['ប', '្', 'រទ', 'េ', 'សកម', '្', 'ព', 'ុ', 'ជ', 'ា។', ' ២០២៥', ' hello']
gpt4  : ['ប', '្រទ', 'េសកម', '្ព', 'ុជ', 'ាម', 'ានទ', 'ីក', '្រ', 'ុងភ']
```

GPT-2 emits `្` alone — a subscript joiner with nothing to join.

**There is a crossover.** Word-level pre-splitting wins at v=512 and loses above it:
constraining merges helps when vocabulary is scarce and becomes a ceiling when it
isn't. And the aggregate winner segments badly — `none` scores best while producing
`'ភ','្ន','ំព','េញ'`, whereas `Khmer word` produces `'ភ្នំ','ពេញ'` correctly and uses
*fewer* tokens on that sentence. **Compression and linguistic validity rank these
patterns differently.**

## 4. SentencePiece beats byte-level BPE

Byte BPE starts from 256 byte values, so ~2 merges are spent rebuilding every Khmer
*character* before subword learning begins. English gets characters free at 1 byte.
SentencePiece works on the character stream and never pays that tax.

| tokenizer | v=512 | v=1024 | v=2048 | v=3500 |
|---|---|---|---|---|
| SP **bpe** | 1.698 | 2.547 | 3.329 | **3.877** |
| SP unigram | 1.746 | 2.586 | 3.149 | 3.595 |
| byte-BPE (best pattern) | 1.606 | — | 2.984 | ~3.5 |

SentencePiece wins **with a smaller vocabulary than its nearest competitor**
(3,500 vs 4,096).

**BPE beat unigram, contrary to Thai practice** (WangchanBERTa, the standard Thai
model, uses unigram). The piece distribution explains it: unigram builds *longer*
pieces (mean 6.64 chars, 78.7% ≥4 chars) than BPE (4.48, 56.8%) but they generalize
worse to held-out text. At 120K training characters there isn't enough data for
likelihood pruning to find reusable long units. **Untested prediction: this flips
with more data.**

Two settings silently corrupt Khmer and must be set explicitly:

```python
spm.SentencePieceTrainer.train(
    model_type='bpe',
    normalization_rule_name='identity',  # default nmt_nfkc reorders Khmer marks
    split_by_whitespace=False,           # Khmer words aren't whitespace-delimited
    byte_fallback=True,
    character_coverage=1.0,
)
```

## 5. Versus production tokenizers

20,000 held-out Khmer characters. `Khmer %` = share of vocabulary containing Khmer script.

| tokenizer | vocab | tokens | chars/tok | Khmer pieces | Khmer % |
|---|---|---|---|---|---|
| **ours, SP-bpe** | **16,000** | **4,185** | **4.779** | 15,731 | 98.32% |
| ours, SP-bpe | 8,000 | 4,570 | 4.376 | 7,733 | 96.66% |
| ours, SP-bpe | 4,000 | 5,366 | 3.727 | 3,735 | 93.38% |
| XLM-RoBERTa | 250,002 | 6,908 | 2.895 | 1,965 | 0.79% |
| NLLB-200 | 256,204 | 8,255 | 2.423 | 1,598 | 0.62% |
| SEA-LION v4 | 262,145 | 10,349 | 1.933 | 599 | 0.23% |
| GPT-4o (`o200k_base`) | 200,019 | 12,511 | 1.599 | 323 | 0.16% |
| Qwen2.5 / Sailor2 | 151,665 | 24,404 | 0.820 | 38 | 0.03% |
| GPT-4 (`cl100k_base`) | 100,277 | 32,913 | 0.608 | **1** | 0.00% |

Four findings:

1. **GPT-4's tokenizer holds one Khmer-containing piece out of 100,277.** At 0.608
   chars/token it processes Khmer as raw bytes. A Khmer sentence costs a GPT-4 user
   **7.9× more tokens** than it costs a 16k Khmer-specific tokenizer.
2. **Sailor2's tokenizer is byte-identical to Qwen2.5's** — same 151,665 vocab, same
   24,404 tokens, same 38 Khmer pieces. A model marketed for Southeast Asia improved
   Khmer via *data upsampling* while leaving tokenization untouched.
3. **SEA-LION v4 (599 Khmer pieces) is worse for Khmer than XLM-R (1,965)**, a general
   multilingual model from 2019. SEA-LION adopted the Gemma 3 tokenizer citing lower
   SEA fertility; that improvement may hold averaged across its eleven SEA languages
   while not reaching Khmer specifically.
4. **No tokenizer spends more than 0.79% of its vocabulary on Khmer.**

## 6. Downstream: does it matter to a language model?

Identical GPT (4 layers, `n_embd=128`, `block_size=64`, 4,000 steps), trained twice on
`khmer_input.txt`. Only the tokenizer differs.

| tokenizer | vocab | val loss | **bits/char** | train |
|---|---|---|---|---|
| char-level | 95 | 2.2328 | 3.2213 | 97s |
| SP-BPE | 2,048 | 6.0198 | **2.9055** | 89s |

**Raw loss is a trap.** It says char-level wins by 3×. Bits/char — the only metric
comparable across vocabularies — says BPE wins by 10%:

```
bits/char = loss_nats × (tokens / characters) / ln(2)
```

BPE also buys context for free: the same 64-token window spans **64 characters**
char-level versus **191** with BPE, at identical compute.

### Samples

char-level:
```
បណ្ដង់បានបោះសម័យកាត់នៃភាពត្រការបានដំបូង កើតឡើងបានផ្លូវ ពីដោយបានផ្លួនប្រទេស។
ប្រទេសចិន នរាជ្យ អវិជា និងក្នុងឆ្នាំ២០១ ដែលសានីពាជានាធម្រាមដឹកអង្អំពីអ្នក
```

SP-BPE 2048 (· = zero-width space):
```
ប្រទេសជិតរ្ដ្រសង់ នៅអក្សរចសំពោះទាំងគំរាមណាស់លើ·ជាដាតីជា·គំនិតបុរេជាតំបន់·មាន·
ការ·ថា·បញ្ក្រល្ត្តា·ភ្·ច·ជា·ព្រះ ឬ·ដែរ·តៈ·វប្បធម៌·វត្ថុយូរ·គ្នា·ស្រោត·ចស់·
ទីក្រុង ព្រែក·ព្រ·។·ដែល·គ្នា·កំណួប·ជឿ·ទឿ·ប្រជា·ទីយ៉ាង·ព្រះបាទ·ក្នុង·ជនជាតិ
```

Both produce **real Khmer words** — ប្រទេស (country), ទីក្រុង (city), វប្បធម៌
(culture), ព្រះបាទ (king), ជនជាតិ (ethnicity), ឯករាជ្យ (independence) — from a
~1M-parameter model trained for 90 seconds.

**Honest note:** the char-level sample arguably reads *more fluently* despite worse
bits/char. The BPE model over-generates ZWSP (SentencePiece folded it into pieces) and
sees only 327K training tokens against char-level's 976K, so it does ~25 epochs to
char-level's 8. Bits/char measures compression, not fluency, and at this scale they
disagree. Do not overclaim from these samples.

## 7. Known limitations

- **Domain match favors us.** Our tokenizers train on the same corpus the test slice
  comes from (held out, but same source). Production tokenizers never saw it.
  Out-of-domain Khmer — news, social media, literature — is required before the
  comparison in §5 is robust.
- **Ours is Khmer-only** and would be dreadful on English. The honest claim is not
  "we beat GPT-4o" but "a small language-specific vocabulary vastly outperforms a
  large multilingual one on its language, and Khmer's current allocation is <1%."
- **`chars/token` is not comparable across scripts.** Khmer scored *higher* than
  English (3.877 vs 3.148), which does not mean Khmer is cheaper — a Khmer character
  is a denser unit. An honest cross-lingual number needs **parallel text**
  (FLORES-200, `khm_Khmr`).
- **Gemma 2 is license-gated** on HuggingFace and did not load; mT5 ships a
  SentencePiece `.model` rather than `tokenizer.json`.
- **The language models are tiny** (~1M parameters, 90 seconds). They demonstrate the
  pipeline, not model quality.

## 8. Reproduce

```bash
python make_khmer_only.py       # khmer_corpus.txt -> khmer_only.txt
python bench_patterns2.py       # regex pre-split patterns × vocab size   (§3)
python bench_sp2.py             # SentencePiece bpe vs unigram            (§4)
python bench_sota.py            # vs XLM-R, NLLB, SEA-LION, GPT-4/4o...   (§5)
python ../06-gpt/khmer_bpe_gpt.py   # char vs BPE language model          (§6)
```

Requires `sentencepiece`, `regex`, `tokenizers`, `tiktoken`, `huggingface_hub`, `torch`.

### Bugs fixed in `minbpe/` along the way
- `regex.py` had a `SyntaxError` (`return  = b"".joex`) — the package could not import.
- `KHMER_SPLIT_PATTERN` used Java/ICU syntax (`\x{17B6}`, `[a&&[^b]]`) and never
  compiled. Rewritten for Python's `regex`.
- The pattern classed Khmer **punctuation** (`U+17D4–17DC`: ។ ៕ ៗ ៛) as trailing
  diacritics. Now split as standalone chunks.
- `train()` crashed with `ValueError: max() arg is an empty sequence` when merges ran
  out. Now stops cleanly and reports the vocabulary actually reached — upstream
  `minbpe` has the same hole.

## 9. Next

1. **FLORES-200** (`khm_Khmr`) — parallel text is the only route to an honest
   cross-lingual fertility number, and it is also what parity-aware BPE requires.
2. **Parity-aware BPE** ([arXiv 2508.04796](https://arxiv.org/abs/2508.04796)) needs
   multiple corpora, one per language. On Khmer alone there is no parity term and it
   degenerates to standard BPE — the comparison is only meaningful over a *shared*
   multilingual vocabulary.
3. **Out-of-domain evaluation** to remove the domain-match advantage.
4. **Re-test unigram vs BPE at 1M+ training characters** to check the §4 prediction.
