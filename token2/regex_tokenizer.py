import regex as re

from lol import BasicTokenizer

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicTokenizer):

    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        ids = [list(chunk.encode("utf-8")) for chunk in self.compiled_pattern.findall(text)]

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                for pair, count in self.get_stats(chunk_ids).items():
                    stats[pair] = stats.get(pair, 0) + count

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self.merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids
