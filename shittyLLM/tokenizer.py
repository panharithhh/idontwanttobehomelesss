class BasicTokenizer:
    """
    Minimal Byte Pair Encoding (BPE) Tokenizer based on Andrej Karpathy's minbpe.
    Starts from 256 raw byte tokens and iteratively merges the most frequent pairs.
    """
    def __init__(self):
        self.merges = {}  # (p0, p1) -> new_token_id
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # id -> bytes

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        
        # Start with raw UTF-8 bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]!r}) had {stats[pair]} occurrences")

    def encode(self, text):
        # Convert text to initial byte tokens
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Find the pair with lowest merge index (highest priority)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # Nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab.get(idx, b"") for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
