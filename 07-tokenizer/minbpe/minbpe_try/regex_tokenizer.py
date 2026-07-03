import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

from os import replace
from helper import get_stats, merge

     
class RegexTokenizer:

    def __init__(self):

        self.merges = {}

        self.vocab =  {}
     

    def train(self, text, vocab_size, verbose = False):

        assert vocab_size >= 256

        num_merges = vocab_size - 256
        
        text_chunk = re.findall(GPT4_SPLIT_PATTERN, text)
        ids = (list(ch.encode("utf-8") for ch in  text_chunk))

        merges = {}
        vocab = {idx:bytes([idx]) for idx in range(256)} # idx btes

        for i in range(num_merges): 
            

            

            stats = get_stats(ids)
            top_pair = max(stats, key = stats.get)
            idx = 256 + i

            ids = merge(ids,top_pair,idx)
            
            #use in encode
            merges[top_pair] = idx
             
            #use in decode
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")

        self.merges = merges
        self.vocab  = vocab
        
           
    def decode(self, ids):

        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def encode(self, text):
        
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:

            stats = get_stats(text)

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # copy pasta
            # core idea we take 

            if pair not in self.merges:
                break
        
# GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
       

t = Tokenizer()

text = open("s.txt", 'r').read()

t.train(text, 270, True)

print(t.encode("hello"))
print(t.decode([104, 101, 108, 108, 111]))


