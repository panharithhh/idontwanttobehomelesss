from pathlib import Path
class BasicTokenizer:
    
    def get_stats(self, ids):
        counts = {} 
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair,0) + 1
        return counts 

    def merge(self, ids, pair, idx):
        # ids:  the current list of token integers (e.g. [1, 2, 3, 1, 2])
        # pair: the tuple pair we want to replace (e.g. (1, 2))
        # idx:  the new token ID to replace the pair with (e.g. 256)
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2 
            else:
                new_ids.append(ids[i])
                i +=1  
        return new_ids 
    
    def train(self,text,vocab_size, verbose = False):
        tokens = list(text.encode("utf-8"))

        new_tokens = vocab_size-256
        idx = 256
        
        self.merges = {} # for encode have from 256 to tokens_max
        
        ids = tokens 
        for i in range(new_tokens):
            stats = self.get_stats(ids)
            idx = 256 + i 
            top_pair = max(stats, key= stats.get)
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # ths is just looping all of the eng stuff
# use for decode it will load the 256 raw bytes stream from 0 -> tokens_max
        for (p0,p1),idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]    
         
    def encode(self, text):
        tokens = b"".join(text.encode("utf-8"))
        while len(tokens) >= 2 :
            stats = self.get_stats(tokens) 
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf"))) # take the lowest index inside the merge
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx) 
        return tokens
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace") 
        return text 
