from types import LambdaType
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from torch.random import manual_seed 


words = open('names.txt','r').read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {c:i+1 for i,c in enumerate(chars)} 

stoi['.'] = 0

itos = {i:c for c,i in stoi.items()}

print(stoi)
print(itos)

N = torch.zeros((27,27), dtype=torch.int32)
N = N + 1 
# P which is the probabiliy and N would be the array 27 x 27 

for w in words:

    w = list('.') + list(w) + list('.')
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1][ix2] +=1 
        
#andrej karpathy code dim = 1 == axis 1 (stupid update) dim = 1 is the row 

P = N/N.sum(dim = 1, keepdim=True)
print(P[:1].sum(dim=1, keepdim=True))

loglikelyhood = 0
count = 0

for w in words[:1]: 
    
    w = list('.') + list(w) + list('.')
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1][ix2] +=1
        count +=1 
        prob = P[ix1][ix2]
        log_prob = torch.log(prob)
        loglikelyhood += -log_prob

# print(loglikelyhood/count)

### This is for Neural nets  

xs, ys = [],[]
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27),requires_grad=True, generator=g)

for w in words:
    w = list('.') + list(w) + list('.')
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2) 

print(xs)
print(ys)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

num = xs.nelement()

#Forward prop and backprop 
for _ in range(10): 
    #forward prop 

    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W     
    counts = logits.exp() # similar to counts 
    probs = counts/counts.sum(dim=1, keepdim=True) # probability 
    loss = probs(torch.arange(num), ys).log().mean()
    print(loss)
    # we can find the odds now 
    
    #  Backprop 
    W.grad = None
    loss.backward()
    
    W.data = W.data +(-(W.grad*50))


    
    




# plt.figure(figsize=(16, 16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()
#
        
# E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model? 

#E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see? 

# E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve? 
#
# E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W? 
#
# E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead? 
#
# E06: meta-exercise! Think of a fun/interesting exercise and complete it.
