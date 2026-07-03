def get_stats(ids : list) -> dict:

    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair]  = counts.get(pair, 0) + 1 
    return counts


def merge(ids, pair, idx) -> list:
    
    new_ids = []
    i = 0

    while i < len(ids):
            
        if i < len(ids) - 1 and pair[0] ==ids[i] and pair[1] ==ids[i+1]:
            new_ids.append(idx)  
            i+=2 

        else :
            new_ids.append(ids[i])
            i+=1 
    return new_ids
    

def get_stats(ids, counts = None): # the count shoudl be what 
        
    counts = {} if counts is None else counts 
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair,0) + 1 
    
    return counts

    



