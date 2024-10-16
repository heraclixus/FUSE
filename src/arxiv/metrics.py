import torch 
import numpy as np



"""
fuzzy jaccard index based loss 
|union| / |intersection| 
emb1: torch.Tensor 
emb2: torch.Tensor 
returns: scalar loss 
"""
def fuzzy_jaccard_index(emb1, emb2):
    union = emb1 + emb2 - emb1 * emb2 
    intersection = emb1 * emb2 
    jaccard_index = torch.norm(intersection) / torch.norm(union)
    return jaccard_index