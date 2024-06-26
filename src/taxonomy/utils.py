import numpy as np
import pytz
import torch.nn as nn
import torch
from datetime import datetime, timezone


def print_local_time():
    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('US/Pacific')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))


"""
Metrics used for Taxonomy Expansion
"""

def accuracy(pred,gt):
    pred = np.squeeze(pred[:,0])
    acc = np.sum(pred==gt)/len(gt)
    return acc


def mrr_score(pred,gt):
    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j]==gt[i]:
                mrr+=1/(j+1)
    mrr = mrr/len(gt)
    return mrr


def wu_p_score(pred, gt,path2root):

    pred = np.squeeze(pred[:,0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        shared_nodes = set(path_pred)&set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node]), lca_depth)
        wu_p+=2*lca_depth/(len(path_pred)+len(path_gt))
    
    wu_p = wu_p/len(gt)

    return wu_p
        

def metrics(pred, gt,path2root):

    acc = accuracy(pred,gt)
    mrr = mrr_score(pred,gt)
    wu_p = wu_p_score(pred, gt,path2root)
    return acc,mrr,wu_p


"""
Regularizer for Fuzzy Embedding Model 
"""
class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class SigmoidRegularizer(nn.Module):
    def __init__(self, vector_dim, dual=False):
        """
        :param dual: Split each embedding into 2 chunks.
                     The first chunk is property values and the second is property weight.
                     Do NOT sigmoid the second chunk.
        """
        super(SigmoidRegularizer, self).__init__()
        self.vector_dim = vector_dim
        # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
        self.weight = nn.Parameter(torch.Tensor([8]))
        self.bias = nn.Parameter(torch.Tensor([-4]))

        self.dual = dual

    def __call__(self, entity_embedding):
        if not self.dual:
            return torch.sigmoid(entity_embedding * self.weight + self.bias)
        else:
            # The first half is property values and the second is property weight.
            # Do NOT sigmoid the second chunk. The second chunk will be free parameters
            entity_vals, entity_val_weights = torch.chunk(entity_embedding, 2, dim=-1)
            entity_vals = torch.sigmoid(entity_vals * self.weight + self.bias)
            return torch.cat((entity_vals, entity_val_weights), dim=-1)


    def soft_discretize(self, entity_embedding, temperature=10):
        return torch.sigmoid((entity_embedding * self.weight + self.bias)*temperature)  # soft

    def hard_discretize(self, entity_embedding, temperature=10, thres=0.5):
        discrete = self.soft_discretize(entity_embedding, temperature)
        discrete[discrete>=thres] = 1
        discrete[discrete<thres] = 0
        return discrete


def get_regularizer(regularizer_type, entity_dim):
    if regularizer_type == '01':
        regularizer = Regularizer(base_add=0, min_val=0, max_val=1)
    elif regularizer_type == 'sigmoid_vec':
        regularizer = SigmoidRegularizer(entity_dim)
    elif regularizer_type == "sigmoid":
        regularizer = nn.Sigmoid()
    elif regularizer_type == "softmax":
        regularizer = nn.Softmax(dim=-1)
    return regularizer
