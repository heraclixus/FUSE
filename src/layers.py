import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import get_regularizer




class MLP(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)


    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x



class MLP_VEC(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP_VEC,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)
        # self.fc2 = nn.Linear(hidden, hidden, bias=True)


    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        
        return x






class LINEAR(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super(LINEAR,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)


    def forward(self,x):

        x = self.fc1(x)
        
        return x
    

"""
simple fuzzy set related operators 
"""
"""
simple mLP for the projection layers
the goal is to return a fuzzy embedding (in fuzzyqe)
or i-th partition of a PL-Fuzzyset (our case)
"""
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, regularizer, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, output_dim)  # final projection
        
        for nl in range(2, self.num_hidden_layers + 2):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_hidden_layers+2):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
            
    # forward pass to replicate concatenation of e,r
    def forward(self, x):
        for nl in range(1, self.num_hidden_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.regularizer(x)
        return x  # (B,1)


"""
Entity mapping: takes entity embeddings and map them into a PL-Fuzzy set in [0,1]^d
"""
class FuzzyMapping(nn.Module):
    def __init__(self, entity_dim, hidden_dim, 
                 num_hidden_layers,
                 regularizer,
                 n_partitions, modulelist):
        super(FuzzyMapping, self).__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.n_partitions = n_partitions
        self.modulelist = modulelist
        
        # parallel einsum w.r.t partitions       
        self.pl_fuzzyset_maps = SimpleMLP(input_dim=self.entity_dim, hidden_dim=self.hidden_dim,
                                          num_hidden_layers=self.num_hidden_layers,
                                          regularizer=self.regularizer,
                                          output_dim=self.n_partitions) # n_partitions for output_dim 

    """
    e_embedding: embedding of shape (B,e)
    returns shape (B,d)
    """
    def forward(self, e_embedding):
        # (B,e)
        # print(f"forward: e_embedding = {e_embedding.shape}")
        if len(e_embedding.shape) == 2:
            pl_fuzzyset = []
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1)        
                inter_embedding = self.relu(torch.einsum("bde,deh->bdh",e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdh,dhl->bd", inter_embedding, self.mapping_weights2))
            else:
                return self.pl_fuzzyset_maps(e_embedding)
        else: # (B,n,e)
            pl_fuzzyset = []
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1, 1)
                inter_embedding = self.relu(torch.einsum("bdne,deh->bdnh", e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdnh,dhl->bnd", inter_embedding, self.mapping_weights2)).permute(1,0,2)
            else:
                return self.pl_fuzzyset_maps(e_embedding).permute(1,0,2)            