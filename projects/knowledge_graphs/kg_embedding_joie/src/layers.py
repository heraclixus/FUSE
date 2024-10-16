import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_regularizer

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
    def __init__(self, 
                 entity_dim, 
                 hidden_dim, 
                 num_hidden_layers,
                 regularizer,
                 n_partitions, 
                 modulelist):

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
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1)        
                inter_embedding = self.relu(torch.einsum("bde,deh->bdh",e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdh,dhl->bd", inter_embedding, self.mapping_weights2))
            else:
                return self.pl_fuzzyset_maps(e_embedding)
        else: # (B,n,e)
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1, 1)
                inter_embedding = self.relu(torch.einsum("bdne,deh->bdnh", e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdnh,dhl->bnd", inter_embedding, self.mapping_weights2))
            else:
                return self.pl_fuzzyset_maps(e_embedding)           



# relationship projection
class Projection(nn.Module):
    def __init__(
            self,
            nrelation,
            entity_dim,
            n_partitions,
            regularizer_setting,
            num_rel_base,  # for 'rtransform'
    ):
        super(Projection, self).__init__()

        # # temporary testing
        regularizer_setting = {
                'type': 'sigmoid',
                "dual": False 
            }

        self.regularizer = get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True)
        # for projection
        self.entity_dim = entity_dim
        self.n_partitions = n_partitions

        self.dual = regularizer_setting['dual']

        n_base = num_rel_base
        if not self.dual:
            self.hidden_dim = self.n_partitions
            self.rel_base = nn.Parameter(torch.zeros(n_base, self.hidden_dim, self.hidden_dim))
            self.rel_bias = nn.Parameter(torch.zeros(n_base, self.hidden_dim))
            self.rel_att = nn.Parameter(torch.zeros(nrelation, n_base))
            self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            # new initialization
            torch.nn.init.orthogonal_(self.rel_base)
            torch.nn.init.xavier_normal_(self.rel_bias)
            torch.nn.init.xavier_normal_(self.rel_att)

        else:
            self.hidden_dim = self.n_partitions //2
            # for property vals
            self.rel_base1 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
            self.rel_bias1 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_att1 = nn.Parameter(torch.randn(nrelation, n_base))
            self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            # new initialization
            torch.nn.init.orthogonal_(self.rel_base1)
            torch.nn.init.xavier_normal_(self.rel_bias1)
            torch.nn.init.xavier_normal_(self.rel_att1)
            # for property weights
            self.rel_base2 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
            nn.init.xavier_uniform_(self.rel_base2, a=0, b=1e-2)
            self.rel_bias2 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_att2 = nn.Parameter(torch.randn(nrelation, n_base))
            self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            # new initialization
            torch.nn.init.orthogonal_(self.rel_base2)
            torch.nn.init.xavier_normal_(self.rel_bias2)
            torch.nn.init.xavier_normal_(self.rel_att2)
      

    # e_embedding has shape of (b,1,d) or (b,n,d)
    # rid is (b, 1)
    def forward(self, e_embedding, rid):
        rid = rid.squeeze()
        if not self.dual:
            project_r = torch.einsum('br,rio->bio', self.rel_att[rid], self.rel_base)            
            if self.rel_bias.shape[0] == self.rel_base.shape[0]:
                bias = torch.einsum('br,ri->bi', self.rel_att[rid], self.rel_bias)
            else:
                bias = self.rel_bias[rid]
            bias = bias.unsqueeze(1).repeat(1, e_embedding.shape[1], 1)
            output = torch.einsum('bio,bni->bno', project_r, e_embedding) + bias # (b, n, d)
            output = self.norm(output)
        else:
            e_embedding1, e_embedding2 = torch.chunk(e_embedding, 2, dim=-1)
            project_r1 = torch.einsum('br,rio->bio', self.rel_att1[rid], self.rel_base1)
            bias1 = self.rel_bias1[rid]
            output1 = torch.einsum('bio,bi->bo', project_r1, e_embedding1) + bias1
            output1 = self.norm1(output1)

            project_r2 = torch.einsum('br,rio->bio', self.rel_att2[rid], self.rel_base2)
            bias2 = self.rel_bias2[rid]
            output2 = torch.einsum('bio,bi->bo', project_r2, e_embedding2) + bias2
            output2 = self.norm2(output2)
            output = torch.cat((output1, output2), dim=-1)
        
        output = self.regularizer(output)
        return output

# unit tests
if __name__ == "__main__": 
    dummy_emb = torch.rand((32, 10, 5))
    rids = torch.randint(1, 99, (32,1))

    projection_layer = Projection(nrelation=100, entity_dim=5, n_partitions=5, 
                                  regularizer_setting="sigmoid", num_rel_base=25)
    dummy_rel_proj = projection_layer(dummy_emb, rids)
    print(dummy_emb)