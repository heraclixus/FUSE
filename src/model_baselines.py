import os
import pickle as pkl
import torch.nn.functional as F 
import torch.nn as nn
from utils import * 
from layers import MLP
from transformers import BertModel


"""
use of code from github.com/statsl0217/beurre
as a baseline for the gumbel-box model
to avoid complication of the shallow embedding, adapt the MLP from BoxEmb 
and use different ways to calculate intersection (soft/gumbel)
"""

class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


class FuzzyBoxEmb(nn.Module):

    def __init__(self,args, tokenizer):

        super(FuzzyBoxEmb, self).__init__()

        self.args = args
        self.data = self.__load_data__(self.args.dataset)
        self.FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.train_taxo_dict = self.data["train_taxo_dict"]
        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]


        self.pre_train_model = self.__load_pre_trained__()

        # parameters for the volume-basic losses 
        self.strength_alpha = args.strength_alpha
        self.strength_beta = args.strength_beta
        self.gamma_coeff = args.gamma_coeff
        self.margin = args.margin

        # gumbel box specific 
        self.euler_gamma = 0.57721566490153286060
        self.gumbel_beta = 0.01 
        
        # MLPs for projetion into gumbel box parameterization
        self.projection_min = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size)
        self.projection_max = MLP(input_dim=768,hidden=self.args.hidden,output_dim=self.args.embed_size)
        self.dropout = nn.Dropout(self.args.dropout)


    def __load_data__(self,dataset):
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_"+str(self.args.expID)+"_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data


    def __load_pre_trained__(self):
        
        pre_trained_dic = {
            "bert": [BertModel,"bert-base-uncased"]
        }

        pre_train_model, checkpoint = pre_trained_dic[self.args.pre_train]
        model = pre_train_model.from_pretrained(checkpoint)

        return model
    
    
    """
    intersection 
    """
    def intersection(self, boxes1, boxes2):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )

        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box
        
        
    def log_volumes(self, boxes, temp=1., gumbel_beta=1., scale=1.):
        eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(boxes.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        return log_vol
    
    
    """
    similar to our own parent child possibility
    only now with different way to calculuate volume for gumbel box
    """
    
    def parent_child_possibility(self, child_box, parent_box, neg_parent_box):
        pos_pair_possibility = self.log_volumes(self.intersection(child_box, parent_box)) # intersection possibility
        child_possibility = self.log_volumes(child_box) # child possibility 
        neg_pair_possibility = self.log_volumes(self.intersection(child_box, neg_parent_box))
        return pos_pair_possibility, child_possibility, neg_pair_possibility
    

    """
    generate a box embedding from the input bert embedding
    """
    def projection_box(self,encode_inputs):
        cls = self.pre_train_model(**encode_inputs)
        cls = self.dropout(cls[0][:, 0, :])
        min = self.projection_min(cls)
        max = self.projection_max(cls)
        return Box(min_embed=min, max_embed=max)



    # between child and parent
    def condition_score(self, child_box, parent_box):
        pair_possibility = self.log_volumes(self.intersection(child_box, parent_box))
        child_possibility = self.log_volumes(child_box)
        # print(f"volume1 = {pair_possibility}, volume2 = {child_possibility}")
        
        if self.args.box_score_type == "ratio":
            condition_score = pair_possibility / child_possibility
        else: # diff 
            condition_score = pair_possibility - child_possibility
        return condition_score

    # condition score used in validation and prediction, takes input min and max tensors
    def condition_score_(self, child_min, child_max, parent_min, parent_max):
        child_box = Box(min_embed=child_min, max_embed=child_max)
        parent_box = Box(min_embed=parent_min, max_embed=parent_max)
        return self.condition_score(child_box, parent_box)
    
    # condition score calculation, but now instead constructing one box, create multiple boxes
    def condition_score1(self, child_min, child_max, parent_min, parent_max):
        n_boxes = len(child_min)
        condition_score = []
        for i in range(n_boxes):
            child_box = Box(child_min[i], child_max[i])
            parent_box  = Box(parent_min[i], parent_max[i])
            score = self.condition_score(child_box=child_box, parent_box=parent_box)
            condition_score.append(score)
        return torch.tensor(condition_score)

    # parent child possiblity loss
    def parent_child_possibility_loss(self, pos_pair_possibility, neg_pair_possibility):
        diff = -F.logsigmoid(self.gamma_coeff*(pos_pair_possibility - self.margin - neg_pair_possibility))           
        sample_loss = torch.mean(diff, dim=-1)
        return sample_loss
     
    def parent_child_asym_poss_loss_neg(self, neg_pair_possibility, child_possibility):
        condition_score = neg_pair_possibility / child_possibility
        asymmetry_loss = torch.mean(torch.square(condition_score))
        return asymmetry_loss
    
    def parent_child_asym_poss_loss_pos(self, pos_pair_possibility, child_possibility):
        condition_score = pos_pair_possibility / child_possibility
        asymmetry_loss = (torch.mean(torch.square(condition_score-1)))
        return asymmetry_loss

    def parent_child_pair_loss(self, child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset):
        pos_pair_possibility, child_possibility, neg_pair_possibility = self.parent_child_possibility(child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset)
        possibility_loss = self.parent_child_possibility_loss(pos_pair_possibility, neg_pair_possibility)
        asym_poss_loss_p = self.parent_child_asym_poss_loss_pos(pos_pair_possibility, child_possibility)
        asym_poss_loss_n = self.parent_child_asym_poss_loss_neg(neg_pair_possibility, child_possibility)
        asym_poss_loss = asym_poss_loss_p + asym_poss_loss_n
        total_loss = possibility_loss + self.strength_alpha * asym_poss_loss
        return total_loss, possibility_loss, asym_poss_loss
    

    def forward(self,encode_parent=None,encode_child=None,encode_negative_parents=None,flag="train"):
        # print(f"encode_parent = {encode_parent}")
        # print(f"encode_child = {encode_child}")
        parent_box = self.projection_box(encode_parent)
        child_box = self.projection_box(encode_child)
        neg_parent_box = self.projection_box(encode_negative_parents)
        return self.parent_child_pair_loss(child_box, parent_box, neg_parent_box)        
    
   
    