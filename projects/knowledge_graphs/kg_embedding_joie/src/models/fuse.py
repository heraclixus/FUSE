import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F 
from regularizer import get_regularizer 
from layers import FuzzyMapping
from transformers import BertModel


"""
several components of a simple fuzzy set:
- measure space with learnable weights
- number of partitions
- In case of shallow embedding, the number of entities. 
"""


class SimpleFuzzySet(nn.Module):
    
    def __init__(self,args):
        super(SimpleFuzzySet, self).__init__()
        
        # same as the box embed for dataset level information
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
        self.dropout = nn.Dropout(self.args.dropout)

        # fuzzy logic related settings
        self.use_volume_weights = args.use_volume_weights
        self.regularize_volume = args.regularize_volume
        self.regularize_entropy = args.regularize_entropy
        self.score_type = args.score_type        
        self.strength_alpha = args.strength_alpha
        self.strength_beta = args.strength_beta
        self.gamma_coeff = args.gamma_coeff
        self.n_partitions = args.n_partitions
        self.margin = args.margin
        # fuzzy logic related operators        
        # for the current taxonomy task, no need to include logical expressions
        self.entity_regularizer = get_regularizer(args.regularizer_type, args.entity_dim)
        self.partition_regularizer = get_regularizer(args.partition_reg_type, self.n_partitions)

        self.fuzzymap = FuzzyMapping(entity_dim=args.entity_dim, hidden_dim=args.hidden_dim,
                                     num_hidden_layers=args.num_hidden_layers,
                                     regularizer=self.entity_regularizer,
                                     n_partitions=args.n_partitions,
                                     modulelist=args.modulelist)
        
        self.partition_weights = nn.Parameter(torch.ones((self.n_partitions, )))
        
    # loading data         
    def load_data(self,dataset):
        pass

    # load language model
    def load_language_model(self):
        pass 
        # pre_trained_dic = {
        #     "bert": [BertModel,"bert-base-uncased"]
        # }

        # pre_train_model, checkpoint = pre_trained_dic[self.args.pre_train]
        # model = pre_train_model.from_pretrained(checkpoint)

        # return model
    
    
    """
    pair with cosine weighted 
    """
    def cal_pair_weighted_cosine_possibility(self, entity_pl_fuzzyset, query_pl_fuzzyset):
        # experiments with the weighted cosine score 
        entity_pl_fuzzyset = F.normalize(entity_pl_fuzzyset, dim=-1) # (bnd)
        query_pl_fuzzyset = F.normalize(query_pl_fuzzyset, dim=-1) # (bd)

        # cosine without others
        if not self.use_volume_weights:
            score = torch.einsum("bd,bd->b", entity_pl_fuzzyset, query_pl_fuzzyset)
            return score

        if self.regularize_volume:
            weighted_entity =  entity_pl_fuzzyset * self.partition_regularizer(self.partition_weights)
            weighted_query = query_pl_fuzzyset * self.partition_regularizer(self.partition_weights)
        else:  # global 
            weighted_entity =  entity_pl_fuzzyset * self.partition_weights # bnd
            weighted_query = query_pl_fuzzyset * self.partition_weights # bd 
        score = torch.einsum("bd,bd->b", weighted_entity, weighted_query)
        return score
    
    def cal_single_weighted_cosine_possibility(self, fuzzy_set):
        # without volume weight, this is just the fuzzy set itself 
        fuzzy_set = F.normalize(fuzzy_set, dim=-1)
        if not self.use_volume_weights:
            return torch.sum(fuzzy_set, dim=-1)
        if self.regularize_volume:
            score = torch.sum(fuzzy_set * self.partition_regularizer(self.partition_weights),dim=-1) 
        else:
            score = torch.sum(fuzzy_set * self.partition_weights, dim=-1) 
        return score    
    
        
    """
    possibility of a pair of fuzzy sets
    shape of inputs (B,d) (B,d)
    """
    def cal_pair_fuzzy_possibility(self, entity_pl_fuzzyset, query_pl_fuzzyset):
        intersection_fuzzyset = torch.einsum("bd,bd->bd", entity_pl_fuzzyset, query_pl_fuzzyset)

        # without any volume weights 
        if not self.use_volume_weights:
            score = torch.sum(intersection_fuzzyset, dim=-1) # bn
            return score
        # possibility score 
        if self.regularize_volume: # regularize volume 
            score = torch.sum(intersection_fuzzyset * self.partition_regularizer(self.partition_weights), dim=-1) # (B,n)
        else:
            score = torch.sum(intersection_fuzzyset * self.partition_weights, dim=-1) # (B,n)
        return score
    
    
    """
    possibility of a single fuzzy set
    """
    def cal_single_fuzzy_possibility(self, fuzzy_set):
        # without volume weight, this is just the fuzzy set itself 
        if not self.use_volume_weights:
            return torch.sum(fuzzy_set, dim=-1)
        if self.regularize_volume:
            score = torch.sum(fuzzy_set * self.partition_regularizer(self.partition_weights),dim=-1)
        else:
            score = torch.sum(fuzzy_set * self.partition_weights, dim=-1)
        return score
    

    # two options, weighted cosine vs. possibility
    def parent_child_possibility(self, child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset):
        if self.score_type == "possibility":
            pos_pair_possibility = self.cal_pair_fuzzy_possibility(child_fuzzyset, parent_fuzzyset) # intersection possibility
            child_possibility = self.cal_single_fuzzy_possibility(child_fuzzyset) # child possibility 
            neg_pair_possibility = self.cal_pair_fuzzy_possibility(child_fuzzyset, neg_parent_fuzzyset)
        else:
            pos_pair_possibility = self.cal_pair_weighted_cosine_possibility(child_fuzzyset, parent_fuzzyset)
            child_possibility = self.cal_single_weighted_cosine_possibility(child_fuzzyset)
            neg_pair_possibility = self.cal_pair_weighted_cosine_possibility(child_fuzzyset, neg_parent_fuzzyset)
        
        # print(f"pos_pair = {pos_pair_possibility.shape}, child = {child_possibility.shape}, neg_pair = {neg_pair_possibility.shape}")
        
        return pos_pair_possibility, child_possibility, neg_pair_possibility
    
    
    # between child and parent
    def condition_score(self, child_fuzzyset, parent_fuzzyset):
        if self.score_type == "possibility":
            pair_possibility = self.cal_pair_fuzzy_possibility(child_fuzzyset, parent_fuzzyset)
            child_possibility = self.cal_single_fuzzy_possibility(child_fuzzyset)
        else:
            pair_possibility = self.cal_pair_weighted_cosine_possibility(child_fuzzyset, parent_fuzzyset)
            child_possibility = self.cal_single_weighted_cosine_possibility(child_fuzzyset)
        condition_score = pair_possibility / child_possibility
        return condition_score
    

    # this loss is a combination of pair possibility + asymmetry (possibility)
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
     
    
    # possibility + asymmetry (hamming distance)
    def parent_child_pair_loss(self, child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset):
        pos_pair_possibility, child_possibility, neg_pair_possibility = self.parent_child_possibility(child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset)
        possibility_loss = self.parent_child_possibility_loss(pos_pair_possibility, neg_pair_possibility)
        asym_poss_loss_p = self.parent_child_asym_poss_loss_pos(pos_pair_possibility, child_possibility)
        asym_poss_loss_n = self.parent_child_asym_poss_loss_neg(neg_pair_possibility, child_possibility)
        asym_poss_loss = asym_poss_loss_p + asym_poss_loss_n
        total_loss = possibility_loss + self.strength_alpha * asym_poss_loss
        return total_loss, possibility_loss, asym_poss_loss
    
    # project
    def project_fuzzyset(self, encode_inputs):
        cls = self.pre_train_model(**encode_inputs)
        cls = self.dropout(cls[0][:, 0, :]) # (B, 768)
        fuzzy_set = self.fuzzymap(cls) # (B, d)
        return fuzzy_set
    
    
    def forward(self,encode_parent=None,encode_child=None,encode_negative_parents=None,flag="train"):
        parent_fuzzyset = self.project_fuzzyset(encode_parent)
        child_fuzzyset = self.project_fuzzyset(encode_child)
        neg_parent_fuzzyset = self.project_fuzzyset(encode_negative_parents)
        return self.parent_child_pair_loss(child_fuzzyset, parent_fuzzyset, neg_parent_fuzzyset)