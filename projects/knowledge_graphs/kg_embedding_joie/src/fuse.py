import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F 
from regularizer import get_regularizer 
from layers import FuzzyMapping, Projection
from transformers import BertModel
from KG import KG


"""
several components of a simple fuzzy set:
- measure space with learnable weights
- number of partitions
- In case of shallow embedding, the number of entities. 
"""


class SimpleFuzzySet(nn.Module):
    
    # requires info from KG1, KG2, KGA
    def __init__(self,
                 n_ents,
                 n_rels,
                 onto_rel_index,
                 args):
        super(SimpleFuzzySet, self).__init__()
        self.args = args
        # self.pre_train_model = self.__load_pre_trained__()
        self.dropout = nn.Dropout(self.args.dropout)
        self.device = args.device
        # load data
        self.n_entities = n_ents
        self.n_relations = n_rels # ontology is different relation
        self.scale = args.scale # scale up condition score 

        # constants
        self.entity_embed_dim = args.entity_embed_dim
        self.n_partitions = args.n_partitions
        self.regularizer = args.regularizer
        self.n_rel_basis = args.n_rel_basis
        self.onto_rel_index = onto_rel_index
        self.alpha = args.alpha
        self.beta = args.beta

        # fuzzy logic related settings
        self.use_volume_weights = args.use_volume_weights
        self.regularize_volume = args.regularize_volume
        self.score_type = args.score_type        
        # fuzzy logic related operators        
        # for the current taxonomy task, no need to include logical expressions
        self.entity_regularizer = get_regularizer(self.regularizer, self.entity_embed_dim)
        self.partition_regularizer = get_regularizer(self.regularizer, self.n_partitions)

        # shallow emebdding
        # TODO: BACKWARD COMPATIBILITY 
        # when comparing with existing KGE models like TransE, we may want to have relation embedding 
        self.entity_embedding = nn.Embedding(self.n_entities, self.entity_embed_dim)
        
        self.relationship_transform = Projection(self.n_relations, 
                                                 self.entity_embed_dim,
                                                 self.n_partitions, 
                                                 self.partition_regularizer, 
                                                 self.n_rel_basis)
        
        self.relationship_inverse_transform = Projection(self.n_relations, 
                                                 self.entity_embed_dim,
                                                 self.n_partitions, 
                                                 self.partition_regularizer, 
                                                 self.n_rel_basis)
        

        self.fuzzymap = FuzzyMapping(entity_dim=self.entity_embed_dim, 
                                     hidden_dim=args.hidden_dim,
                                     num_hidden_layers=args.num_hidden_layers,
                                     regularizer=self.entity_regularizer,
                                     n_partitions=self.n_partitions,
                                     modulelist=args.modulelist)
        
        self.partition_weights = nn.Parameter(torch.ones((self.n_partitions, )))
                
    # load language model
    def load_language_model(self):
        pass 
        # pre_trained_dic = {
        #     "bert": [BertModel,"bert-base-uncased"]
        # }

        # pre_train_model, checkpoint = pre_trained_dic[self.args.pre_train]
        # model = pre_train_model.from_pretrained(checkpoint)

        # return model


    #####################################################################
    ########## Scores related to Ontology (FUSE) ########################
    #####################################################################
    
    # unlike in the taxonomy case, here instead of "query" we call it "tail"
    # instead of "entity" we call it "head" 
    # this is how the ontology dataset is formulated 
    # later on, we can try data augmentation
    
    """
    pair with cosine weighted
    """
    def cal_pair_weighted_cosine_possibility(self, head_onto, tail_onto, mode):
        # experiments with the weighted cosine score 
        # weighted means normalize, unweighed means no-normalization
        # head_onto = F.normalize(head_onto, dim=-1) # (bnd)
        # tail_onto = F.normalize(tail_onto, dim=-1) # (bmd)

        # cosine without others
        if not self.use_volume_weights:
            if mode == "single":
                score = torch.einsum("bnd,bnd->bn", head_onto, tail_onto)
            elif mode == "head-batch":
                score = torch.einsum("bnd,bmd->bn", head_onto, tail_onto)
            else:
                score = torch.einsum("bnd,bmd->bm", head_onto, tail_onto)
            return score

        if self.regularize_volume:
            weighted_head =  head_onto * self.partition_regularizer(self.partition_weights)
            weighted_tail = tail_onto * self.partition_regularizer(self.partition_weights)
        else:  # global 
            weighted_head =  head_onto * self.partition_weights # bnd
            weighted_tail = tail_onto * self.partition_weights # bd 
        if mode == "single":
                score = torch.einsum("bnd,bnd->bn", weighted_head, weighted_tail)
        elif mode == "head-batch":
            score = torch.einsum("bnd,bmd->bn", weighted_head, weighted_tail)
        else:
            score = torch.einsum("bnd,bmd->bm", weighted_head, weighted_tail)

        return score
    
    # input is of shape (b,n,d)
    def cal_single_weighted_cosine_possibility(self, fuzzy_set):
        # without volume weight, this is just the fuzzy set itself 
        fuzzy_set = F.normalize(fuzzy_set, dim=-1)
        if not self.use_volume_weights:
            return torch.sum(fuzzy_set, dim=-1)
        if self.regularize_volume:
            score = torch.sum(fuzzy_set * self.partition_regularizer(self.partition_weights),dim=-1) 
        else:
            score = torch.sum(fuzzy_set * self.partition_weights, dim=-1) 
        # (bn)
        return score
    
        
    """
    possibility of a pair of fuzzy sets
    this is defined as intersection between head and tail
    this intersection in volume shuld be close to head
    shape of inputs (B,d) (B,d)
    """
    def cal_pair_fuzzy_possibility(self, head_onto, tail_onto, mode):

        if mode == "single":
            intersection_fuzzyset = torch.einsum("bnd,bnd->bnd", head_onto, tail_onto)
        elif mode == "head-batch":
            intersection_fuzzyset = torch.einsum("bnd,bmd->bnd", head_onto, tail_onto)
        else:
            intersection_fuzzyset = torch.einsum("bnd,bmd->bmd", head_onto, tail_onto)

        # without any volume weights 
        if not self.use_volume_weights:
            score = torch.sum(intersection_fuzzyset, dim=-1)
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
    

    # between head and tail
    # head is child, tail is parent
    def cal_pair_and_condition_score(self, head_onto, tail_onto, mode):
        if self.score_type == "possibility":
            pair_possibility = self.cal_pair_fuzzy_possibility(head_onto, tail_onto, mode)
            head_possibility = self.cal_single_fuzzy_possibility(head_onto)
        else:
            pair_possibility = self.cal_pair_weighted_cosine_possibility(head_onto, tail_onto, mode)
            head_possibility = self.cal_single_weighted_cosine_possibility(head_onto)
        condition_score = pair_possibility / head_possibility
        return head_possibility, pair_possibility, condition_score
     
    
    # project
    def project_fuzzyset(self, encode_inputs):
        cls = self.pre_train_model(**encode_inputs)
        cls = self.dropout(cls[0][:, 0, :]) # (B, 768)
        fuzzy_set = self.fuzzymap(cls) # (B, d)
        return fuzzy_set
    
    def project_fuzzy_shallow(self, entity_emb):
        return self.fuzzymap(entity_emb)


    #####################################################################
    ########## Main Functions for Training ##############################
    #####################################################################
        
    
    # helper to obtain encoding for kg and for ontology
    # for shallow embeddings
    # returns 
    def obtain_embedding_for_kg_and_ontology(self, heads, rels, tails, mode="single", negative_sample_size=None):

        # print(f"rels = {rels}")
        # print(f"onto_rel_index = {self.onto_rel_index}")
        all_ontology_index = (rels >= self.onto_rel_index).nonzero()
        all_not_ontology_index = (rels < self.onto_rel_index).nonzero()
        # print(all_ontology_index)
        # print(all_not_ontology_index)
        
        head_indices, rel_indices, tail_indices = heads[all_not_ontology_index], rels[all_not_ontology_index], tails[all_not_ontology_index]
        head_onto_indices, tail_onto_indices = heads[all_ontology_index], tails[all_ontology_index]
        head_indices = head_indices.to(self.device)
        rel_indices = rel_indices.to(self.device)
        tail_indices = tail_indices.to(self.device)
        
        # 10/05: handling the case where kg 
        head_kg, tail_kg = None, None
        if len(head_indices) != 0 and len(tail_indices)!= 0: 
            if mode == "head-batch":
                # print(f"head indices = {head_indices}")
                # (batch_size, negative_sample_size, dim)
                head_kg = self.entity_embedding(head_indices).view(len(head_indices), negative_sample_size, -1)
            else:            
                head_kg = self.entity_embedding(head_indices)        
        
            if mode == "tail-batch":
                tail_kg = self.entity_embedding(tail_indices).view(len(tail_indices), negative_sample_size, -1)
            else:
                # (batch_size, 1, dim)
                tail_kg = self.entity_embedding(tail_indices)
            
            tail_kg = self.project_fuzzy_shallow(tail_kg)
            head_kg = self.project_fuzzy_shallow(head_kg)
                
        # ontology triple embeddings
        head_onto, tail_onto = None, None
        if len(head_onto_indices) > 0:
            head_onto_indices = head_onto_indices.to(self.device)
            tail_onto_indices = tail_onto_indices.to(self.device)
            if mode == "head-batch":
                head_onto = self.entity_embedding(head_onto_indices).view(len(head_onto_indices), negative_sample_size, -1)
            else:
                head_onto = self.entity_embedding(head_onto_indices)        
            if mode == "tail-batch":
                tail_onto = self.entity_embedding(tail_onto_indices).view(len(tail_onto_indices), negative_sample_size, -1)
            else:
                tail_onto = self.entity_embedding(tail_onto_indices)
                
            tail_onto = self.project_fuzzy_shallow(tail_onto)
            head_onto = self.project_fuzzy_shallow(head_onto)
        
        return head_kg, rel_indices, tail_kg, head_onto, tail_onto

    
    
    # single mode, batch of triples 
    # head-batch: first part positive samples, second part negative samples (for head)
    # (pos_heads, neg_heads), rel, tails
    # tail-batch: heads, rel, (pos_tails, neg_tails)
    # encode triple from sample needs to take care of the relations that are ontology and those that are not ontology
    def encode_kg_triples_from_sample(self, sample, mode="single"):        
        if mode == 'single':
            negative_sample_size = 1
            all_rels = sample[:,1]
            all_heads = sample[:,0]
            all_tails = sample[:,2]
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            negative_sample_size = head_part.size(1)
            all_heads = head_part
            all_rels = tail_part[:,1]
            all_tails = tail_part[:, 2]
                        
        elif mode == 'tail-batch':
            head_part, tail_part = sample # head = positive head, tail = negative tails
            negative_sample_size = tail_part.size(1)
            all_heads = head_part[:,0]
            all_rels = head_part[:,1]
            all_tails = tail_part

        else:
            raise ValueError('mode %s not supported' % mode)

        
        head_kg, rel_kg, tail_kg, head_onto, tail_onto = self.obtain_embedding_for_kg_and_ontology(all_heads, all_rels,all_tails, mode, negative_sample_size)
        return head_kg, rel_kg, tail_kg, head_onto, tail_onto
    

    
    ############ KG based scores #################################################
    
    def compute_score_for_kg_triples(self, head_kg, rel_kg, tail_kg, mode="single"): 
        if mode == "head-batch":
            # head-batch contains negative heads and positive tail
            tail_kg = tail_kg.repeat(1, head_kg.shape[1], 1)
        if mode == "tail-batch":
            # tail-batch contains negative tails and positive head
            head_kg = head_kg.repeat(1, tail_kg.shape[1], 1)
        # if mode == "single":
        #     tail_kg = tail_kg.unsqueeze(1)
        #     head_kg = head_kg.unsqueeze(1)
        
        # L_triple
        if self.regularize_volume:
            head_to_tail_score = torch.abs(self.relationship_transform(head_kg, rel_kg)- tail_kg) @ self.partition_regularizer(self.partition_weights)
            # L_inverse
            tail_to_head_score = torch.abs(self.relationship_inverse_transform(tail_kg, rel_kg)-head_kg) @ self.partition_regularizer(self.partition_weights)

            # L_symmetry 
            tail_to_tail_score = torch.abs(self.relationship_transform(
                self.relationship_inverse_transform(tail_kg, rel_kg),rel_kg
                ) - tail_kg) @ self.partition_regularizer(self.partition_weights)
            
            head_to_head_score = torch.abs(self.relationship_inverse_transform(
                self.relationship_transform(head_kg, rel_kg), rel_kg
                ) - head_kg) @ self.partition_regularizer(self.partition_weights)
            
            score = (head_to_tail_score + tail_to_head_score + tail_to_tail_score + head_to_head_score) / 4      

        else:  
            head_to_tail_score = torch.abs(self.relationship_transform(head_kg, rel_kg)- tail_kg) @ self.partition_weights
            # L_inverse
            tail_to_head_score = torch.abs(self.relationship_inverse_transform(tail_kg, rel_kg) - head_kg) @ self.partition_weights
            # L_symmetry 
            tail_to_tail_score = torch.abs(self.relationship_transform(
                self.relationship_inverse_transform(tail_kg, rel_kg),rel_kg
                ) - tail_kg) @ self.partition_weights
            
            head_to_head_score = torch.abs(self.relationship_inverse_transform(
                self.relationship_transform(head_kg, rel_kg),rel_kg
                ) - head_kg) @ self.partition_weights
                        
            score = (head_to_tail_score + tail_to_head_score + tail_to_tail_score + head_to_head_score) / 4
        
        return score
                    


    ########### Ontology based scores ###############################################
    
    # ontology loss is the old FUSE lose   
    def compute_score_for_ontology_triples(self, head_onto, tail_onto, mode="single"):
        if mode == "head-batch":
            # head-batch contains negative heads and positive tail
            tail_onto = tail_onto.repeat(1, head_onto.shape[1], 1)
        if mode == "tail-batch":
            # tail-batch contains negative tails and positive head
            head_onto = head_onto.repeat(1, tail_onto.shape[1], 1)
        # the output should be a combination of the asymmetrical conditional score + the possibility
        # inclusion score is large if we have positive pairs
        _, inclusion_score, condition_score = self.cal_pair_and_condition_score(head_onto, tail_onto, mode)

        # print(f"inclusion_score mean = {torch.mean(torch.mean(inclusion_score))}")
        # print(f"condition_score mean = {torch.mean(torch.mean(condition_score))}")
        
        return inclusion_score + condition_score * self.scale
        
        
    ########### Forward #############################################################
    # return alpha * score_kg + beta * score_ontology 
    # for positive samples
    # modify the current logic to directly return loss function. 
    def forward(self, sample, mode="single"):
        # PART ONE: obtain the embedded triples (or encoded triples) from the dataset
        head_kg, rel_kg, tail_kg, head_onto, tail_onto = self.encode_kg_triples_from_sample(sample, mode)
        # PART TWO: calculate loss function using kg and using ontology
        # score_kg has shape (b,n)
        score = None
        if head_kg != None:
            score = self.compute_score_for_kg_triples(head_kg, rel_kg, tail_kg)
            score_kg = torch.mean(torch.mean(score, dim=-1), dim=-1)
            # print(f'forward pass mean score_kg = {score_kg}')
        score_onto = None
        if head_onto != None:
            # score_onto has shape (b,m)
            score_onto = self.compute_score_for_ontology_triples(head_onto, tail_onto)
            # print(f'forward pass score_onto = {torch.mean(torch.mean(score_onto))}')        
        if torch.is_tensor(score) and torch.is_tensor(score_onto):
           score = torch.cat((score, score_onto))
        if torch.is_tensor(score_onto) and not torch.is_tensor(score):
            score = score_onto
        # print(f"score = {score}")
        return score