"""
Given an already trained model, try to query with logical operations
"""
import os
import numpy as np
from transformers import BertTokenizer
from model_fuzzy import SimpleFuzzySet
import pickle as pkl
import torch
import argparse
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()  

##data 
parser.add_argument('--dataset', type=str, default='environment', help='dataset') 
## Model parametersddd
parser.add_argument('--pre_train', type=str, default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64, help='dimension of hidden layers in MLP')
parser.add_argument('--embed_size', type=int, default=12, help='dimension of box embeddings')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--margin', type=float, default=0.05, help='margin for containing')
parser.add_argument('--epsilon', type=float, default=-0.03, help='margin for negative contain')
parser.add_argument('--size', type=float, default=0.03, help='minimum volumn of box')
parser.add_argument('--alpha', type=float, default=1.0, help='weight of contain loss')
parser.add_argument('--beta', type=float, default=1.0, help='weight of negative contain loss')
parser.add_argument('--gamma', type=float, default=1.0, help='weight of regularization loss')
parser.add_argument('--extra', type=float, default=0.1, help='weight of prob loss')

## Training hyper-parameters
parser.add_argument('--expID', type=int, default=0, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate for pre-trained model')
parser.add_argument('--lr_projection', type=float, default=1e-3, help='learning rate for projection layers')
parser.add_argument('--eps', type=float, default=1e-8, help='adamw_epsilon')
parser.add_argument('--optim', type=str, default="adamw", help='Optimizer')

## Others
parser.add_argument('--cuda', type=bool, default=True, help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=6, help='which gpu')

## configurations for Simple Fuzzy Set 
parser.add_argument("--use_fuzzyset", action="store_true") # flag used to decide whether to use SFS or box
parser.add_argument("--modulelist", action="store_true")
parser.add_argument("--n_partitions", default=100, type=int)
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension in simple MLP")
parser.add_argument("--entity_dim", type=int, default=768, help="dimension of entity (after BERT)")
parser.add_argument("--gamma_coeff", type=float, default=20)
parser.add_argument("--strength_beta", type=float, default=0, help="degree of asymmetry based on hamming loss")
parser.add_argument("--strength_alpha", type=float, default=0, help="degree of asymmetry based on possibility")
parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of hidden layers in simple MLP")
parser.add_argument("--regularizer_type", type=str, default="01", choices=["01","sigmoid","sigmoid_vec","softmax"])
parser.add_argument("--partition_reg_type", type=str, default="sigmoid", choices=["01","sigmoid","sigmoid_vec","softmax"])
parser.add_argument("--use_volume_weights", action="store_true")
parser.add_argument("--regularize_volume", action="store_true")
parser.add_argument("--regularize_entropy", action="store_true", help="determine whether weight regularization should be applied")
parser.add_argument("--regularize_intensity", default=1.0, type=float, help="strength of entropy reg")
parser.add_argument("--score_type", type=str, default="weighted_cos", choices=["weighted_cos", "possibility"])
parser.add_argument("--seed", type=int, default=0, help="help to identify repeated experiments and model lodaing")
parser.add_argument("--margin", type=float, default=0.0, help="gap between the positive and the negative, shrink")

# debug purpose, test only
parser.add_argument("--test_only", action="store_true", help="flag to determine if only evaluation is done")


args = parser.parse_args()
args.cuda = True if torch.cuda.is_available() and args.cuda else False


class Examine_Set_Relations(Dataset):
    
    def __init__(self, args=args):
        super(Examine_Set_Relations, self).__init__()
        

        self.args = args
        self.data = self.__load_data__(self.args.dataset)

        self.tokenizer = self.__load_tokenizer__()
        
        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]
        self.train_concept_set = self.data["train_concept_set"]
        self.train_parent_list = self.data["train_parent_list"]
        self.train_child_list = self.data["train_child_list"]
        self.train_negative_parent_dict = self.data["train_negative_parent_dict"]
        self.train_sibling_dict = self.data["train_sibling_dict"]
        self.child_parent_pair = self.data["child_parent_pair"]
        self.child_neg_parent_pair = self.data["child_neg_parent_pair"]
        self.child_sibling_pair = self.data["child_sibling_pair"]
        
        # generate dictionaries for the queries
        self.generate_dicts()
        
        
        self.model = SimpleFuzzySet(args)
        self.exp_setting = str(self.args.pre_train)+"_"+str(self.args.dataset)+"_"+str(self.args.expID)+"_"+str(self.args.epochs)\
            +"_"+str(self.args.embed_size)+"_"+str(self.args.batch_size)+"_"+str(self.args.margin)+"_"+str(self.args.epsilon)\
                +"_"+str(self.args.size)+"_"+str(self.args.alpha)+"_"+str(self.args.beta)+"_"+str(self.args.gamma)+"_"+str(self.args.extra)\
                    +"_"+str(self.args.n_partitions)+"_"+str(self.args.hidden_dim)+"_"+str(self.args.gamma_coeff)+"_"+str(self.args.strength_beta)\
                        +"_"+str(self.args.strength_alpha)+"_"+str(self.args.num_hidden_layers)+"_"+str(self.args.regularizer_type)+"_"+str(self.args.partition_reg_type)\
                            +"_"+f"volume_weight={self.args.use_volume_weights}_" + f"regularize_volume={self.args.regularize_volume}_"\
                                +f"regularize_entropy={self.args.regularize_entropy}_" + str(self.args.regularize_intensity) + "_" + str(self.args.score_type) + f"_{self.args.seed}"
        self.__load_model()
        

        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]
        print ("Training samples: {}".format(len(self.train_child_parent_negative_parent_triple)))

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        
        self.generate_embeddings()
        
        
    """
    load necesasry modules
    """

    def __load_tokenizer__(self):
        
        pre_trained_dic = {
            "bert": [BertTokenizer,"bert-base-uncased"]   
        }

        pre_train_tokenizer, checkpoint = pre_trained_dic[self.args.pre_train]
        tokenizer = pre_train_tokenizer.from_pretrained(checkpoint)

        return tokenizer


    def __load_data__(self,dataset):
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_0_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data


    def __load_model(self):
        self.model.load_state_dict(torch.load(os.path.join("../result",self.args.dataset,"model","fuzzy_model_"+self.exp_setting+".checkpoint")))

    
    def generate_dicts(self):

        self.union_pairs = {parent: [] for parent in self.train_parent_list}
        
        for (child, parent) in self.child_parent_pair:
            self.union_pairs[parent].append(child)
        
        with open("parent_child_pairs.pickle", "wb") as f:
            pkl.dump(self.union_pairs, f)

        
            
    def project(self, id):
        
        print(self.index_token_ids(self.encode_all, id))
        
        return self.model.project_fuzzyset(self.index_token_ids(self.encode_all, id))
    
    
    def generate_embeddings(self):
        self.all_embeddings = torch.stack([self.project(child) for child in range(429)])
        print(self.all_embeddings.shape)
        torch.save("embeddings.pt", self.all_embeddings)
    
    
            
    # def examine_union_relations(self):
    #     for parent in self.union_pairs.keys():
            
    #         parent_embedding = self.project(parent)
    #         children = self.union_pairs[parent]
    #         children_embedding = [self.project(child) for child in children]
            
    #         # union all children 
    #         union_embedding = None
    #         for child_embedding in child_embedding:
    #             if union_embedding == None:
    #                 union_embedding = child_embedding
    #             else:
    #                 union_embedding = (union_embedding + child_embedding - union_embedding * child_embedding) # product logic union
            
    #         # rank all the parents' embedding compared to union
                

    """ 
    bert related utilities
    """
    def generate_all_token_ids(self,tokenizer):

        all_nodes_context = [self.id_context[cid] for cid in self.concept_set]
        encode_all = tokenizer(all_nodes_context, padding=True,return_tensors='pt')
        
    
        a_input_ids = encode_all['input_ids'].cuda()
        a_token_type_ids = encode_all['token_type_ids'].cuda()
        a_attention_mask = encode_all['attention_mask'].cuda()

        encode_all = {'input_ids' : a_input_ids, 
                    'token_type_ids' : a_token_type_ids, 
                    'attention_mask' : a_attention_mask} 
        return encode_all




    def index_token_ids(self,encode_dic,index):

        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"],encode_dic["token_type_ids"],encode_dic["attention_mask"]
        
        res_dic = {'input_ids' : input_ids[index], 
                        'token_type_ids' : token_type_ids[index], 
                        'attention_mask' : attention_mask[index]}


        return res_dic


    def generate_parent_child_token_ids(self,index):

        child_id,parent_id,negative_parent_id = self.train_child_parent_negative_parent_triple[index]
        encode_child = self.index_token_ids(self.encode_all,child_id)
        encode_parent = self.index_token_ids(self.encode_all,parent_id)
        encode_negative_parents = self.index_token_ids(self.encode_all,negative_parent_id)

        return encode_parent, encode_child,encode_negative_parents


    def __getitem__(self, index):

        encode_parent, encode_child,encode_negative_parents = self.generate_parent_child_token_ids(index)

        return encode_parent, encode_child,encode_negative_parents


    def __len__(self):
        
        return len(self.train_child_parent_negative_parent_triple)
    
    


examiner = Examine_Set_Relations()