import os
import time
import numpy as np
import pickle as pkl
import torch
from torch import optim
from transformers import BertTokenizer
from utils import metrics
from data import Taxo_Data_Train, Taxo_Data_Test, load_data
from model_fuzzy import SimpleFuzzySet, FuzzQE_Taxo
from model_baselines import FuzzyBoxEmb, BoxEmbed


class Experiments(object):

    def __init__(self,args):
        super(Experiments,self).__init__()
        
        self.args = args
        self.tokenizer = self.__load_tokenizer__()
        self.train_loader,self.train_set = load_data(self.args, self.tokenizer,"train")
        self.test_loader,self.test_set = load_data(self.args, self.tokenizer,"test")

        self.use_fuzzyset = args.use_fuzzyset

        if args.use_fuzzyset and args.use_fuzzqe:
            self.model = FuzzQE_Taxo(args) # fuzzqe 
        elif args.use_fuzzyset: 
            self.model = SimpleFuzzySet(args)
        elif args.run_gumbel_box or args.run_soft_box:
            self.model = FuzzyBoxEmb(args, self.tokenizer)
        else:
            self.model = BoxEmbed(args,self.tokenizer)
        
        self.optimizer_pretrain, self.optimizer_projection = self._select_optimizer()
        self._set_device()
        # added configs for fuzzy set 
        self.exp_setting= str(self.args.pre_train)+"_"+str(self.args.dataset)+"_"+str(self.args.expID)+"_"+str(self.args.epochs)\
            +"_"+str(self.args.embed_size)+"_"+str(self.args.batch_size)+"_"+str(self.args.margin)+"_"+str(self.args.epsilon)\
                +"_"+str(self.args.size)+"_"+str(self.args.alpha)+"_"+str(self.args.beta)+"_"+str(self.args.gamma)+"_"+str(self.args.extra)\
                    +"_"+str(self.args.n_partitions)+"_"+str(self.args.hidden_dim)+"_"+str(self.args.gamma_coeff)+"_"+str(self.args.strength_beta)\
                        +"_"+str(self.args.strength_alpha)+"_"+str(self.args.num_hidden_layers)+"_"+str(self.args.regularizer_type)+"_"+str(self.args.partition_reg_type)\
                            +"_"+f"volume_weight={self.args.use_volume_weights}_" + f"regularize_volume={self.args.regularize_volume}_"\
                                +f"regularize_entropy={self.args.regularize_entropy}_" + str(self.args.regularize_intensity) + "_" + str(self.args.score_type) + f"_{self.args.seed}"
        
        setting={
            "pre_train":self.args.pre_train,
            "dataset":self.args.dataset,
            "expID":self.args.expID,
            "epochs":self.args.epochs,
            "embed_size":self.args.embed_size,
            "batch_size":self.args.batch_size,
            "margin":self.args.margin,
            "epsilon":self.args.epsilon,
            "size":self.args.size,
            "alpha":self.args.alpha,
            "beta":self.args.beta,
            "gamma":self.args.gamma,
            "extra":self.args.extra}
        print (setting)
        self.tosave_box={}
        self.tosave_pred={}


    def __load_tokenizer__(self):
        
        pre_trained_dic = {
            "bert": [BertTokenizer,"bert-base-uncased"]   
        }

        pre_train_tokenizer, checkpoint = pre_trained_dic[self.args.pre_train]
        tokenizer = pre_train_tokenizer.from_pretrained(checkpoint)

        return tokenizer



    def _select_optimizer(self):
        
        pre_train_parameters = [{"params": [p for n, p in self.model.named_parameters() if n.startswith("pre_train")],
                "weight_decay": 0.0},]
        projection_parameters = [{"params": [p for n, p in self.model.named_parameters() if n.startswith("projection")],
                "weight_decay": 0.0},]

        if self.args.optim=="adam":
            optimizer_pretrain = optim.Adam(pre_train_parameters, lr=self.args.lr)
            optimizer_projection = optim.Adam(projection_parameters, lr=self.args.lr_projection)
        elif self.args.optim=="adamw":
            optimizer_pretrain = optim.AdamW(pre_train_parameters,lr=self.args.lr, eps=self.args.eps)
            optimizer_projection = optim.AdamW(projection_parameters,lr=self.args.lr_projection, eps=self.args.eps)

        return optimizer_pretrain,optimizer_projection

    
    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()




    def train_one_step(self,it,encode_parent, encode_child,encode_negative_parents):
        
        """
        what does the encode input look like?
        """
        # print(f"encode_parent = {encode_parent}")
        # print(f"encode_child = {encode_child}")
        # print(f"encode_negative_parent = {encode_negative_parents}")

        self.model.train()
        self.optimizer_pretrain.zero_grad()
        self.optimizer_projection.zero_grad()
        
        if self.args.run_gumbel_box or self.args.run_soft_box:
            loss, possibility_loss, asym_poss_loss = self.model(encode_parent, encode_child,encode_negative_parents)
        elif not self.use_fuzzyset:
            loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob = self.model(encode_parent, encode_child,encode_negative_parents)
        else:
            loss, possibility_loss, asym_poss_loss = self.model(encode_parent, encode_child,encode_negative_parents)
        
        loss.backward()
        self.optimizer_pretrain.step()
        self.optimizer_projection.step()

        if self.args.run_gumbel_box or self.args.run_soft_box:
            return loss, possibility_loss, asym_poss_loss
        elif not self.use_fuzzyset:
            return loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob
        else:
            return loss, possibility_loss, asym_poss_loss

    def train(self):
        
        time_tracker = []
        test_acc = test_mrr = test_wu_p = 0
        for epoch in range(self.args.epochs):
            epoch_time = time.time()

            train_loss = []
            train_contain_loss = []
            train_negative_loss = []
            train_regular_loss = []
            train_pos_prob_loss = []
            train_neg_prob_loss = []
            
            # fuzzy set related losses
            fuzzy_train_loss = []
            gumbel_train_loss = []
            train_possibility_loss = []
            gumbel_train_possibility_loss = []
            train_asym_poss_loss = []
            gumbel_train_asym_poss_loss = []

            for i, (encode_parent,encode_child,encode_negative_parents) in enumerate(self.train_loader):
                            
                if self.args.use_fuzzyset: 
                    loss, possibility_loss, asym_poss_loss = self.train_one_step(it=i,encode_parent=encode_parent,encode_child=encode_child,encode_negative_parents=encode_negative_parents)
                    fuzzy_train_loss.append(loss.item())
                    train_possibility_loss.append(possibility_loss.item())
                    train_asym_poss_loss.append(asym_poss_loss.item())
                
                elif self.args.run_gumbel_box or self.args.run_soft_box:
                    loss, possibility_loss, asym_poss_loss = self.train_one_step(it=i,encode_parent=encode_parent,encode_child=encode_child,encode_negative_parents=encode_negative_parents)
                    gumbel_train_loss.append(loss.item())
                    gumbel_train_possibility_loss.append(possibility_loss.item())
                    gumbel_train_asym_poss_loss.append(asym_poss_loss.item())
                else:
                    loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob = self.train_one_step(it=i,encode_parent=encode_parent,encode_child=encode_child,encode_negative_parents=encode_negative_parents)
                    train_loss.append(loss.item())
                    train_contain_loss.append(loss_contain.item())
                    train_negative_loss.append(loss_negative.item())
                    train_regular_loss.append(regular_loss.item())
                    train_pos_prob_loss.append(loss_pos_prob.item())
                    train_neg_prob_loss.append(loss_neg_prob.item())
        
            if self.args.run_gumbel_box or self.args.run_soft_box:
                gumbel_train_loss = np.average(gumbel_train_loss)
                gumbel_train_possibility_loss = np.average(gumbel_train_possibility_loss)
                gumbel_train_asym_poss_loss = np.average(gumbel_train_asym_poss_loss)
                print(f"fuzzy box total loss = {gumbel_train_loss}, gumbel train_possibility_loss = {gumbel_train_possibility_loss}, gumbel training_asym_poss_loss = {gumbel_train_asym_poss_loss}")
            
            elif not self.args.use_fuzzyset: 
                train_loss = np.average(train_loss)
                train_contain_loss = np.average(train_contain_loss)
                train_negative_loss = np.average(train_negative_loss)
                train_regular_loss = np.average(train_regular_loss)
                train_pos_prob_loss = np.average(train_contain_loss)
                train_neg_prob_loss = np.average(train_neg_prob_loss)
            else:
                fuzzy_train_loss = np.average(fuzzy_train_loss)
                train_possibility_loss = np.average(train_possibility_loss)
                train_asym_poss_loss = np.average(train_asym_poss_loss)
                print(f"fuzzy total loss = {fuzzy_train_loss}, train_possibility_loss = {train_possibility_loss}, training_asym_poss_loss = {train_asym_poss_loss}")

            if self.args.run_gumbel_box or self.args.run_soft_box:
                test_acc,test_mrr,test_wu_p  = self.validation_fuzzy_box()
            elif self.use_fuzzyset:
                test_acc,test_mrr,test_wu_p  = self.validation_fuzzy()
            else:
                test_acc,test_mrr,test_wu_p  = self.validation(flag="all")
                            
            time_tracker.append(time.time()-epoch_time)
            
            if self.args.run_gumbel_box or self.args.run_soft_box:
                reported_loss = gumbel_train_loss
            elif self.use_fuzzyset:
                reported_loss = fuzzy_train_loss
            else:
                reported_loss = train_loss 
 
            print('Epoch: {:04d}'.format(epoch + 1),
                ' train_loss:{:.05f}'.format(reported_loss),
                'acc:{:.05f}'.format(test_acc),
                'mrr:{:.05f}'.format(test_mrr),
                'wu_p:{:.05f}'.format(test_wu_p),
                ' epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                ' remain_time:{:.01f}s'.format(np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                )

        #Use the model in final epoch
        if not os.path.exists(os.path.join("../result", self.args.dataset, "model")):
            os.mkdir(os.path.join("../results", self.args.dataset, "model"))
            
        if self.args.run_gumbel_box or self.args.run_soft_box:
            model_name = "gumbel_box_"
        elif self.use_fuzzyset:
            model_name = "fuzzy_model_"
        else:
            model_name = "exp_model_"
        torch.save(self.model.state_dict(), os.path.join("../result",self.args.dataset,"model",model_name+self.exp_setting+".checkpoint"))            
        


    def validation(self,flag):

        encode_query = self.test_set.encode_query
        gt_label = self.test_set.test_gt_id           

        self.model.eval()
        score_list = []
        volumn_list = []
        contain_list = []
        with torch.no_grad():
            query_center,query_delta = self.model.projection_box(encode_query)
            num_query=len(query_center)
            for i in range(num_query):

                sorted_scores = []
                sorted_volumn = []
                hard_contain = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_center,candidate_delta = self.model.projection_box(encode_candidate)
                    num_candidate= len(candidate_center)

                    extend_center = [ query_center[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_delta = [ query_delta[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_center,extend_delta = torch.cat(extend_center,0),torch.cat(extend_delta,0)

                    score,volumn = self.model.condition_score(extend_center,extend_delta,candidate_center,candidate_delta)
                    is_contain = self.model.is_contain(extend_center,extend_delta,candidate_center,candidate_delta)

                    sorted_scores.append(score) 
                    sorted_volumn.append(volumn)
                    hard_contain.append(is_contain)
                sorted_scores = torch.cat(sorted_scores)
                sorted_volumn = torch.cat(sorted_volumn)
                hard_contain = torch.cat(hard_contain)

                score_list.append(sorted_scores.unsqueeze(dim=0))
                volumn_list.append(sorted_volumn.unsqueeze(dim=0))
                contain_list.append(hard_contain.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_volumn = torch.cat(volumn_list,0)
            pred_contain = torch.cat(contain_list,0)
            pred_scores,pred_volumn = pred_scores.detach().cpu().numpy(), pred_volumn.detach().cpu().numpy()
            pred_contain = pred_contain.detach().cpu().numpy()
            ind = np.lexsort((pred_volumn,pred_scores*(-1))) # Sort by pred_scores, then by pred_volumn

            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]
                
            acc,mrr,wu_p = metrics(pred,gt_label,self.test_set.path2root)

        return acc,mrr,wu_p



    """
    validation in the case of fuzzy set 
    """
    def validation_fuzzy(self):
        encode_query = self.test_set.encode_query
        gt_label = self.test_set.test_gt_id           

        self.model.eval()
        score_list = []
        with torch.no_grad():            
            query_fuzzyset = self.model.project_fuzzyset(encode_query)            
            num_query=len(query_fuzzyset)
            for i in range(num_query):

                sorted_scores = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_fuzzyset = self.model.project_fuzzyset(encode_candidate)
                    num_candidate= len(candidate_fuzzyset)
                    
                    extend_fuzzyset = [query_fuzzyset[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_fuzzyset = torch.cat(extend_fuzzyset, 0)
                    
                    score = self.model.condition_score(extend_fuzzyset,candidate_fuzzyset)
                    sorted_scores.append(score)
                    
                sorted_scores = torch.cat(sorted_scores)
                score_list.append(sorted_scores.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_scores = pred_scores.detach().cpu().numpy()
            ind = np.lexsort((pred_scores*(-1), )) # Sort by pred_score
            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]
                
            acc,mrr,wu_p = metrics(pred,gt_label,self.test_set.path2root)

        return acc,mrr,wu_p
        

    """
    validation in the case of gumbel box embeddings
    """
    def validation_fuzzy_box(self):
        encode_query = self.test_set.encode_query
        gt_label = self.test_set.test_gt_id           

        self.model.eval()
        score_list = []
        with torch.no_grad():            
            query_fuzzybox = self.model.projection_box(encode_query)
            query_fuzzybox_max, query_fuzzybox_min = query_fuzzybox.max_embed, query_fuzzybox.min_embed           
            num_query=len(query_fuzzybox.min_embed)
            for i in range(num_query):

                sorted_scores = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_box = self.model.projection_box(encode_candidate)
                    candidate_max, candidate_min = candidate_box.max_embed, candidate_box.min_embed
                    num_candidate= len(candidate_box.min_embed)
                                        
                    extend_max = [query_fuzzybox_max[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_min = [query_fuzzybox_min[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_max, extend_min = torch.cat(extend_max, 0), torch.cat(extend_min,0)
                    
                    if self.args.box_score_mode == "whole":
                        score = self.model.condition_score_(extend_max, extend_min, candidate_max, candidate_min)
                    else:  # average
                        score = self.model.condition_score1(extend_max, extend_min, candidate_max, candidate_min)
                    sorted_scores.append(score)
                sorted_scores = torch.cat(sorted_scores)
                score_list.append(sorted_scores.unsqueeze(dim=0))
  
            pred_scores = torch.cat(score_list,0)
            pred_scores = pred_scores.detach().cpu().numpy()
            ind = np.lexsort((pred_scores*(-1), )) # Sort by pred_score
            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]
                
            acc,mrr,wu_p = metrics(pred,gt_label,self.test_set.path2root)

        return acc,mrr,wu_p
    
    


    def predict(self):
        
        print ("Prediction starting.....")
        self.model.load_state_dict(torch.load(os.path.join("../result",self.args.dataset,"model","exp_model_"+self.exp_setting+".checkpoint")))
        self.model.eval()
        score_list = []
        volumn_list = []
        contain_list = []
        with torch.no_grad():
            query_center,query_delta = self.model.projection_box(self.test_set.encode_query)
            num_query=len(query_center)
            for i in range(num_query):

                sorted_scores = []
                sorted_volumn = []
                hard_contain = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_center,candidate_delta = self.model.projection_box(encode_candidate)
                    num_candidate= len(candidate_center)

                    extend_center = [ query_center[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_delta = [ query_delta[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_center,extend_delta = torch.cat(extend_center,0),torch.cat(extend_delta,0)

                    score,volumn = self.model.condition_score(extend_center,extend_delta,candidate_center,candidate_delta)
                    is_contain = self.model.is_contain(extend_center,extend_delta,candidate_center,candidate_delta)

                    sorted_scores.append(score) 
                    sorted_volumn.append(volumn)
                    hard_contain.append(is_contain)
                sorted_scores = torch.cat(sorted_scores)
                sorted_volumn = torch.cat(sorted_volumn)
                hard_contain = torch.cat(hard_contain)

                score_list.append(sorted_scores.unsqueeze(dim=0))
                volumn_list.append(sorted_volumn.unsqueeze(dim=0))
                contain_list.append(hard_contain.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_volumn = torch.cat(volumn_list,0)
            pred_contain = torch.cat(contain_list,0)
            pred_scores,pred_volumn = pred_scores.detach().cpu().numpy(), pred_volumn.detach().cpu().numpy()
            pred_contain = pred_contain.detach().cpu().numpy()
            ind = np.lexsort((pred_volumn,pred_scores*(-1))) # Sort by pred_scores, then by pred_volumn
            
            print(f"pred_scores = {pred_scores.shape}")
            print(f"pred_volumn = {pred_volumn.shape}")
            print(f"pred_contain = {pred_contain.shape}")
            print(f"ind = {ind}, shape = {ind.shape}")

            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            print(f"pred_before = {pred}")
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]

            print(f"pred_after = {pred}")
            print(f"gt = {self.test_set.test_gt_id}")            
            # save prediction results
            base_path = os.path.join("../result", self.args.dataset, "prediction")
            with open(os.path.join(base_path, "exp_pred.npy"), "wb") as f: 
                np.save(f, pred)
            with open(os.path.join(base_path, "exp_test.npy"), "wb") as f:
                np.save(f, self.test_set.test_gt_id)

            acc,mrr,wu_p = metrics(pred,self.test_set.test_gt_id,self.test_set.path2root)
        
        print('score: acc: {:.05f}'.format(acc),
                'mrr:{:.05f}'.format(mrr),
                'wu_p:{:.05f}'.format(wu_p),)


        self.tosave_pred["metric"] = (acc,mrr,wu_p)
        self.tosave_pred["pred"] = pred
        self.tosave_pred["pred2"] = 0
        self.tosave_pred["pred_scores"] = pred_scores
        self.tosave_pred["pred_volumn"] = pred_volumn
        self.tosave_pred["pred_contain"] = pred_contain
        self.tosave_pred["gt"] = self.test_set.test_gt_id
        self.tosave_pred["path2root"]=self.test_set.path2root



    """
    prediction in fuzzy case 
    """
    def predict_fuzzy(self):
        print ("Prediction starting.....")
        self.model.load_state_dict(torch.load(os.path.join("../result",self.args.dataset,"model","fuzzy_model_"+self.exp_setting+".checkpoint")))
        self.model.eval()
        score_list = []
        with torch.no_grad():
            query_fuzzyset = self.model.project_fuzzyset(self.test_set.encode_query)
            
            # temporarily, generate the embeddings
            embeddings_train = self.model.project_fuzzyset(self.train_set.encode_all)
            embeddings_test = self.model.project_fuzzyset(self.test_set.encode_all)
            torch.save(embeddings_train, "embedding_train_science.pt")
            torch.save(embeddings_test, "embedding_test_science.pt")           
            
            num_query=len(query_fuzzyset)
            for i in range(num_query):
                sorted_scores = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_fuzzyset = self.model.project_fuzzyset(encode_candidate)
                    num_candidate= len(candidate_fuzzyset)

                    extend_fuzzyset = [ query_fuzzyset[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_fuzzyset = torch.cat(extend_fuzzyset,0)

                    score = self.model.condition_score(extend_fuzzyset,candidate_fuzzyset)
                    sorted_scores.append(score)
                    
                sorted_scores = torch.cat(sorted_scores)

                score_list.append(sorted_scores.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_scores = pred_scores.detach().cpu().numpy()
            
            # print(f"pred_scores = {pred_scores}, shape = {pred_scores.shape}")
            
            ind = np.lexsort((pred_scores*(-1), )) # Sort by pred_scores, then by pred_volumn
            
            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            # print(f"pred_before = {pred}")
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]

            # print(f"pred_after = {pred}")
            
             # save prediction results
            base_path = os.path.join("../result", self.args.dataset, "prediction")
            with open(os.path.join(base_path, "fuzzy_pred.npy"), "wb") as f: 
                np.save(f, pred)
            with open(os.path.join(base_path, "fuzzy_test.npy"), "wb") as f:
                np.save(f, self.test_set.test_gt_id)
            
            print(f"gt = {self.test_set.test_gt_id}")
            acc,mrr,wu_p = metrics(pred,self.test_set.test_gt_id,self.test_set.path2root)
            self.save_prediction_fuzzy(pred=pred, actual=self.test_set.test_gt_id)
        
        print('score: acc: {:.05f}'.format(acc),
                'mrr:{:.05f}'.format(mrr),
                'wu_p:{:.05f}'.format(wu_p),)

        self.tosave_pred["metric"] = (acc,mrr,wu_p)
        self.tosave_pred["pred"] = pred
        self.tosave_pred["pred2"] = 0
        self.tosave_pred["pred_scores"] = pred_scores
        self.tosave_pred["gt"] = self.test_set.test_gt_id
        self.tosave_pred["path2root"]=self.test_set.path2root



    """
    prediction in fuzzy case 
    """
    def predict_fuzzy_box(self):
        print ("Prediction starting.....")
        self.model.load_state_dict(torch.load(os.path.join("../result",self.args.dataset,"model","gumbel_box_"+self.exp_setting+".checkpoint")))
        self.model.eval()
        score_list = []
        with torch.no_grad():
            query_fuzzybox = self.model.projection_box(self.test_set.encode_query)           
            query_fuzzybox_max, query_fuzzybox_min = query_fuzzybox.max_embed, query_fuzzybox.min_embed           
            num_query=len(query_fuzzybox.min_embed)
            for i in range(num_query):
                sorted_scores = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_box = self.model.projection_box(encode_candidate)
                    candidate_max, candidate_min = candidate_box.max_embed, candidate_box.min_embed
                    num_candidate= len(candidate_max)
                    extend_max = [query_fuzzybox_max[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_min = [query_fuzzybox_min[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_max, extend_min = torch.cat(extend_max, 0), torch.cat(extend_min,0)
                    if self.args.box_score_mode == "whole":
                        score = self.model.condition_score_(extend_max, extend_min, candidate_max, candidate_min)
                    else:  # average
                        score = self.model.condition_score1(extend_max, extend_min, candidate_max, candidate_min)
                    sorted_scores.append(score)
                    
                sorted_scores = torch.cat(sorted_scores)
                score_list.append(sorted_scores.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_scores = pred_scores.detach().cpu().numpy()
                        
            ind = np.lexsort((pred_scores*(-1), )) # Sort by pred_scores, then by pred_volumn
            
            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]
            
             # save prediction results
            base_path = os.path.join("../result", self.args.dataset, "prediction")
            with open(os.path.join(base_path, "fuzzy_box_pred.npy"), "wb") as f: 
                np.save(f, pred)
            with open(os.path.join(base_path, "fuzzy_box_test.npy"), "wb") as f:
                np.save(f, self.test_set.test_gt_id)
            
            print(f"gt = {self.test_set.test_gt_id}")
            acc,mrr,wu_p = metrics(pred,self.test_set.test_gt_id,self.test_set.path2root)
            self.save_prediction_fuzzy_box(pred=pred, actual=self.test_set.test_gt_id)
        
        print('score: acc: {:.05f}'.format(acc),
                'mrr:{:.05f}'.format(mrr),
                'wu_p:{:.05f}'.format(wu_p),)

        self.tosave_pred["metric"] = (acc,mrr,wu_p)
        self.tosave_pred["pred"] = pred
        self.tosave_pred["pred2"] = 0
        self.tosave_pred["pred_scores"] = pred_scores
        self.tosave_pred["gt"] = self.test_set.test_gt_id
        self.tosave_pred["path2root"]=self.test_set.path2root


    def save_prediction(self):
        
        self.model.eval()
        encode_dic = self.train_set.encode_all
        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"],encode_dic["token_type_ids"],encode_dic["attention_mask"]
        length = self.args.batch_size
        l = 0
        r = length
        center_list = []
        delta_list = []
        with torch.no_grad():
            while l < (len(input_ids)):
                r = min (r,len(input_ids))
                encode = {
                    "input_ids":input_ids[l:r],
                    "token_type_ids":token_type_ids[l:r],
                    "attention_mask":attention_mask[l:r]
                }
                center,delta = self.model.projection_box(encode)
                center = center.detach().cpu().numpy()
                delta = delta.detach().cpu().numpy()
                center_list.append(center)
                delta_list.append(delta)
                l = r
                r+=length
        center = np.concatenate(center_list)
        delta = np.concatenate(delta_list)
                

        self.tosave_box["left"] = center-delta
        self.tosave_box["right"] = center+delta
        self.tosave_box["center"] = center
        self.tosave_box["delta"] = delta

        if not os.path.exists(os.path.join("../result", self.args.dataset, "box")):
            os.mkdir(os.path.join("../results", self.args.dataset, "box"))
        
        if not os.path.exists(os.path.join("../result", self.args.dataset, "prediction")):
            os.mkdir(os.path.join("../results", self.args.dataset, "prediction"))

        with open(os.path.join("../result",self.args.dataset,"box","exp_box_"+self.exp_setting+".pkl"),"wb") as f:
            pkl.dump(self.tosave_box,f)

        with open(os.path.join("../result",self.args.dataset,"prediction","exp_pred_"+self.exp_setting+".pkl"),"wb") as f:
            pkl.dump(self.tosave_pred,f)
        

        print ("================================Save results done!================================")




    """
    in this case, print out the actual entities predicted and the correct entities
    """
    def save_prediction_fuzzy(self, pred, actual):
        import pickle
        data_path = os.path.join("../data", self.args.dataset, "processed", "taxonomy_data_0_.pkl")
        with open(data_path, "rb") as f:
            data_file = pickle.load(f)
        id2_concept = data_file["id2concept"]
        
        pred_entities = np.vectorize(id2_concept.get)(pred)
        actual_entities = np.vectorize(id2_concept.get)(actual)
        
        with open(os.path.join("../result", self.args.dataset, "prediction", "fuzzy_pred_entities" + self.exp_setting + "npy"), "wb") as f:
            np.save(f, pred_entities)
        with open(os.path.join("../result", self.args.dataset, "prediction", "fuzzy_test_entities" + self.exp_setting + "npy"), "wb") as f:
            np.save(f, actual_entities)

        print ("================================Save results done!================================")


    """
    in this case, print out the actual entities predicted and the correct entities
    """
    def save_prediction_fuzzy_box(self, pred, actual):
        import pickle
        data_path = os.path.join("../data", self.args.dataset, "processed", "taxonomy_data_0_.pkl")
        with open(data_path, "rb") as f:
            data_file = pickle.load(f)
        id2_concept = data_file["id2concept"]
        
        pred_entities = np.vectorize(id2_concept.get)(pred)
        actual_entities = np.vectorize(id2_concept.get)(actual)
        
        with open(os.path.join("../result", self.args.dataset, "prediction", "fuzzy_box_pred_entities" + self.exp_setting + "npy"), "wb") as f:
            np.save(f, pred_entities)
        with open(os.path.join("../result", self.args.dataset, "prediction", "fuzzy_box_test_entities" + self.exp_setting + "npy"), "wb") as f:
            np.save(f, actual_entities)

        print ("================================Save results done!================================")
