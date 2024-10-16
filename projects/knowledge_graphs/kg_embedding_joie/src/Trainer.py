from KG import KG
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from tqdm import tqdm
from data_loader import TrainDataset, TestDataset, BidirectionalOneShotIterator
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from fuse import SimpleFuzzySet

"""
Trainer class thta builds up the dataloader and performs training.
two options: either pass the model in constructor or construct the model inside the constructor. 
"""
class Trainer():
    def __init__(self, args, shuffle=True):
        # attributes
        self.args = args
        self.kg_file_path = args.kg_file_path
        self.kg_ontology_path = args.kg_ontology_path

        self.kg_train_file_path = args.kg_train_file_path
        self.kg_train_ontology_path = args.kg_train_ontology_path

        self.kg_testfile_path = args.kg_testfile_path
        self.kg_test_ontology_path = args.kg_test_ontology_path

        self.kg_batchsize = args.kg_batchsize
        self.test_batch_size = args.test_batch_size
        self.n_negative_samples = args.n_negative_samples
        self.max_steps = args.max_steps
        self.every_n_steps_to_save = args.every_n_steps_to_save
        self.every_n_steps_to_log = args.every_n_steps_to_log
        self.every_n_steps_scheduler_update = args.every_n_steps_scheduler_update
        self.valid_steps = args.valid_steps
        self.warm_up_steps = args.warm_up_steps
        self.lr = args.lr
        self.margin_p = args.margin_p
        self.margin_n = args.margin_n
        self.uni_weight = args.uni_weight

        self.shuffle = shuffle
        self.device = args.device
        self.cpu_num = args.cpu_num
        self.from_checkpoint = args.from_checkpoint
        self.checkpoint_path = args.checkpoint_path

        self.ontology_weight = args.alpha
        self.kg_weight = args.beta

        # data objects 
        self.kg = self.setup_KG()
        self.onto_rel_index = self.kg.onto_rel_index
        self.kg_len = self.kg.num_triples()
        self.n_batches_kg = int(self.kg_len / self.kg_batchsize)

        # model
        self.model = SimpleFuzzySet(self.kg.n_ents, 
                                    self.kg.n_rels, 
                                    self.kg.onto_rel_index,
                                    args).to(self.device)
        
        # optimizer, scheduler
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr)
        
        if args.scheduler == "multistep":
            self.scheduler = MultiStepLR(self.optimizer, 
                                     milestones=[self.max_steps // 2,
                                                 int(self.max_steps // 1.5)])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)



        
    # set up multiG object
    def setup_KG(self):
        kg = KG()
        
        kg.load_all_triples(self.kg_file_path, 
                            self.kg_ontology_path)

        kg.load_triples_train(self.kg_train_file_path, 
                              self.kg_train_ontology_path)
        
        kg.load_triples_test(self.kg_testfile_path, 
                             self.kg_test_ontology_path)
        
        self.all_true_triples = kg.triples_record

        print(f"finished setting up KG.")
        return kg

    
    # each step of training: if relationship embedding index is in the range of ontology relation
    # we use a different score/loss function. 
    def train_step(self, train_iterator):
        self.model.train()
        self.optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if self.device != "cpu":
            positive_sample = positive_sample.to(self.device)
            negative_sample = negative_sample.to(self.device)
            subsampling_weight = subsampling_weight.to(self.device)

        start_time = time.time()
        negative_score = self.model((positive_sample, negative_sample), mode=mode)
        # NOTE: logsigmoid doesn't work if the loss vaues have large absolute values 
        # in the perspective of loss: negative score needs to be large while positive scores need to be negative or small.
        negative_score = negative_score.mean(dim=1)
        positive_score = self.model(positive_sample)
        positive_score = positive_score.squeeze(dim=1)

        if self.uni_weight:
            positive_sample_loss = positive_score.mean()
            negative_sample_loss = negative_score.mean()
        else:
            positive_sample_loss = (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        # NOTE: scores for positive samples should be large while for negative scores should be close to zero.
        # NOTE: try weighted sum with more focus on the negatives
        loss = negative_score + 0.5 * (self.args.n_partitions - positive_score)
        
        # loss = negative_sample_loss - positive_sample_loss
        # positive scores should be positively large, while negative should be positive small  
        # loss = (negative_sample_loss + positive_sample_loss)/2
        print(f"loss = {loss}")
        loss.backward()
        self.optimizer.step()
        # print(f"forward time takes = {(forward_time - start_time)} seconds")
        # print(f"negative pass takes = {negative_time - start_time} seconds")
        # print(f"positive pass takes = {forward_time-negative_time} seconds")
        # print(f"backward time takes = {backward_time - forward_time} seconds")

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log 
            
    # training logic: kg training is by steps rather than epochs
    # training loader hsa head-batch vs. tail-batch mode, resulting in the same for model. 
    # training iterator goes through head-batch then tail-batch
    def train(self):
        train_dataloader_head = DataLoader(
            TrainDataset(self.kg.train_triples, self.kg.n_ents, 
                         self.kg.n_rels, self.n_negative_samples, 'head-batch'), 
            batch_size=self.kg_batchsize,
            shuffle=True, 
            num_workers=0,
            # num_workers=max(1, self.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(self.kg.train_triples, self.kg.n_ents, 
                         self.kg.n_rels, self.n_negative_samples, 'tail-batch'), 
            batch_size=self.kg_batchsize,
            shuffle=True, 
            num_workers=0,
#             num_workers=max(1, self.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        init_step = 0
        
        # if loading a pretrained checkpoint
        if self.from_checkpoint:
            checkpoint = torch.load(os.path.join(self.checkpoint_path, "checkpoint_latest.pt"))
            init_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.lr = checkpoint['current_learning_rate']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        t_start = time.time()
        training_logs = []
        positive_sample_losses, negative_sample_losses, total_losses = [],[],[]
        running_avg_pos, running_avg_neg, running_avg_total = [],[],[]
        for step in tqdm(range(init_step, self.max_steps)):
            
            one_step_starttime = time.time()
            log =  self.train_step(train_iterator)
            one_step_endtime = time.time()
            print(f"train one iteration on batch takes {one_step_endtime - one_step_starttime} seconds")
            training_logs.append(log)
            positive_sample_losses.append(log["positive_sample_loss"])
            negative_sample_losses.append(log["negative_sample_loss"])
            total_losses.append(log["loss"])

            if step > 0 and step % self.every_n_steps_to_log == 0: 
                avg_positive_loss = np.mean(np.array(positive_sample_losses))
                avg_negative_loss = np.mean(np.array(negative_sample_losses))
                avg_total_loss = np.mean(np.array(total_losses))
                running_avg_pos.append(avg_positive_loss)
                running_avg_neg.append(avg_negative_loss)
                running_avg_total.append(avg_total_loss)
                
                print(f"step = {step}, avg positive loss = {avg_positive_loss}")
                print(f"average negative loss = {avg_negative_loss}")
                print(f"average combined loss = {avg_total_loss}")
            if step > 0 and step % self.every_n_steps_scheduler_update:
                self.scheduler.step()
            if step > 0 and step % self.every_n_steps_to_save == 0:
                date = "{:%Y_%m_%d}".format(datetime.now())
                torch.save(self.model, f"checkpoints/fuse_ckpt_{step}_{date}.pt")
                print(f"saving model at step = {step}")
            if step > 0 and step % self.valid_steps == 0:
                print(f"validation at step {step}")
                metrics = self.evaluate()
                print(metrics)
        
        t_end = time.time()
        print(f"elapsed time = {t_end - t_start} seconds")
        # visualization
        indices = [i * self.every_n_steps_to_log for i in range(len(running_avg_total))]
        _, axes = plt.subplot(nrows=3, ncols=1)
        axes[0].set_title("training positive loss")
        axes[0].plot(indices, running_avg_pos)
        axes[1].set_title("training negative loss")
        axes[1].plot(indices, running_avg_neg)
        axes[2].set_title("training combined loss")
        axes[2].plot(indices, running_avg_total)
        date = "{:%Y_%m_%d}".format(datetime.now())
        plt.savefig(f"fuse_training_loss_{date}.png")


    
    
    ########################################################
    ################ Test Flows ############################
    ########################################################
    
    def evaluate(self):
        self.model.eval()
        test_dataloader_head = DataLoader(
            TestDataset(
                self.kg.test_triples,
                self.kg.triples_record,
                self.kg.n_ents,
                self.kg.n_rels,
                'head-batch'
            ),
            batch_size=self.test_batch_size,
            num_workers=0,
            # num_workers=max(1, self.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                self.kg.test_triples,
                self.kg.triples_record,
                self.kg.n_ents,
                self.kg.n_rels,
                'tail-batch'
            ),
            batch_size=self.test_batch_size,
            num_workers=0,
            # num_workers=max(1, self.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if self.device != "cpu":
                        positive_sample = positive_sample.to(self.device)
                        negative_sample = negative_sample.to(self.device)
                        filter_bias = filter_bias.to(self.device)
                    batch_size = positive_sample.size(0)
                    
                    score = self.model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    argsort = torch.argsort(score, dim = 1, descending=True)
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % self.every_n_steps_to_log == 0:
                        print('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)
            
        return metrics        