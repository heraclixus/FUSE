from Trainer import Trainer
from data_loader import TrainDataset, TestDataset, BidirectionalOneShotIterator
from torch.utils.data import DataLoader
from KG import KG
import pickle
import argparse
import torch


def examine_a_batch(positive_sample, negative_samples, subsampling_weight, mode, index2ent, index2rel):
    print(f"mode = {mode}")
    print(f"subsampling weight = {subsampling_weight}")

    
    if mode == "head-batch": # head-batch, expect to have 
        positive_head, rel, positive_tail = positive_sample[:,0], positive_sample[:,1], positive_sample[:,2]
        negative_heads = negative_samples
    if mode == "tail-batch": 
        positive_head, rel, positive_tail = positive_sample[:,0], positive_sample[:,1], positive_sample[:,2]
        negative_tails = negative_samples    
    
    print(f"positive head index = {positive_head.item()}, entity = {index2ent[positive_head.item()]}")
    print(f"positive tail index = {positive_tail.item()}, entity = {index2ent[positive_tail.item()]}")
    print(f"relation = {rel.item()}, or {index2rel[rel.item()]}")
    
    if mode == "head-batch":
        negative_heads = negative_heads.squeeze()
        for negative_head in negative_heads:
            print(f"negative head = {negative_head}, entity = {index2ent[negative_head.item()]}")        
        
    if mode == "tail-batch":
        negative_tails = negative_tails.squeeze()
        for negative_tail in negative_tails:
            print(f"negative tail = {negative_tail}, entity = {index2ent[negative_tail.item()]}")




def obtain_one_sample_from_data_loader(data_loader: DataLoader, 
                                       index2ent: dict, 
                                       index2rel: dict):
    
    for batch in data_loader:
        positive_sample, negative_samples, subsampling_weight, mode = batch
        break
    examine_a_batch(positive_sample, negative_samples, subsampling_weight, mode, index2ent, index2rel)



if __name__ == "__main__":
    print("test trainer class")

    parser = argparse.ArgumentParser(description="Training FUSE for Knowledge Graph Embedding")
    # fuzzy set modeling related
    parser.add_argument("--entity_embed_dim", type=int, default=512)
    parser.add_argument("--n_partitions", type=int, default=100)
    parser.add_argument("--regularizer", type=str, default="sigmoid")
    parser.add_argument("--n_rel_basis", type=int, default=50)
    parser.add_argument("--partition_reg_type", type=str, default="sigmoid")
    parser.add_argument("--modulelist", action="store_true")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--use_volume_weights", action="store_true")
    parser.add_argument("--regularize_volume", action="store_true")
    parser.add_argument("--score_type", type=str, default="cosine")
    parser.add_argument("--dropout", type=float, default=0.3)

    
    # resource related
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--cpu_num", type=int, default=10)
    
    # paths
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--kg_file_path", type=str, default="data/dbpedia/db_kg_mini.txt")
    parser.add_argument("--kg_ontology_path", type=str, default="data/dbpedia/db_InsType_mini.txt")
    parser.add_argument("--kg_train_file_path", type=str, default="data/dbpedia/db_kg_train.txt")
    parser.add_argument("--kg_train_ontology_path", type=str, default="data/dbpedia/db_InsType_train.txt")
    parser.add_argument("--kg_testfile_path", type=str, default="data/dbpedia/db_kg_test.txt")
    parser.add_argument("--kg_test_ontology_path", type=str, default="data/dbpedia/db_InsType_test.txt")    
    
    
    # training configurations    
    parser.add_argument("--kg_batchsize", type=int, default=2056)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--n_negative_samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--every_n_steps_to_log", type=int, default=1000)
    parser.add_argument("--every_n_steps_to_save", type=int, default=10000)
    parser.add_argument("--every_n_steps_scheduler_update", type=int, default=5000)
    parser.add_argument("--valid_steps", default=10000, type=int)
    parser.add_argument("--warm_up_steps", default=None, type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin_p", type=float, default=0)
    parser.add_argument("--margin_n", type=float, default=0)
    parser.add_argument("--uni_weight", action="store_true")
    parser.add_argument("--scheduler", type=str, default="multistep")

    # loss weights
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--scale", type=int, default=10)

    args = parser.parse_args()
    trainer = Trainer(args)
    
    # get mappings
    index_ents, index_rels = trainer.kg.index_ents, trainer.kg.index_rels
    index2ent = {val:key for (key,val) in index_ents.items()}
    index2rel = {val:key for (key,val) in index_rels.items()}


    # test data loader with one minibatch
    train_dataloader_head = DataLoader(
        TrainDataset(trainer.kg.train_triples, trainer.kg.n_ents, 
                        trainer.kg.n_rels, trainer.n_negative_samples, 'head-batch'), 
        batch_size=1,
        shuffle=True, 
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    
    train_dataloader_tail = DataLoader(
        TrainDataset(trainer.kg.train_triples, trainer.kg.n_ents, 
                        trainer.kg.n_rels, trainer.n_negative_samples, 'tail-batch'), 
        batch_size=1,
        shuffle=True, 
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    
    
    # take a look at first train data loader 
    print("train dataloader head")
    obtain_one_sample_from_data_loader(train_dataloader_head, index2ent, index2rel)
    print("train dataloader tail")
    obtain_one_sample_from_data_loader(train_dataloader_tail, index2ent, index2rel)
    

    print("bidirectional iterator")    
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    positive_sample1, negative_samples1, subsampling_weight1, mode1 = next(train_iterator)
    examine_a_batch(positive_sample1, negative_samples1, subsampling_weight1, mode1, index2ent, index2rel)
    print("another batch....")
    positive_sample, negative_samples, subsampling_weight, mode = next(train_iterator)
    examine_a_batch(positive_sample, negative_samples, subsampling_weight, mode, index2ent, index2rel)
    
    # now examine model
    
    train_dataloader_head = DataLoader(
        TrainDataset(trainer.kg.train_triples, trainer.kg.n_ents, 
                        trainer.kg.n_rels, trainer.n_negative_samples, 'head-batch'), 
        batch_size=64,
        shuffle=True, 
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    
    train_dataloader_tail = DataLoader(
        TrainDataset(trainer.kg.train_triples, trainer.kg.n_ents, 
                        trainer.kg.n_rels, trainer.n_negative_samples, 'tail-batch'), 
        batch_size=64,
        shuffle=True, 
        num_workers=0,
        collate_fn=TrainDataset.collate_fn
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    positive_sample1, negative_samples1, subsampling_weight1, mode1 = next(train_iterator)
    positive_sample, negative_samples, subsampling_weight, mode = next(train_iterator)
    print("examining the FUSE model on batch size = 64")
    model = trainer.model

    print(subsampling_weight)
    
    # positive samples
    print("-----------------------------------------------------------------")
    print("looking at how positive sample (single mode) are forwarded...")
    # positive_sample = (positive_head, rel, negative_head) and mode = "head-batch" 
    # by default this is the single mode, where we process the triples.
    head_kg, rel_kg, tail_kg, head_onto, tail_onto = model.encode_kg_triples_from_sample(positive_sample1)
    print(f"number of kg_samples = {len(head_kg)}, onto_samples = {len(head_onto)}")
    if head_onto != None: 
        print(head_onto.shape)
    if tail_onto != None: 
        print(tail_onto.shape)
    score_kg = model.compute_score_for_kg_triples(head_kg, rel_kg, tail_kg, mode=mode)
    print(f"mean kg score = {torch.mean(torch.mean(score_kg))}, shape = {score_kg.shape}")
    if head_onto != None and tail_onto != None:
        score_onto = model.compute_score_for_ontology_triples(head_onto, tail_onto, mode=mode)
        print(f"mean onto score = {torch.mean(torch.mean(score_onto))}, shape = {score_onto.shape}")


    print("-----------------------------------------------------------------") 
    print("looking at how negative samples (head-batch) are calculated...")
    print(mode)
    head_kg, rel_kg, tail_kg, head_onto, tail_onto = model.encode_kg_triples_from_sample((positive_sample, negative_samples), mode=mode)
    print(f"number of kg_samples = {len(head_kg)}, onto_samples = {len(head_onto)}")
    if head_onto != None: 
        print(head_onto.shape)
    if tail_onto != None: 
        print(tail_onto.shape)
    score_kg = model.compute_score_for_kg_triples(head_kg, rel_kg, tail_kg, mode=mode)
    print(f"mean kg score = {torch.mean(torch.mean(score_kg))}, shape = {score_kg.shape}")
    if head_onto != None and tail_onto != None:
        score_onto = model.compute_score_for_ontology_triples(head_onto, tail_onto, mode=mode)
        print(f"mean onto score = {torch.mean(torch.mean(score_onto))}, shape = {score_onto.shape}")    

    print("-----------------------------------------------------------------")
    print("looking at how negative samples (tail-mode) are calculated...")
    print(mode)
    head_kg, rel_kg, tail_kg, head_onto, tail_onto = model.encode_kg_triples_from_sample((positive_sample1, negative_samples1), mode=mode1)
    print(f"number of kg_samples = {len(head_kg)}, onto_samples = {len(head_onto)}")
    if len(head_onto) > 0: 
        print(head_onto.shape)
    if len(tail_onto) > 0:
        print(tail_onto.shape)
    score_kg = model.compute_score_for_kg_triples(head_kg, rel_kg, tail_kg, mode=mode1)
    print(f"mean kg score = {torch.mean(torch.mean(score_kg))}, shape = {score_kg.shape}")
    if head_onto != None and tail_onto != None:
        score_onto = model.compute_score_for_ontology_triples(head_onto, tail_onto, mode=mode1)
        print(f"mean onto score = {torch.mean(torch.mean(score_onto))}, shape = {score_onto.shape}")
    
    
    print("-----------------------------------------------------------------")
    print("end to end compute a score from the model")

    score = model((positive_sample1, negative_samples1), mode=mode1)
    print(f"negative (tail) score = {score}")


    score = model((positive_sample, negative_samples), mode=mode)
    print(f"negative (head) score = {score}")

    score = model(positive_sample)
    print(f"positive score = {score}")