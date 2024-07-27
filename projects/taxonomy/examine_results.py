import torch
import pickle as pkl
import os
import numpy as np
import sys

"""
This script is used to perform the (additional) experiments related to set union and set complement 
"""


"""
input: union embedding of all child 
       rank its position in all parents
       and compare it against the actual parent's id.
"""
def compute_rank(id2concept, union_all_children, children, parent_id, parent_embs):
    parent_list = data[f"{mode}_parent_list"]
    print(parent_list)
    dist_list = [torch.norm(parent_emb.cpu() - union_all_children.cpu()) for parent_emb in parent_embs]
    print(f"dist_list = {dist_list}")
    print(np.argsort(dist_list))
    ranks = [parent_list[idx] for idx in np.argsort(dist_list)]
    rank_parent = ranks.index(parent_id)
    correctness = (parent_id == ranks[0])
    print(f"children = {children}")
    print(f"parent_id = {parent_id}, concept = {id2concept[parent_id]}")
    print(f"ranks = {ranks}")
    print("top 5 ranked concepts are: ")
    for rank in ranks[:5]:
        print(f"{id2concept[rank]}")
    return rank_parent+1, correctness


"""
take the union of the embeddings and examine the ranks 
"""
def examine_union(embedding, parent_embs, id2concept, union_pairs):
    n = len([key for key in union_pairs.keys()])
    mrr = 0
    n_corrects = 0 
    for parent in union_pairs.keys():
        children = [id2concept[child] for child in union_pairs[parent]]
        children_emb = embedding[union_pairs[parent]]
        union_all_children = None
        for child_emb in children_emb:
            if union_all_children == None:
                union_all_children = child_emb
            else:
                union_all_children = (union_all_children + child_emb - union_all_children * child_emb)
        rank, correctness = compute_rank(id2concept, union_all_children, children, parent, parent_embs)
        mrr += (1/rank)
        n_corrects += correctness
    mrr /= n 
    accuracy = n_corrects / n 
    print(f"mrr = {mrr}")
    print(f"accuracy = {accuracy}")
    


"""
input: complement embedding 
       rank its position in all children
       and compare it against the actual child's id.
"""
def compute_rank_c(data, complement_emb, child_id):
    child_list = data[f"{mode}_child_list"]
    dist_list = [torch.norm(parent_emb.cpu() - complement_emb.cpu()) for parent_emb in parent_embs]
    print(f"dist_list = {dist_list}")
    print(np.argsort(dist_list))
    ranks = [child_list[idx] for idx in np.argsort(dist_list)]
    rank_parent = ranks.index(child_id)
    correctness = (child_id == ranks[0])
    print(f"child_id = {child_id}, concept = {id2concept[child_id]}")
    print(f"ranks = {ranks}")
    print("top 5 ranked child concepts are: ")
    for rank in ranks[:5]:
        print(f"{id2concept[rank]}")
    return rank_parent+1, correctness


"""
take the complement of concepts and examine the results rank
"""
def examine_complement(data, complement_pairs, embedding):
    n = len([key for key in complement_pairs.keys()])
    mrr = 0
    n_corrects = 0 
    for parent in complement_pairs.keys():
        children = complement_pairs[parent]
        parent_emb = embedding[parent]
        remaining = children.pop(-1)
        children_emb = embedding[children]
        
        union_all_children = None
        for child_emb in children_emb:
            if union_all_children == None:
                union_all_children = child_emb
            else:
                union_all_children = (union_all_children + child_emb - union_all_children * child_emb)
        
        minus_emb = parent_emb * (1 - union_all_children)
        print(f"parent = {id2concept[parent]}")
        print(f"all rest children = {[id2concept[child] for child in children]}")
        print(f"left out children = {id2concept[remaining]}")
        rank, correctness = compute_rank_c(data, minus_emb, remaining)
        mrr += (1/rank)
        n_corrects += correctness
    mrr /= n 
    accuracy = n_corrects / n 
    print(f"mrr = {mrr}")
    print(f"accuracy = {accuracy}")





if __name__ == "__main__":
    dataset, mode = sys.argv[-2], sys.argv[-1]

    # load embedding
    embedding = torch.load(f"../results/taxonomy/embedding_{mode}_{dataset}.pt")

    with open(os.path.join("../data/taxonomy/",dataset,"processed","taxonomy_data_0_.pkl"),"rb") as f:
        data = pkl.load(f)

    train_parent_list = data[f"{mode}_parent_list"]
    child_parent_pair = data["child_parent_pair"]

    union_pairs = {parent: [] for parent in train_parent_list}

    for (child, parent) in child_parent_pair:
        union_pairs[parent].append(child)

    # complement: parent minus all but one child
    # only take 2 or more children parents 
    complement_pairs = {}
    for parent in union_pairs.keys():
        if len(union_pairs[parent]) > 1:
            complement_pairs[parent] = union_pairs[parent]

    parent_embs = embedding[train_parent_list]
    id2concept = data["id2concept"]

    print("examine fuzzy set embedding union results")
    examine_union(embedding,  parent_embs, id2concept, union_pairs)
    print("examine fuzzy set embedding complement results")
    examine_complement(data, complement_pairs, embedding)