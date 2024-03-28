import pickle as pkl
import pandas as pd
import random
from torch.utils.data import Dataset

"""
create all necessary data structures
- category2id: {Str: Int} maps category (cs.AI) to its id
- id2category: {Int: Str} maps id to its category 
- id2title: {Int: Str} maps id to title (string)
- title2Id: {Str: Int} maps title to id
- id2context: {Int: Str} maps category id to category description
- title2catid: {Int: Int} maps title id to category id
- triples: [(Int, Int, Int)] triples of (category_id, positive_title_id, negative_title_id)

arguments:
    - n_negatives: determines how many negative samples per positive sample for triplets
"""
def create_dataset(n_negatives=2):
    df_categories = pd.read_csv("../../data/arxiv/categories.txt", delimiter="\t", header=None, index_col=None)
    categories = df_categories.iloc[:, 0].to_list()
    descriptions = df_categories.iloc[:,1].to_list()
    catids = [i for i in range(len(categories))]

    id2category = {catids[i]: categories[i] for i in range(len(categories))}
    category2id = {categories[i]: catids[i] for i in range(len(categories))}
    id2context = {catids[i]: descriptions[i] for i in range(len(categories))}

    df_arxiv = pd.read_csv("../../data/arxiv/arxiv.taxo", delimiter="\t", header=None, index_col=None)
    titles = list(set(df_arxiv.iloc[:, 0].to_list()))
    
    titleids = [len(categories) + i for i in range(len(titles))]
    id2title = {titleids[i]: titles[i] for i in range(len(titles))}
    title2id = {titles[i]: titleids[i] for i in range(len(titles))}

    title2catid = {} 
    cat2titleid = {} 
    for i in range(len(df_arxiv)):
        entry = df_arxiv.iloc[i]
        title, category = entry[0], entry[1]
        title_id, category_id = title2id[title], category2id[category]
        if title not in title2catid.keys():
            title2catid[title_id] = set([category_id])
        else:
            title2catid[title_id].add(category_id)
        if category_id not in cat2titleid.keys():
            cat2titleid[category_id] = set([title_id])
        else:
            cat2titleid[category_id].add(title_id)


    # generate triplets based on positive and negative pairs for categories
    triples = []
    catids, titleids = set(catids), set(titleids)
    for cid in cat2titleid.keys():
        # print(id2category[cid])
        tids = cat2titleid[cid]
        nids = titleids.difference(tids)
        tids = list(tids)
        nids = list(nids)
        # print(len(nids))
        # print(n_negatives * len(tids))
        tids_n = random.sample(nids, n_negatives * len(tids))
        for i in range(len(tids)):
            for j in range(n_negatives):
                triples.append((cid, tids[i], tids_n[j + n_negatives * i]))
    print(f"number of triples = {len(triples)}")
    print("saving processed data")
    save_data = {
        "categories": categories,
        "catids": list(catids),
        "titles": titles,
        "titleids": list(titleids),
        "id2category": id2category,
        "category2id": category2id,
        "id2context": id2context,
        "id2title": id2title,
        "title2id": title2id,
        "title2catid": title2catid, 
        "cat2titleid": cat2titleid,
        "triples": triples
    }
    with open("../../data/arxiv/processed/arxiv_data.pkl","wb") as f:
        pkl.dump(save_data,f)


""" 
Dataset for the task of set representation learning on arxiv dataset
with the goal of generating new sets from old sets (based on category)
cs.AI OR cs.CV, etc. 
"""
# this generates logical expressions based on the sets
class SetDataset(Dataset):
    def __init__(self, mode, tokenizer, device, data_file="../../data/arxiv/processed/arxiv_data.pkl"):
        self.data_file = data_file
        with open(data_file, "rb") as f:
            data_file = pkl.load(f)
            self.categories = data_file["categories"]
            self.catids = data_file["catids"]
            self.category2id = data_file["category2id"]
            self.cat2titleid = data_file["cat2titleid"]
            self.id2context = data_file["id2context"]
            self.id2title = data_file["id2title"]
            self.titleids = data_file["titleids"]
            self.triples = data_file["triples"]
        self.mode = mode
        self.device = device
        self.tokenizer = tokenizer
        self.logical_clauses = ["OR", "AND", "NOT"]
        # if mode == "logic":
            # TODO: generate new sets from existing sets using set operations 
            # self._generate_expressions()
            # self.save_new_sets() # new set data
            # self.generate_pair_data() # new graph
        self.encode_all = self.generate_all_token_ids(self.tokenizer)


    """
    for all the category contexts and all the titles 
    generate their embedding using BERT tokenizer
    """
    def generate_all_token_ids(self, tokenizer):
        all_contexts = [self.id2context[cid] for cid in self.catids] # string description for categories
        all_titles = [self.id2title[cid] for cid in self.titleids] # string for title
        all_contexts.extend(all_titles) 
        encode_all = tokenizer(all_contexts, padding=True,return_tensors='pt')
        
        if self.device != "cpu":
            a_input_ids = encode_all['input_ids'].to(self.device)
            a_token_type_ids = encode_all['token_type_ids'].to(self.device)
            a_attention_mask = encode_all['attention_mask'].to(self.device)

            encode_all = {'input_ids' : a_input_ids, 
                        'token_type_ids' : a_token_type_ids, 
                        'attention_mask' : a_attention_mask} 
        return encode_all


    """
    provide indices for the tokens 
    """
    def index_token_ids(self,encode_dic,index):

        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"], encode_dic["token_type_ids"], encode_dic["attention_mask"]
        
        res_dic = {'input_ids' : input_ids[index], 
                        'token_type_ids' : token_type_ids[index], 
                        'attention_mask' : attention_mask[index]}
        return res_dic


    """
    This determines the input fed into the downstream model. 
    """
    def generate_category_title_token_ids(self,index):
        cat_id, title_id, negative_title_id = self.triples[index]
        encode_category = self.index_token_ids(self.encode_all, cat_id)
        encode_title = self.index_token_ids(self.encode_all, title_id)
        encode_negative_title = self.index_token_ids(self.encode_all,negative_title_id)
        return encode_category, encode_title, encode_negative_title


    """
    in training, the model receives parent child pairs and negative parent child pairs. 
    """
    def __getitem__(self, index):
        return self.generate_category_title_token_ids(index)

    def __len__(self):
        return len(self.triples)


if __name__ == "__main__":
    create_dataset(n_negatives=2)

