import numpy as np
import os 
import pickle
from tqdm import tqdm

class KG(object):
    '''
    This class stores triple data, descriptions, and word embeddings for a langauge.
    '''
    def __init__(self):
        # entities (fuzzy sets)
        self.ents = {}
        self.ent_tokens = {}
        # standard relations
        self.rels = {}
        # is-a relations (all the ontology relations have the same encoding in our framework)
        self.onto_rel_index = 0  
        self.index_ents = {}
        self.index_rels = {}
        self.n_ents = 0
        self.n_rels = 0
        
        # save triples as array of indices (train | test)
        self.triples = [] 
        self.triples_record = set([])
        self.train_triples = []
        self.train_triples_record = set([])
        self.test_triples = []
        self.test_triples_record = set([])
        
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        # word embeddings
        self.tokens = []
        self.wv = np.array([0])
        self.token_index = {}
        self.loaded_wv = False
        self.n_tokens = 0
        # descriptions (in our case, also can be seen as "contexts")
        self.descriptions = {}
        self.desc_embed = {}
        self.avg_embed = {}
        self.desc_embed_padded = np.array([0])
        self.avg_embed_padded = np.array([0])
        self.desc_length = 100
        self.desc_index = np.array([0])

        self.dim = 100
        self.wv_dim = 100
        self.batch_size = 1024


    # load all triples from "mini"
    def load_all_triples(self, triple_filename, ontology_filename, splitter="\t", line_end="\n"):
        '''Load the dataset.'''
        last_c = self.n_ents -1 
        last_r = self.n_rels - 1 

        # process standard triple files. 
        for line in open(triple_filename):
            line = line.rstrip(line_end).split(splitter)
            if self.index_ents.get(line[0]) == None:
                last_c += 1
                self.ents[last_c] = line[0]
                self.index_ents[line[0]] = last_c
                self.ent_tokens[last_c] = set(line[0].replace('(','').replace(')','').split(' '))
            if self.index_ents.get(line[2]) == None:
                last_c += 1
                self.ents[last_c] = line[2]
                self.index_ents[line[2]] = last_c
                self.ent_tokens[last_c] = set(line[2].replace('(','').replace(')','').split(' '))
            if self.index_rels.get(line[1]) == None:
                last_r += 1
                self.rels[last_r] = line[1]
                self.index_rels[line[1]] = last_r
            h = self.index_ents[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_ents[line[2]]
            self.triples.append([h, r, t])
            self.triples_record.add((h, r, t))

        # process ontology based triples
        last_r += 1
        self.onto_rel_index = last_r 
        self.rels[self.onto_rel_index] = "type"
        for line in open(ontology_filename):
            line = line.rstrip(line_end).split(splitter)
            if self.index_ents.get(line[0]) == None:
                last_c += 1
                self.ents[last_c] = line[0]
                self.index_ents[line[0]] = last_c
                self.ent_tokens[last_c] = set(line[0].replace('(','').replace(')','').split(' '))
            if self.index_ents.get(line[2]) == None:
                last_c += 1
                self.ents[last_c] = line[2]
                self.index_ents[line[2]] = last_c
                self.ent_tokens[last_c] = set(line[2].replace('(','').replace(')','').split(' '))
            # relation ontology all have the same index
            self.index_rels[line[1]] = self.onto_rel_index
            h = self.index_ents[line[0]]
            t = self.index_ents[line[2]]
            self.triples.append([h, self.onto_rel_index, t])
            self.triples_record.add((h, self.onto_rel_index, t))

        self.n_ents = last_c + 1
        # all ontology belongs to one type of relation
        self.n_rels = last_r

        # negative sampling
        tph_array = np.zeros((len(self.rels), len(self.ents)))
        hpt_array = np.zeros((len(self.rels), len(self.ents)))
        for h,r,t in self.triples:
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
        self.tph = np.mean(tph_array, axis = 1)
        self.hpt = np.mean(hpt_array, axis = 1)

        # saving state dicts for the KG for later easier loading
        print("Loaded triples from", triple_filename, ". #triples, #ents, #rels:", len(self.triples), self.n_ents, self.n_rels)
        print("saving triples")
        # triples = np.array(self.triples)
        triple_filename = triple_filename.replace("/", "_")
        os.makedirs("data/tables/", exist_ok=True)        
        with open(f"data/tables/{triple_filename}_ent2idx.pickle", "wb") as f:
            pickle.dump(self.index_ents, f)
        with open(f"data/tables/{triple_filename}_rel2idx.pickle", "wb") as f:
            pickle.dump(self.index_rels, f)        
        with open(f"data/tables/{triple_filename}_triples_train.pickle", "wb") as f:
            pickle.dump(self.triples, f)
        print("saved index dicts.")

        

    # use it for loading training kg (test kg has different behavior)
    def load_triples_train(self, 
                     triple_filename,
                     ontology_filename,
                     splitter = '\t', line_end = '\n'):

        # process standard triple files. 
        for line in open(triple_filename):
            line = line.rstrip(line_end).split(splitter)
            h = self.index_ents[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_ents[line[2]]
            self.train_triples.append([h, r, t])
            self.train_triples_record.add((h, r, t))

        # process ontology based triples
        for line in open(ontology_filename):
            line = line.rstrip(line_end).split(splitter)
            h = self.index_ents[line[0]]
            t = self.index_ents[line[2]]
            self.train_triples.append([h, self.onto_rel_index, t])
            self.train_triples_record.add((h, self.onto_rel_index, t))

        # self.train_triples = np.array(self.train_triples)
        # save 
        os.makedirs("data/tables/", exist_ok=True)
        triple_filename = triple_filename.replace("/", "_")
        with open(f"data/tables/{triple_filename}_triples_train.pickle", "wb") as f:
            pickle.dump(self.train_triples, f)


    # To load test kg, we use the same index as before; the difference is that 
    # we don't do any negative sampling and the triples are simply stored as index triples. 
    def load_triples_test(self, 
                          triple_filename,
                          ontology_filename,
                          splitter = '\t', 
                          line_end = '\n'):
        
        # process standard triple files. 
        for line in open(triple_filename):
            line = line.rstrip(line_end).split(splitter)
            h, r, t = self.index_ents[line[0]], self.index_rels[line[1]], self.index_ents[line[2]]
            self.test_triples_record.add((h, r, t))
            self.test_triples.append([h,r,t])
        
        for line in open(ontology_filename):
            line = line.rstrip(line_end).split(splitter)
            h, t = self.index_ents[line[0]], self.index_ents[line[2]]
            self.test_triples_record.add((h, self.onto_rel_index, t))
            self.test_triples.append([h,self.onto_rel_index,t])
        
        # self.test_triples = np.array(self.test_triples)
        os.makedirs("data/tables/", exist_ok=True)
        triple_filename = triple_filename.replace("/", "_")
        with open(f"data/tables/{triple_filename}_triples_test.pickle", "wb") as f:
            pickle.dump(self.test_triples, f)

    """
    Functions that require further work (customization) after contexts become available
    - load_word2vec
    - load_descriptions
    - map_descriptions 
    """

    # load word embeddings from contexts, if available
    def load_word2vec(self, filepath, splitter=' '):
        self.tokens, emb = [], []
        for lineno, l in tqdm(enumerate(open(filepath)), desc='load word embedding', unit=' word'):
            tokens = l.strip().split(splitter)
            if lineno == 0:
                dim = int(tokens[1])
                emb.append(np.zeros(dim))
                self.tokens.append('  ')
                continue
            if len(tokens) == 1 + dim:
                self.tokens.append(tokens[0])
                emb.append([float(_) for _ in tokens[1:]])
        self.wv = np.array(emb)
        self.wv_dim = dim
        self.token_index = {w:i for i, w in enumerate(self.tokens)}
        self.loaded_wv = True
        self.n_tokens = len(self.tokens)
        print("Loaded token embeddings from",filepath)



    def load_descriptions(self, titlefile, tokenfile, splitter=' ', desc_length = 100, lower=True, stop_words=None, padding_front=False):
        if self.loaded_wv == False:
            print ("Fail: Load word embeddings first")
            return
        self.desc_length = desc_length
        self.n_descriptions = 0
        titles = {}
        index = 0
        for line in open(titlefile):
            line = line.strip()
            titles[index] = line
            index += 1
        index = 0
        remove = None
        if not (stop_words is None):
            remove = set(stop_words)
        self.desc_index = []
        avg_length = 0.
        max_length = -1.
        max_sen = ""
        for line in open(tokenfile):
            title = titles[index]
            index += 1
            ent_id = self.ent_str2index(title)
            if ent_id == None:
                continue
            if not self.descriptions.get(ent_id) is None:
                continue
            if lower:
                line = line.strip().lower().split(splitter)
            else:
                line = line.strip().split(splitter)
            line = title.replace('(','').replace(')','').split(' ') + line + title.replace('(','').replace(')','').split(' ')
            desc_word_index = []
            for word in line:
                if (remove is None) or (word not in remove):
                    this_wd_index = self.word_str2index(word, default=False)
                    if not this_wd_index is None:
                        desc_word_index.append(this_wd_index)
            if len(desc_word_index) == 0:
                continue
            self.descriptions[ent_id] = np.array(desc_word_index)
            self.n_descriptions += 1
            avg_length = (avg_length * (self.n_descriptions - 1) + len(desc_word_index)) / self.n_descriptions
            if len(desc_word_index) > max_length:
                max_length = len(desc_word_index)
                max_sen = line
            desc_embed = []
            for i in desc_word_index:
                vec = self.wv[i]
                desc_embed.append(vec)
            self.avg_embed[ent_id] = np.average(desc_embed, axis = 0)
            if len(desc_embed) > desc_length:
                desc_embed = np.array(desc_embed)[:desc_length]
            elif len(desc_embed) < desc_length:
                if not padding_front:
                    #for t in range(len(desc_embed), desc_length):
                    #    desc_embed.append(np.zeros(self.wv_dim))
                    #no longer use zero padding
                    gap = desc_length - len(desc_embed)
                    desc_embed += desc_embed * int(gap/len(desc_embed)) + desc_embed[: int(gap % len(desc_embed))]
                    desc_embed = np.array(desc_embed, dtype=np.float32)
                else:
                    #zero padding to the front
                    front = []
                    for t in range(len(desc_embed), desc_length):
                        front.append(np.zeros(self.wv_dim))
                    desc_embed = front + desc_embed
                    desc_embed = np.array(desc_embed, dtype=np.float32)
            self.desc_embed[ent_id] = desc_embed
            self.desc_index.append(ent_id)
        self.desc_index = np.array(self.desc_index)
        print("Loaded descriptions from", tokenfile, ":", self.n_descriptions)
        print("AVG LEN=", avg_length, '\n', "MAX LEN=", max_length)
        print(max_sen)
        self.desc_embed_padded = []
        self.avg_embed_padded = []
        for i in range(self.num_ents()):
            vec = self.desc_embed.get(i)
            avg_vec = self.avg_embed.get(i)
            assert((vec is None and avg_vec is None) or (not (vec is None) and not (avg_vec is None)))
            if vec is None:
                vec = np.zeros((self.desc_length, self.wv_dim))
                avg_vec = np.zeros(self.wv_dim)
            self.desc_embed_padded.append(vec)
            self.avg_embed_padded.append(avg_vec)
        self.desc_embed_padded = np.array(self.desc_embed_padded, dtype=np.float32)#np.reshape(np.array(self.desc_embed_padded), [-1, self.desc_length * self.wv_dim, 1]) 
        self.avg_embed_padded = np.array(self.avg_embed_padded, dtype=np.float32)
        assert (not np.any(np.isnan(self.desc_embed_padded)))
        print("Padded desc embeddings to", self.desc_embed_padded.shape)
    
    def map_descriptions(self, titlefile, tokenfile, splitter=' ', lower=True, stop_words=None, padding_front=False):
        desc_length = self.desc_length
        titles = {}
        ent_ids, desc_embed_list = [], []
        index = 0
        for line in open(titlefile):
            line = line.strip()
            titles[index] = line
            index += 1
        index = 0
        remove = None
        if not (stop_words is None):
            remove = set(stop_words)
        for line in open(tokenfile):
            title = titles[index]
            index += 1
            ent_id = self.ent_str2index(title)
            if ent_id == None:
                continue
            if lower:
                line = line.strip().lower().split(splitter)
            else:
                line = line.strip().split(splitter)
            line = title.replace('(','').replace(')','').split(' ') + line + title.replace('(','').replace(')','').split(' ')
            desc_word_index = []
            for word in line:
                if (remove is None) or (word not in remove):
                    this_wd_index = self.word_str2index(word, default=False)
                    if not this_wd_index is None:
                        desc_word_index.append(this_wd_index)
            if len(desc_word_index) == 0:
                continue
            desc_embed = []
            for i in desc_word_index:
                vec = self.wv[i]
                desc_embed.append(vec)
            if len(desc_embed) > desc_length:
                desc_embed = np.array(desc_embed)[:desc_length]
            elif len(desc_embed) < desc_length:
                if not padding_front:
                    gap = desc_length - len(desc_embed)
                    desc_embed += desc_embed * int(gap/len(desc_embed)) + desc_embed[: int(gap % len(desc_embed))]
                    desc_embed = np.array(desc_embed, dtype=np.float32)
                else:
                    #zero padding to the front
                    front = []
                    for t in range(len(desc_embed), desc_length):
                        front.append(np.zeros(self.wv_dim))
                    desc_embed = front + desc_embed
                    desc_embed = np.array(desc_embed, dtype=np.float32)
            desc_embed_list.append( desc_embed )
            ent_ids.append(ent_id)
        desc_embed_list = np.array(desc_embed_list, dtype=np.float32)#np.reshape(np.array(self.desc_embed_padded), [-1, self.desc_length * self.wv_dim, 1]) 
        assert (not np.any(np.isnan(desc_embed_list)))
        print("Padded desc embeddings to", desc_embed_list.shape, ". Returned indices and embedding list.")
        return ent_ids, desc_embed_list
    
    def word_str2index(self, str, default = True):
        rst = self.token_index.get(str)
        if rst == None:
            low = str.lower()
            if low != str:
                rst = self.token_index.get(str)
            if rst == None and default:
                rst = 0
        return rst

    def get_desc_embed(self, ent_id):
        return self.desc_embed.get(ent_id)
    
    def num_ents(self):
        '''Returns number of entities. 

        This means all entities have index that 0 <= index < num_ents().
        '''
        return self.n_ents

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return self.n_rels

    def num_triples(self):
        return len(self.train_triples)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels.get(rel_index)

    def ent_str2index(self, ent_str):
        '''For entity `ent_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_ents.get(ent_str)

    def ent_index2str(self, ent_index):
        '''For entity `ent_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.ents.get(ent_index)

    def rel(self):
        return np.array(range(self.num_rels()))

            
    def load_stop_words(self, filepath):
        stopwords = []
        for line in open(filepath):
            line = line.strip()
            stopwords.append(line)
        return stopwords

    def save(self, filename):
        f = open(filename,'wb')
        #self.desc_embed = self.desc_embed_padded = None
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
        
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
        print("===============\nCaution: need to reload desc embeddings.\n=====================")