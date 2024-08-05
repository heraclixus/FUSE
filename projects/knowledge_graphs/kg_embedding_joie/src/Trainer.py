"""
wrapper class around the legacy multiG class to load data for training
"""
from KG import KG
from multiG import multiG
import numpy as np
import time
from tqdm import tqdm


"""
Trainer class thta builds up the dataloader and performs training.
two options: either pass the model in constructor or construct the model inside the constructor. 
"""
class Trainer():
    def __init__(self, 
                 model,
                 kg1_path,
                 kg2_path,
                 kgA_path,
                 kg1_batchsize, 
                 kg2_batchsize,
                 kgA_batchsize,
                 n_epochs,
                 every_n_epoch_to_save,
                 lr,
                 margin,
                 n_negative_samples=1,
                 shuffle=True
                 ):
        # attributes
        self.kg1_path = kg1_path
        self.kg2_path = kg2_path 
        self.kgA_path = kgA_path
        self.kg1_batchsize = kg1_batchsize
        self.kg2_batchsize = kg2_batchsize
        self.kgA_batchsize = kgA_batchsize
        self.n_negative_samples = n_negative_samples

        self.n_epochs = n_epochs
        self.every_n_epoch_to_save = every_n_epoch_to_save
        self.lr = lr
        self.margin = margin
        self.shuffle = shuffle

        # objects 
        self.multiG = self.setup_KG()
        self.kg1_len = self.multiG.KG1.triples.shape[0]
        self.kg2_len = self.multiG.KG2.triples.shape[0]
        self.kga_len = len(self.multiG.align)  
        self.n_batches_kg1 = int(self.multiG.KG1.num_triples() / self.kg1_batchsize)
        self.n_batches_kg2 = int(self.multiG.KG2.num_triples() / self.kg2_batchsize)
        self.n_batches_kgA = int(self.multiG.num_align() / self.kgA_batchsize)        

        # model
        self.model = model

        # optimization
        # TODO: optimizer and scheduler


    # set up multiG object
    def setup_KG(self):
        KG1, KG2 = KG(), KG() 
        KG1.load_triples(filename=self.kg1_path)
        KG2.load_triples(filename=self.kg2_path)
        this_multiG = multiG(KG1, KG2)
        this_multiG.load_align(filename=self.kgA_path, lan1="ins", lan2="onto", splitter="\t", line_end="\n")
        this_multiG.batch_sizeK1 = self.kg1_batchsize
        this_multiG.batch_sizeK2 = self.kg2_batchsize
        this_multiG.batch_sizeA = self.kgA_batchsize
        return this_multiG
    
    # set up the instance view KG minibatches
    def gen_KG_batch(self, KG_index, forever=False):
        KG = self.multiG.KG1
        triple_size = self.kg1_len
        batch_size = self.kg1_batchsize
        if KG_index == 2:
            KG = self.multiG.KG2
            triple_size = self.kg2_len
            batch_size = self.kg2_batchsize
        triples = KG.triples
        while True:
            if self.shuffle:
                np.random.shuffle(triples)
            for i in range(0, triple_size, batch_size):
                batch = triples[i: i+batch_size, :]
                if batch.shape[0] < batch_size:
                    batch = np.concatenate((batch, self.multiG.triples[:batch_size - batch.size[0]]),axis=0)
                neg_batch = KG.corrupt_batch(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break


    # set up the instance view KG minibatches
    def setup_KGA_batch(self, forever):
        align = self.multiG.align
        while True:
            if self.shuffle:
                np.random.shuffle(align)
            for i in range(0, self.kga_len, self.kgA_batchsize):
                batch = align[i: i+self.kgA_batchsize, :]
                if batch.shape[0] < self.kgA_batchsize:
                    batch = np.concatenate((batch, align[:self.kgA_batchsize-batch.shape[0]]),axis=0)
                n_batch = self.multiG.corrupt_align_batch(batch, tar=1)
                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], n_batch[:, 0], n_batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64), e1_nbatch.astype(np.int64), e2_nbatch.astype(np.int64)
            if not forever:
                break


    
    # TODO: finish the logic after fuse model implementation
    def train_one_epoch(self):
        
        this_gen_A_batch = self.gen_KM_batch(KG_index=1, batchsize=self.batch_sizeK1, forever=True)
        this_gen_B_batch = self.gen_KM_batch(KG_index=2, batchsize=self.batch_sizeK2,forever=True)
        this_gen_AM_batch = self.gen_AM_batch(forever=True)
        this_loss = []
        loss_A = loss_B = 0

        """
        TODO: design the combination of losses: instance, ontology, cross, etc. 
        """
        for batch_id in range(self.n_batches_kg1):
            A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index  = next(this_gen_A_batch) # from kg1

        for batch_id in range(self.n_batches_kg2):
            B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index  = next(this_gen_B_batch) # from kg2

        for batch_id in range(self.n_batches_kgA):
            e1_index, e2_index, e1_nindex, e2_nindex  = next(this_gen_AM_batch)
    
    # TODO: finish the loop after fuse model implementation
    def train(self):
        t_start = time.time()
        for epoch in tqdm(range(self.n_epochs)):
            self.train_one_epoch()

    

# simple tests
if __name__ == "__main__":
    trainer = Trainer(model=None,
                      kg1_path="../data/yago/yago_insnet_train.txt",
                      kg2_path="../data/yago/yago_ontonet_train.txt",
                      kgA_path="../data/yago/yago_InsType_train.txt",
                      kg1_batchsize=128,
                      kg2_batchsize=128,
                      kgA_batchsize=32,
                      n_epochs=100,
                      every_n_epoch_to_save=10,
                      lr=1e-3,
                      margin=0.2)
    print(trainer.kg1_len)
    print(trainer.kg2_len)
    print(trainer.kga_len)