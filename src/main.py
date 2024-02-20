import time
import torch
import argparse
from  pre_process import *
from utils import *
from exp import Experiments
from utils import print_local_time


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


# baseline 
parser.add_argument("--run_gumbel_box", action="store_true")
parser.add_argument("--run_soft_box", action="store_true")
parser.add_argument("--softbox_temp", type=float, default=0.1)
parser.add_argument("--gumbel_beta", type=float, default=0.1)
parser.add_argument("--box_score_type", type=str, default="ratio")
parser.add_argument("--box_score_mode", type=str, default="whole")


## configurations for Simple Fuzzy Set 
parser.add_argument("--use_fuzzyset", action="store_true") # flag used to decide whether to use SFS or box
parser.add_argument("--use_fuzzqe", action="store_true") # flag used to decide whether to use fuzzQE under SFS 

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

start_time = time.time()
print ("Start time at : ")
print_local_time()

args = parser.parse_args()
args.cuda = True if torch.cuda.is_available() and args.cuda else False
if args.cuda:
    torch.cuda.set_device(args.gpu_id)

print (args)

create_data(args)

exp = Experiments(args)

if not args.test_only: 
    """Train the model"""
    exp.train()
if args.run_gumbel_box or args.run_soft_box:
    exp.predict_fuzzy_box()  
elif args.use_fuzzyset:
    exp.predict_fuzzy()
    # exp.save_prediction_fuzzy()
else:
    exp.predict()
    exp.save_prediction()

print ("Time used :{:.01f}s".format(time.time()-start_time))
print ("End time at : ")
print_local_time()
print ("************END***************")


