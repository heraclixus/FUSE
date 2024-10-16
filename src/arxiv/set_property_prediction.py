import torch
import pandas as pd
import os
import pickle
import numpy as np
import argparse
from utils import (plot_training_curve, report_performance_regression, report_performance_regression_np,
                   visualize_list_hist, correlation_plot)
from models import SimpleFuzzySet
from layers import MLP
from metrics import fuzzy_jaccard_index
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from config import (NUM_EPOCHS_MLP, HIDDEN_DIMS, EMBED_DIM)
import random

cuda_num = random.choice([i for i in range(6)])
device = "cuda:0"
root_dir = "../../data/arxiv"


# set ids start with zero as index in embedding layer
def generate_set_size_data():
    with open(f"{root_dir}/processed/arxiv_data.pkl", "rb") as f:
        data_file = pickle.load(f)
    cat2titleid = data_file["cat2titleid"]
    catids = [key for key in cat2titleid.keys()]
    set_sizes = [len(cat2titleid[id]) for id in catids]
    set_sizes = np.log(set_sizes)
    size_df = pd.DataFrame({
        "set_ids" : catids,
        "set_sizes": set_sizes
    })
    np.savez(f"{root_dir}/set_sizes.npz",X=size_df["set_ids"].to_numpy(), Y=size_df["set_sizes"].to_numpy())
    visualize_list_hist(set_sizes, root_dir, name=f"arxiv_category_sizes")


# in arxiv there are 150 categories, so in total there are at most 150 * 150 /2  pairs 
# so we subsample
def generate_overlap_data():
    overlaps = []
    with open(f"{root_dir}/processed/arxiv_data.pkl", "rb") as f:
        data_file = pickle.load(f)
    cat2titleid = data_file["cat2titleid"]
    catids = [key for key in cat2titleid.keys()]
    from itertools import combinations
    pairs = combinations(catids, 2)
    for pair in pairs:
        i,j = pair
        set_i, set_j = cat2titleid[i], cat2titleid[j]
        overlap = len(set(set_i).intersection(set(set_j)))
        # for log purpose
        if overlap == 0:
            overlap += 1
        overlaps.append(overlap)
    np.savez(f"{root_dir}/set_overlaps.npz", X=np.array(pairs), Y=np.log(overlaps))
    visualize_list_hist(np.log(overlaps), root_dir, name=f"arxiv_category_overlaps")



def sklearn_loop_body(model, X_train, Y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    model.fit(rescaled_X_train, Y_train)
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    return predictions



"""
loop body to train the model using triple loss 
"""
def train_loop_body(model, dataloader_tr, dataloader_val):
    best_mse = torch.inf
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150])
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss() # use the L1 loss to deal with extreme values 
    avg_train_losses = []
    avg_val_losses = []
    for i in range(NUM_EPOCHS_MLP):
        model.train()
        model.to(device)
        total_loss = 0
        for x,y in dataloader_tr:
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss = loss_fn(y.to(device),pred.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader_tr)
        avg_train_losses.append(avg_loss)
        if i % 10 == 0:
            print(f"epoch = {i+1}, average training MSE loss per batch = {avg_loss}")
        scheduler.step()
        # validation
        total_val_loss = 0
        model.eval()
        for x,y in dataloader_val:
            pred = model(x.to(device))
            loss = loss_fn(y.to(device),pred.squeeze())
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(dataloader_val)
        if avg_val_loss <= best_mse:
            best_mse = avg_val_loss
        avg_val_losses.append(avg_val_loss)
    return avg_train_losses, avg_val_losses, best_mse


"""
Generate the pair-wise jaccard index for embeddings
use it to compute a pearson correlation with the actual jaccard index
"""
def obtain_pairwise_fuzzy_jaccard(X1, X2):
    js = []
    for i in range(len(X1)):
        fuzzy_jaccard_index = fuzzy_jaccard_index(X1[i], X2[i])
        js.append(fuzzy_jaccard_index.item())
    return js

# task 1: size prediction
# task 2: overlap prediction
# task 3: membership prediction
# task 4: coordinate prediction

def evaluate_prediction_task(model, model_name, task="size", eval_model="mlp"):
    if task == "size":
        if not os.path.exists(f"{root_dir}/set_sizes.npz"):
            generate_set_size_data()
        data = np.load(f"{root_dir}/set_sizes.npz")
    elif task == "overlap":
        if not os.path.exists(f"{root_dir}/set_overlaps.npz"):
            generate_overlap_data()
        data = np.load(f"{root_dir}/set_overlaps.npz")


    """
    description of TODO:
    1. add the simple fuzzy set model 
    2. train it for size and overlap size prediction tasks
    """
    if model_name == "fuzzy":
        model.eval()
        if task == "size":
            X = model.get_set_embedding(torch.tensor(data["X"])).detach().cpu().numpy()
        else:
            set1, set2 = data["X"][:,0], data["X"][:,1] # a pair 
            X1 = model.get_set_embedding(torch.tensor(set1)).detach().cpu()
            X2 = model.get_set_embedding(torch.tensor(set2)).detach().cpu()
            X = torch.mul(X1,X2).numpy() # use the elementwise multiplication as intersection
    else: # graph based model, embedding of hybrid model is the same embedding format
        set_ids = data["X"]
        if task == "size":
            X = np.array([model[set_id] for set_id in set_ids])
        else: # for now, just overlap
            set1, set2 = set_ids[:,0], set_ids[:,1] # a pair 
            X1 = np.array([model[set_id] for set_id in set1])
            X2 = np.array([model[set_id] for set_id in set2])
            X = np.concatenate([X1, X2], axis=-1)
    print(f" X shape = {X.shape}")
    Y = data["Y"]
    train_size = int(len(Y) * 0.7)
    val_size = int(len(Y) * 0.2)
    te_size = int(len(Y) * 0.1)
    X_tr, Y_tr = torch.Tensor(X[:train_size]), torch.Tensor(Y[:train_size])
    X_val, Y_val = torch.Tensor(X[train_size:train_size+val_size]), torch.Tensor(Y[train_size:train_size+val_size])
    X_te, Y_te = torch.Tensor(X[-te_size:]), torch.Tensor(Y[-te_size:])
    print(f"training data shape X = {X_tr.shape}, Y = {Y_tr.shape}")
    print(f"evaluation data shape X = {X_val.shape}, Y = {Y_val.shape}")
    print(f"testing data shape X = {X_te.shape}, Y = {Y_te.shape}")
    dataset_tr = TensorDataset(X_tr, Y_tr)
    dataset_val = TensorDataset(X_val, Y_val)
    dataloader_tr = DataLoader(dataset_tr, batch_size=256)
    dataloader_val = DataLoader(dataset_val, batch_size=256)

    if eval_model == "mlp":
        best_model = None
        best_mse_g = torch.inf
        best_records_avg_tr, best_records_avg_val = [], []
        best_config = []
        for hidden_dims in HIDDEN_DIMS:
            mlp = MLP(input_dim=X.shape[-1], hidden_dims=hidden_dims, device=device)
            avg_train_losses, avg_val_losses, best_mse = train_loop_body(mlp, dataloader_tr, dataloader_val)
            if best_mse <= best_mse_g:
                best_mse_g = best_mse
                best_model = mlp
                best_records_avg_tr = avg_train_losses
                best_records_avg_val = avg_val_losses
                best_config = hidden_dims
        plot_training_curve(best_records_avg_tr, best_records_avg_val, option, model_name, root, args)
        # report test performance
        best_model.eval()
        best_model.to("cpu")
        te_pred = best_model(X_te)
        print(f"best configuration of hidden channels = {best_config}")
        print(f"best validation mse = {best_mse_g}")
        report_performance_regression(Y_te, te_pred, mode="test", config=model_name)
    else: # sklearn based models 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
        pipelines = []
        pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
        pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
        pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
        results = []
        names = []
        best_mse = np.inf
        best_model = None
        for name, model in pipelines:
            kfold = KFold(n_splits=5, random_state=21, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
            if cv_results.mean() <= best_mse:
                best_mse = cv_results.mean()
                best_model = name
        print(f"best cross validation mse = {best_mse}, best model = {best_model}")
        if best_model == "ScaledLR":
            model = LinearRegression()
        elif best_model == "ScaledLASSO":
            model = Lasso()
        else:
            model = GradientBoostingRegressor(n_estimators=200)
        predictions = sklearn_loop_body(model, X_train, Y_train, X_test)
        report_performance_regression_np(Y_test, predictions, mode="test", config=model_name)


def obtain_logic_feature():
    pass 



if __name__ == "__main__":
    # uniform/proximity/enhanced, graph/logic/hybrid, mlp/sklearn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="graph") # graph, logic, hybrid 
    parser.add_argument("--embed_dim",type=int, default=EMBED_DIM)
    parser.add_argument("--walk_length", type=int, default=5)
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="proximity") # add an option of cora
    parser.add_argument("--n_sets", type=int, default=100)
    parser.add_argument("--n_statements", type=int, default=500)
    parser.add_argument("--n_elems", type=int, default=5000)
    parser.add_argument("--large", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enhanced", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_pair", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_def", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--gen_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_model", type=str, default="mlp")
    args = parser.parse_args()

    if args.gen_only: # generate data only
        generate_overlap_data()
        generate_set_size_data()
        exit(0)

    # select dataset 
    dataset = args.dataset
    model_name = args.model
    suffix = "_enhanced" if args.enhanced else ""
    prefix = "lp_" if args.lp else ""
    suffix_l = "_lp" if args.lp else ""
    if args.use_pair:
        suffix += "_use_pair"
    if args.use_def: # use the L1 loss to deal with extreme values 
        suffix += "_use_def"
    if args.model == "graph":
        if args.lp:
            model = np.load(f"{root_dir}/models/{prefix}graph_emb_{dataset}_{args.embed_dim}.npy")
        else:
            model = np.load(f"{root_dir}/models/{dataset}_graph_model_{args.walk_length}_{args.embed_dim}.npy")
    elif args.model == "enhanced":
        if args.lp:
            model = np.load(f"{root_dir}/models/{prefix}graph_emb_{dataset}_enhanced_{args.embed_dim}.npy")
        else:
            model = np.load(f"{root_dir}/models/{dataset}_graph_model_enhanced_{args.walk_length}_{args.embed_dim}.npy")
    elif args.model == "hybrid":
        model = np.load(f"{root_dir}/models/hybrid_emb_{dataset}_enhanced_{args.embed_dim}.npy")
    else: # logic
        model = FuzzyEmbedder(embed_dim=args.embed_dim, n_elements=args.n_elems, n_sets=args.n_sets+args.n_statements, device="cpu")
        model.load_state_dict(torch.load(f"{root_dir}/models/{dataset}_logic_model{suffix_l}.pt"))
    print(f"evaluate the {model_name} model")
    evaluate_prediction_task(model, model_name, dataset, args=args, root=root_dir, task="size", eval_model=args.eval_model)
