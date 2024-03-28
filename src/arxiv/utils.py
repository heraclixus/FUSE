from sklearn import mixture
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
import matplotlib.style as style
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn.functional as F
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import powerlaw
import pyparsing as pp
import torch
from numpy.linalg import norm
import random
from config import (STD_DIR_MULTIPLIER, STD_DIR_MIN, N_ELEMENTS, N_SETS, N_STATEMENTS,
                    MIN_RAD_MULTIPLIER ,SIZE_MULTIPLIER, POWER_LAW_COEFF, N_PAIR_ELEMS,
                    FILE_DIR, FILE_LOGIC_DIR)

###############################################
###############################################
#############   Training       ################
###############################################
###############################################
# ways to combine node features for link prediction
def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

binary_operators = [operator_avg]

def evaluate_not(set1, n_elements):
    sample_space = [i for i in range(n_elements)] # sample space of elements
    return list(set(sample_space) - set(set1))
""" 
sample positive and negative elems to compute triplet margin loss
"""
def sample_negative_elems(set_i, n_elements, k=1):
    n_samples = len(set_i) * k
    negative_elems = evaluate_not(set_i, n_elements)
    negatives = np.random.choice(negative_elems, size=n_samples, replace=False)
    return negatives


# compute margin loss 
def compute_margin_loss(positive_score, negative_score, gamma=0.5, gamma_coff=20):
    positive_dist = 1-positive_score
    negative_dist = 1-negative_score
    positive_unweighted_loss = -F.logsigmoid((gamma - positive_dist)*gamma_coff).squeeze(dim=1)
    negative_unweighted_loss = -F.logsigmoid((negative_dist - gamma)*gamma_coff).mean(dim=1)
    positive_sample_loss = positive_unweighted_loss.sum()
    negative_sample_loss = negative_unweighted_loss.sum()
    loss = (positive_sample_loss + negative_sample_loss) / 2
    return loss


###############################################
###############################################
#############   Data           ################
###############################################
###############################################
# generate a list of length n_sets of set sizes 
# sizes should follow a power law distribution.
def generate_powerlaw_sizes(num_samples, n_sets, root_dir, size=True, power=POWER_LAW_COEFF):
    r = powerlaw.rvs(power, loc=0, scale=int(num_samples * SIZE_MULTIPLIER), size=n_sets)
    r2 = num_samples * SIZE_MULTIPLIER - r
    r3 = r2
    print(np.sum(r3))
    return np.rint(r3).astype(int) if size else np.array(r3)


def plot_size_distribution(sizes, num_samples, n_sets, root_dir, dist, power=None):
    fig, ax = plt.subplots(1, 1)
    # ax.hist(sizes, bins=30, density=False, histtype='stepfilled', alpha=0.2)
    sns.histplot(sizes, bins=30, kde=True)
    plt.title(f"Distribution of set size: {num_samples} elems, {n_sets} sets")
    plt.xlabel("Set size")
    plt.ylabel("Frequency")
    plt.savefig(f"{root_dir}/visualizations/{dist}_set_size_distribution_{num_samples}_{n_sets}.png")


def generate_powerlaw_radius(num_samples, n_sets, max_coord, min_coord, root_dir):
    max_rad = norm(max_coord - min_coord)
    r3 = generate_powerlaw_sizes(num_samples, n_sets, False, False)
    multiples = np.max(r3) / max_rad
    rads = r3 / multiples
    fig, ax = plt.subplots(1, 1)
    ax.hist(rads, density=True, histtype='stepfilled', alpha=0.2)
    plt.title("Distribution of circle radius")
    plt.xlabel("Radius")
    plt.ylabel("Frequency")
    plt.savefig(f"{root_dir}/visualizations/radius_circle_distribution_{num_samples}_{n_sets}.png")
    return rads

# generate circles using randomly sampled center
# or centers close to the modes 
# and radius following a power law distribution 
def generate_circle(min_coord, max_coord, var, radius=None):
    center_x1 = np.random.uniform(min_coord[0], max_coord[0])
    center_x2 = np.random.uniform(min_coord[1], max_coord[1])
    center = np.array([center_x1, center_x2])
    if radius == None:
        radius = np.random.uniform(var * STD_DIR_MIN, var * STD_DIR_MULTIPLIER)
    return center, radius

# find the intersection of the circle with the entire sample space
def find_circle_intersection(center, radius, samples):
    intersections = []
    for i in range(len(samples)):
        d2c = np.sqrt(norm(samples[i] - center))
        if d2c <= radius:
            intersections.append(i)
    return list(set(intersections))


def print_graph_sample(data):
    print(data)
    edge_indices, edge_labels, edge_label_indices = data.edge_index, data.edge_label, data.edge_label_index
    print(f"edge index shape = {edge_indices.shape}")
    print(f"edge label index = {edge_label_indices}, shape = {edge_label_indices.shape}")
    print(f"edge labels = {edge_labels}, shape = {edge_label_indices.shape}")
    print(f"number of ones = {torch.count_nonzero(edge_labels)}, number of zeros = {torch.numel(edge_labels)-torch.count_nonzero(edge_labels)}")

# utility to separate the positive edges and the negative edges,
# use edge_labels and edge_label_indices to enforce this 
def obtain_pos_neg_rw(data):
    edge_labels, edge_label_indices = data.edge_label, data.edge_label_index
    idx_zeros = (edge_labels == 0).nonzero() # indices of zeros
    idx_ones = (edge_labels != 0).nonzero() # indices of ones
    positive_edges = edge_label_indices[:, idx_ones].squeeze(-1)
    negative_edges = edge_label_indices[:, idx_zeros].squeeze(-1)
    return positive_edges, negative_edges

def get_link_labels(pos_edge_index, neg_edge_index, device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def obtain_pos_neg_labels(data):
    edge_labels = data.edge_label
    return torch.count_nonzero(edge_labels).cpu().numpy(), torch.numel(edge_labels)-torch.count_nonzero(edge_labels).cpu().numpy()


# convert edgelist to set 
# used by the fuzzy logic model, turn train graph split into a set representation
# this is used explicitly by the set definition loss
def convert_edge_to_set(edgelist, n_elements):
    set_elems = {}
    for i in range(edgelist.shape[-1]):
        node1, node2 = edgelist[0,i].item(), edgelist[1,i].item()
        if node1 >= n_elements: # this means it is a set
            if node1 not in set_elems.keys():
                set_elems[node1] = [node2]
            else:
                set_elems[node1].append(node2)
        elif node2 >= n_elements: # node2 is set
            if node2 not in set_elems.keys():
                set_elems[node2] = [node1]
            else:
                set_elems[node2].append(node1)
    return set_elems # {set_id: [elem_id]}



# utility to convert the edgelist in to embedding in fuzzy logic case 
# necessary since there are two embedding layers 
def convert_edges_to_fuzzy_emb(data, model, device, n_elements):
    nodes1, nodes2 = [],[]
    for i in range(len(data)):
        pair = data[i]
        node1, node2 = pair[0], pair[1]
        if node1 >= n_elements:
            nodes1.append(model.get_set_embedding(torch.tensor([node1-n_elements]).to(device)))
        else:
            nodes1.append(model.get_elem_embedding(torch.tensor([node1]).to(device)))
        if node2 >= n_elements: 
            nodes2.append(model.get_set_embedding(torch.tensor([node2-n_elements]).to(device)))
        else:
            nodes2.append(model.get_elem_embedding(torch.tensor([node2]).to(device)))
    return torch.stack(nodes1).squeeze(), torch.stack(nodes2).squeeze()


###############################################
###############################################
#############   Logic          ################
###############################################
###############################################

# parse logical expressions using two stacks
# also returns if the first item in expression is an operand
def prepare_logical_expression(expression):
    expression = expression.split(" ")
    stack_operands, stack_operators = [],[]
    for i in range(len(expression)):
        op = expression[i]
        if op.isdigit():
            stack_operands.append(int(op))
        else:
            stack_operators.append(op)
    return stack_operators, stack_operands, expression[0].isdigit()


###############################################
###############################################
#############   Visualization  ################
###############################################
###############################################
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr
seed_colors = [key for key in mcolors.TABLEAU_COLORS.keys()]


# plot correlation
def correlation_plot(x, y):
    df = pd.DataFrame({
        "Fuzzy Jaccard Index": x,
        "Jaccard Index": y
    })
    corr = pearsonr(x, y)
    print(corr)
    sns.lmplot(x="Fuzzy Jaccard Index", y="Jaccard Index", data=df)
    plt.title(f"Pearson correlation between x and y is {corr}")
    plt.savefig("visualizations/correlation_jaccardindex.png")

"""
visualize a random set of elements and their embedding
the elements are in the 2-D space with their original coordinate
the TSNE embeddiing are plotted in a 2-D space obtained from their element embeddings
"""
def visualize_elements_TSNE_elems(model_name, dataset_name, n_elements=10, enhanced=True):
    tsne_emb = TSNE(random_state=1, n_iter=15000, learning_rate="auto", init="pca", metric="cosine")
    original_samples = np.load("samples.npy")
    random_elem_ids = random.sample([i for i in range(len(original_samples))], n_elements)
    samples = original_samples[random_elem_ids, :] # chosen samples
    suffix = "_enhanced" if enhanced else ""
    embeddings = np.load(f"models/{model_name}_emb_{dataset_name}{suffix}.npy")
    embeddings = tsne_emb.fit_transform(embeddings)
    embedding = embeddings[random_elem_ids, :]
    print(samples.shape)
    print(embedding.shape)
    colors = seed_colors[:n_elements]
    fig, axes = plt.subplots(1,2, figsize=(15,15))
    axes[0].scatter(samples[:,0], samples[:,1], c=colors)
    axes[0].set_title(f"scatter plot {n_elements} original samples")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[1].scatter(embedding[:,0], embedding[:,1], c=colors)
    axes[1].set_title(f"scatter plot {n_elements} TSNE of embeddings")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.savefig(f"visualizations/element_sample_vs_TSNE_{model_name}_{dataset_name}.png")


""" 
Function used to visualize a generated set
plot all the 2-d points in a scatter plot, with an initial color
sets are plotted using differnt colors 
"""

def obtain_colors(root_dir, sets, n_sets):
    original_samples = np.load(f"{root_dir}/samples.npy")
    colors = ["tab:gray"] * len(original_samples) # ambient color
    idx = 1
    while idx <= n_sets:
        set_i = sets.iloc[idx-1]
        color_i = seed_colors[idx]
        for j in range (len(set_i)):
            colors[set_i[j]] = color_i
        idx += 1 
    return original_samples, colors

def obtain_vis_sets(dataset_name, root_dir, n_sets=1):
    sets = pd.read_pickle(f"{root_dir}/{dataset_name}_generated_sets.pickle")
    sets["set_sizes"] = sets["set_elements"].apply(len)
    sets = sets.sort_values("set_sizes", ascending=False).head(n_sets)["set_elements"]
    return sets

def visualize_generated_sets(dataset_name, root_dir, n_sets=4):
    sets = obtain_vis_sets(dataset_name, root_dir, n_sets)
    original_samples, colors = obtain_colors(root_dir, sets, n_sets)
    plt.figure(figsize=(15,15))
    plt.scatter(original_samples[:,0], original_samples[:,1], c=colors)
    plt.title(f"scatter plot elements in {n_sets} original sets")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(f"{root_dir}/visualizations/set_element_sample_{dataset_name}.png")    

def visualize_generated_sets_pair(root_dir, n_sets=4):
    sets_unif = obtain_vis_sets("uniform", root_dir, n_sets)
    sets_prox = obtain_vis_sets("proximity", root_dir, n_sets)
    sets_inter = obtain_vis_sets("intermediate", root_dir, n_sets)

    _, colors_unif = obtain_colors(root_dir, sets_unif, n_sets)
    original_samples, colors_prox = obtain_colors(root_dir, sets_prox, n_sets)
    original_samples, colors_inter = obtain_colors(root_dir, sets_inter, n_sets)

    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].scatter(original_samples[:,0], original_samples[:,1], c=colors_prox)
    axes[0].set_title(f"scatter plot proximally-grouped elements")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[2].scatter(original_samples[:,0], original_samples[:,1], c=colors_unif)
    axes[2].set_title(f"scatter plot uniformly-grouped elements")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[1].scatter(original_samples[:,0], original_samples[:,1], c=colors_inter)
    axes[1].set_title(f"scatter plot intermediately-grouped elements")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.savefig(f"{root_dir}/visualizations/set_element_sample_pair.png")    



"""
visualize set elements by first randomly sample a bunch of sets and then plot their elements
elements belonging to the same set are assigned the same color
"""
def visualize_elements_TSNE_set_elems(model_name, dataset_name, n_sets=5, enhanced=True):
    suffix = "_enhanced" if enhanced else ""
    tsne_emb = TSNE(random_state=1, n_iter=15000, learning_rate="auto", init="pca", metric="cosine")
    original_samples = np.load("samples.npy")
    sets = pd.read_pickle(f"{dataset_name}_generated_sets.pickle").sample(n_sets, random_state=1)["set_elements"]
    embeddings = np.load(f"models/{model_name}_emb_{dataset_name}{suffix}.npy")
    embeddings = tsne_emb.fit_transform(embeddings)
    set_sizes = [len(item) for item in sets]
    colors, set_ids, samples, embedding = [],[],[],[]
    for i in range(n_sets):
        colors.extend([seed_colors[i]] * set_sizes[i])
    for set_elems in sets:
        set_ids.extend(set_elems.tolist())
    for set_id in set_ids:
        samples.append(original_samples[set_id])
        embedding.append(embeddings[set_id])
    samples = np.stack(samples)
    embedding = np.stack(embedding)
    print(samples.shape)
    print(embedding.shape)
    assert len(samples) == len(embedding)
    assert len(embedding) == len(colors)
    fig, axes = plt.subplots(1,2, figsize=(15,15))
    axes[0].scatter(samples[:,0], samples[:,1], c=colors)
    axes[0].set_title(f"scatter plot elements in {n_sets} original sets")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[1].scatter(embedding[:,0], embedding[:,1], c=colors)
    axes[1].set_title(f"scatter plot {n_sets} sets TSNE of embeddings")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.savefig(f"visualizations/set_element_sample_vs_TSNE_{model_name}_{dataset_name}.png")

"""
Plot the training vs. validation losses for a specific
model configuration
"""
def plot_training_curve(losses_tr, losses_val, option, model_name, root, args):
    assert len(losses_tr) == len(losses_val)
    suffix = "_lp" if args.lp else ""
    suffix += "_enhanced" if args.enhanced else suffix
    n_epochs = [i for i in range(len(losses_tr))]
    plt.figure()
    plt.style.use('seaborn')
    plt.plot(n_epochs, losses_tr, label = 'Training MSE per minibatch')
    plt.plot(n_epochs, losses_val, label = 'Validation MSE per minibatch')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Epochs', fontsize = 14)
    plt.title(f'Learning curves for MLP evaluation for {model_name}', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(f"{root}/visualizations/mlp_training_visualization_{option}_{model_name}{suffix}.png")

"""
Plot the training loss only
model configuration
"""
def plot_training_loss(losses_tr, model_name, root, args):
    n_epochs = [i for i in range(len(losses_tr))]
    plt.figure()
    plt.style.use('seaborn')
    plt.plot(n_epochs, losses_tr, label = 'Training MSE per minibatch')
    plt.ylabel('Logsimoid margin loss', fontsize = 14)
    plt.xlabel('Epochs', fontsize = 14)
    plt.title(f'Learning curves for {model_name}', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(f"{root}/visualizations/mlp_training_visualization_{model_name}_{args.fuzzy_mode}_{args.logic_mode}_{args.data_mode}.png")





"""
Visualize several sets generated by circles
"""
def visualize_circles(circles, samples, root_dir, option="seed", universe="GMD", sampling=True):
    plt.style.use('seaborn') #sets the size of the charts
    colors = {0.0: "tab:gray", 1.0: "tab:gray", 2.0: "tab:gray", 
              3.0: "aquamarine", 4.0: "blue"}
    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="tab:gray")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("GMD Samples and Circles")
    ax.legend()
    if len(circles) > 10 and sampling:
        circles = random.sample(circles, 10) # randomly draw 10 circles
    for i in range(len(circles)):
        circle = plt.Circle(xy=circles[i][0], radius=circles[i][1], fill=False)
        ax.add_patch(circle)
        circle.set_clip_box(ax.bbox)
        circle.set_edgecolor("tab:orange")
        circle.set_facecolor("none") 
        circle.set_alpha(1)
    plt.savefig(f"{root_dir}/visualizations/{option}_circles_{universe}.png")


# visualize a list, used to see distribution of downstream target sizes
def visualize_list_hist(lst, root, name="overlap"):
    fig, ax = plt.subplots(1, 1)
    ax.hist(lst, bins=30, density=False, histtype='stepfilled', alpha=0.2)
    plt.title("Distribution of set size")
    plt.xlabel("Set size")
    plt.ylabel("Frequency")
    plt.savefig(f"{root}/visualizations/{name}.png")

# visualize AUC under different configurations
# using a 3D plot
def visualize_auc(df_name):
    df = pd.read_csv(df_name)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,9))
    ax = Axes3D(fig)
    x,y,z = df["WalkLength"], df["EmbeddingDimension"], df["TestAUC"]
    ax.scatter(x,y,z)
    ax.set_xlabel("Walk Length")
    ax.set_ylabel("Embedding Dimension")
    ax.set_zlabel("Test AUC")
    ax.plot(x, z, 'r+', zdir='y', zs=1.5)
    ax.plot(y, z, 'g+', zdir='x', zs=-0.5)
    plt.savefig("AUC_plots.png")


""" 
visualize the graph's degree distribution in rank
"""
def visualize_degree_distribution(graph_root_dir, root_dir, args, g_type):
    import networkx as nx
    G = nx.read_edgelist(f"{graph_root_dir}/raw/pair_data.txt", nodetype=int)
    degrees = [np.log(d) for n, d in G.degree()]
    fig = plt.figure(figsize=(8, 8))
    fig_title = f"Degree Distribution of {args.universe}: n_elems = {args.n_elements}, n_sets = {args.n_sets}, n_statements = {args.n_statements}, mode = {args.mode}"
    fig.suptitle(fig_title, fontsize="small")
    sns.histplot(degrees, kde=True, bins=30)
    plt.xlabel("Degree")
    plt.ylabel("# of Nodes")
    fig.tight_layout()
    plt.savefig(f"{root_dir}/visualizations/Degree_distribution_{args.universe}_{args.n_elements}_{args.n_sets}_{args.n_statements}_{args.mode}_{g_type}.png")


"""
Evaluation Utilities
"""
# mode = validaiton/test, config = logic-enhanced/original
def report_performance_regression(target, pred, mode, config):
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    print(f"{mode} evaluation MAPE = {mean_absolute_percentage_error(target.detach().numpy(),pred.detach().numpy())} on {config} data")
    print(f"{mode} evaluation MAE = {mean_absolute_error(target.detach().numpy(),pred.detach().numpy())} on {config} data")
    print(f"{mode} evaluation MSE = {mean_squared_error(target.detach().numpy(),pred.detach().numpy())} on {config} data")

def report_performance_regression_np(target, pred, mode, config):
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    print(f"{mode} evaluation MAPE = {mean_absolute_percentage_error(target,pred)} on {config} data")
    print(f"{mode} evaluation MAE = {mean_absolute_error(target,pred)} on {config} data")
    print(f"{mode} evaluation MAE = {mean_squared_error(target,pred)} on {config} data")


###############################################
###############################################
#############   Debugging      ################
###############################################
###############################################
"""
Sanity check: the test graph after random link split should not contain any
edges from the training
"""
def sanity_check_train_test_split(train_graph, test_graph):
    edge_indices = train_graph.edge_label_index.T
    test_indices = test_graph.edge_label_index.T
    edge_indices = [edge_indices[i] for i in range(len(edge_indices))]        
    test_indices = [test_indices[i] for i in range(len(test_indices))]
    for train_edge in edge_indices:
        for test_edge in test_indices:
            if torch.equal(train_edge, test_edge):
                print(train_edge)
                return False
    return True

""" 
Check that negative samples are indeed negative
"""
def sanity_check_negative_samples(graph_data, negative_sample):
    pass



# test code for utils.py
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="data_GMD") # GMD or Uniform
    parser.add_argument("--n_sets", type=int, default=N_SETS)
    parser.add_argument("--n_elements", type=int, default=N_ELEMENTS)
    parser.add_argument("--n_statements", type=int, default=N_STATEMENTS)
    parser.add_argument("--mode", type=str, default="uniform") # uniform proximity intermediate
    parser.add_argument("--min_rad_mult", type=int, default=MIN_RAD_MULTIPLIER) # used to decide how large the intermediate seed sets are
    args = parser.parse_args()
    option = args.mode
    root_dir = f"{args.universe}/{args.n_elements}_{args.n_sets}_{args.n_statements}"
    graph_root_dir = f"{root_dir}/{FILE_DIR}_{option}"
    graph_root_dir_logic = f"{root_dir}/{FILE_LOGIC_DIR}_{option}"
    visualize_generated_sets(args.mode, root_dir, 4)
    visualize_generated_sets_pair(root_dir, 4)
    visualize_degree_distribution(graph_root_dir, root_dir, args, "original")
    visualize_degree_distribution(graph_root_dir_logic, root_dir, args, "enhanced")