import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import torch_geometric as torchgeo
from model import *
from sklearn.metrics import normalized_mutual_info_score
import random
nmi = normalized_mutual_info_score


def cluster_accuracy(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def load_pretrain_model(name, dataset):
    if name == 'dino_vits8':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    elif name == 'dino_vits16':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    elif name == 'dino_vitb8':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bulid_graph(features, K):
    sparse_adj = []
    for i in range(len(features)):
        sparse_adj.append(kneighbors_graph(features[i].cpu().numpy(), K, mode='connectivity', metric='cosine'))
    return sparse_adj


def bulid_pyg_data(features, sparse_adj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas = []
    for i in range(len(features)):
        pyg_graph = torchgeo.data.Data()
        pyg_graph.x = features[i].to(device)
        edge_index = torch.from_numpy(np.transpose(np.stack(sparse_adj[i].nonzero(), axis=1))).to(device)
        pyg_graph.edge_index = edge_index
        pyg_graph.edge_index = torchgeo.utils.to_undirected(pyg_graph.edge_index)
        pyg_graph.num_nodes = features[i].shape[0]
        datas.append(pyg_graph)

    return datas
