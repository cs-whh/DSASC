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


def visualize_self_representation_matrix(C, label_list, save_path=None, title='Self-Representation Matrix C'):
    """
    Visualize self-representation matrix C with labels sorted.

    Args:
        C: Self-representation matrix (numpy array or torch tensor)
        label_list: List of sample labels
        save_path: Optional path to save the figure
        title: Title for the plot
    """
    import matplotlib.pyplot as plt

    # Convert to numpy array if tensor
    if torch.is_tensor(C):
        C = C.cpu().detach().numpy()

    # Get sorted indices based on labels
    sorted_idx = np.argsort(label_list)

    # Reorder matrix C according to sorted labels
    C_sorted = C[sorted_idx, :][:, sorted_idx]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(C_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Samples (sorted by label)')
    plt.ylabel('Samples (sorted by label)')

    # Add grid lines at label boundaries (optional)
    unique_labels, counts = np.unique(label_list, return_counts=True)
    cumsum = np.cumsum(counts)[:-1]
    for boundary in cumsum:
        plt.axhline(y=boundary, color='white', linewidth=1)
        plt.axvline(x=boundary, color='white', linewidth=1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()
