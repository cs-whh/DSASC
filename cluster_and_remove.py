import torch
import numpy as np
from sklearn.cluster import KMeans
from myutil import load_pretrain_model, cluster_accuracy, setup_seed
from data import cifar_dataset
from torch.utils.data import DataLoader
import copy
import types

setup_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading DINO model...")
model = load_pretrain_model('dino_vits8', 'cifar10')
model.to(device)

def get_middle_feature(self, x):
    x = self.prepare_tokens(x)
    features = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in [2, 5, 8]:
            features.append(self.norm(x)[:, 0])
    features.append(self.norm(x)[:, 0])
    return features

model.get_middle_feature = types.MethodType(get_middle_feature, model)

print("Loading CIFAR-10 dataset...")
img_size = 80
dataset = cifar_dataset(img_size)
dl = DataLoader(dataset, batch_size=100, shuffle=False)

print("Extracting DINO features...")
model.eval()
features = []
y_true = torch.empty((0,))
for i, (X, y) in enumerate(dl):
    if i == 100:
        break
    X = X.to(device)
    y_true = torch.cat((y_true, y), dim=0)
    with torch.no_grad():
        ls = model.get_middle_feature(X)
    if features == []:
        features = copy.deepcopy(ls)
    else:
        for j in range(len(features)):
            features[j] = torch.cat((features[j], ls[j]), dim=0)

features_final = features[-1].cpu().numpy()
y_true = y_true.numpy()
print(f"Extracted features shape: {features_final.shape}")
print(f"Labels shape: {y_true.shape}")

print("\nPerforming K-means clustering (k=10)...")
kmeans = KMeans(n_clusters=10, random_state=100, n_init=10)
y_pred = kmeans.fit_predict(features_final)

print("Computing initial clustering accuracy...")
initial_acc = cluster_accuracy(y_true, y_pred)
print(f"Initial clustering accuracy: {initial_acc:.4f}")

print("\nRemoving samples farthest from cluster centers...")
centroids = kmeans.cluster_centers_
remove_indices = []

for cluster_id in range(10):
    cluster_mask = (y_pred == cluster_id)
    cluster_features = features_final[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]

    distances = np.linalg.norm(cluster_features - centroids[cluster_id], axis=1)
    sorted_indices = np.argsort(distances)

    n_remove = min(500, len(sorted_indices))
    farthest_indices = sorted_indices[-n_remove:]
    remove_indices.extend(cluster_indices[farthest_indices])

remove_indices = np.array(remove_indices)
keep_mask = np.ones(len(y_true), dtype=bool)
keep_mask[remove_indices] = False

print(f"Total samples removed: {len(remove_indices)}")
print(f"Total samples remaining: {keep_mask.sum()}")

print("\nComputing accuracy after removing farthest samples...")
y_true_remaining = y_true[keep_mask]
y_pred_remaining = y_pred[keep_mask]

remaining_acc = cluster_accuracy(y_true_remaining, y_pred_remaining)
print(f"Remaining samples clustering accuracy: {remaining_acc:.4f}")

print("\n" + "="*50)
print(f"Summary:")
print(f"  Total samples: {len(y_true)}")
print(f"  Initial accuracy: {initial_acc:.4f}")
print(f"  Removed samples: {len(remove_indices)}")
print(f"  Remaining samples: {keep_mask.sum()}")
print(f"  Remaining accuracy: {remaining_acc:.4f}")
print("="*50)
