import os
from datetime import datetime
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from myutil import *
from data import *
from post_clustering import *
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use('Agg')
import hydra
from omegaconf import DictConfig, OmegaConf

# Use consistent naming
cluster_accuracy = acc


class SubspaceClustering(nn.Module):
    def __init__(self, n_samples):
        super(SubspaceClustering, self).__init__()
        self.C = nn.Parameter(1.0e-8 * torch.ones(n_samples, n_samples))

    def forward(self, X):
        # Zero out diagonal to prevent self-expression during training
        C_masked = self.C - torch.diag(torch.diag(self.C))
        reconstructed = torch.mm(C_masked, X)
        return reconstructed


def train_subspace_c(X, lambda_c=1.0, lr=1e-3, epochs=500, checkpoint_epochs=None, checkpoint_callback=None):
    device = X.device
    N, D = X.shape

    model = SubspaceClustering(N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    if checkpoint_epochs is None:
        checkpoint_epochs = [epochs]

    for epoch in tqdm(range(epochs), desc="Training C"):
        reconstructed = model(X)
        recon_loss = mse_loss(X, reconstructed)
        sparsity_loss = torch.linalg.matrix_norm(model.C, 1)
        loss = recon_loss + lambda_c * sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/recon_loss": recon_loss.item(),
            "train/sparsity_loss": sparsity_loss.item(),
            "train/total_loss": loss.item()
        })

        # Checkpoint evaluation
        if (epoch + 1) in checkpoint_epochs and checkpoint_callback is not None:
            C_checkpoint = model.C.detach()
            C_checkpoint = C_checkpoint - torch.diag(torch.diag(C_checkpoint))
            checkpoint_callback(epoch + 1, C_checkpoint)

    # Final C matrix
    C = model.C.detach()
    C = C - torch.diag(torch.diag(C))

    return C


def kmeans_baseline(X, num_clusters, n_init=20):
    """
    K-means clustering baseline

    Args:
        X: feature matrix [N, D] (numpy array)
        num_clusters: number of clusters
        n_init: number of initializations

    Returns:
        y_pred: cluster labels
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=100)
    y_pred = kmeans.fit_predict(X)
    return y_pred


def subspace_clustering_baseline(X, num_clusters, labels, lambda_c=1.0, lr=1e-3, epochs=500,
                                  checkpoint_epochs=None, visualize=True, save_path=None):
    """
    Subspace clustering baseline with sklearn spectral clustering

    Args:
        X: feature matrix [N, D] (torch tensor)
        num_clusters: number of clusters
        labels: ground truth labels for visualization
        lambda_c: regularization weight
        lr: learning rate
        epochs: training epochs
        checkpoint_epochs: list of epochs to evaluate checkpoints (e.g., [1000, 3000, 5000])
        visualize: whether to visualize the self-representation matrix
        save_path: path to save visualization (if set, won't display window)

    Returns:
        results: dict mapping epoch -> (y_pred, acc, nmi)
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [epochs]

    labels_np = labels.cpu().numpy()
    results = {}

    def checkpoint_callback(epoch, C_checkpoint):
        # Apply spectral clustering at checkpoint
        C_np = C_checkpoint.cpu().numpy()
        y_pred = sklearn_spectral_clustering(C_np, num_clusters)

        # Calculate metrics
        acc_score = cluster_accuracy(labels_np, y_pred)
        nmi_score = nmi(labels_np, y_pred)

        results[epoch] = (y_pred, acc_score, nmi_score)

        # Log to wandb with epoch prefix
        wandb.log({
            f"checkpoint/epoch_{epoch}/acc": acc_score,
            f"checkpoint/epoch_{epoch}/nmi": nmi_score
        })

        print(f"[Epoch {epoch}] ACC = {acc_score:.4f}, NMI = {nmi_score:.4f}")

    # Train with checkpoint evaluation
    C_final = train_subspace_c(X, lambda_c=lambda_c, lr=lr, epochs=epochs,
                               checkpoint_epochs=checkpoint_epochs,
                               checkpoint_callback=checkpoint_callback)

    # Visualize final self-representation matrix (only if save_path is provided)
    if visualize and save_path:
        visualize_self_representation_matrix(C_final, labels_np,
                                             save_path=save_path,
                                             title=f'Self-Representation Matrix (lambda={lambda_c})')

    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Convert config to dict for easier access
    config = OmegaConf.to_container(cfg, resolve=True)

    # Get original working directory (Hydra changes it)
    original_wd = hydra.utils.get_original_cwd()

    # Create visualization directory if it doesn't exist
    save_viz_path = config['save_viz']
    if not os.path.isabs(save_viz_path):
        save_viz_path = os.path.join(original_wd, save_viz_path)
    os.makedirs(save_viz_path, exist_ok=True)

    datasets = config['dataset'] if isinstance(config['dataset'], list) else [config['dataset']]
    checkpoint_epochs = config.get('checkpoint_epochs', None)

    # Initialize wandb
    wandb.init(
        project="DSASC-baseline",
        config=config,
        name=f"baseline_{'_'.join(datasets)}_lambda{config['lambda_c']}_lr{config['lr']}"
    )

    print("=" * 80)
    print("Baseline Clustering on DINO Features (Final Layer Only)")
    print("=" * 80)

    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 80}")

        # Set number of clusters
        if dataset == 'cifar100':
            num_clusters = 100
        else:
            num_clusters = 10

        # Load DINO features
        saved_features = torch.load(os.path.join(original_wd, features_save_dir, dataset + features_suffix))
        features = saved_features['data']
        labels = saved_features['label']

        # Get final layer features
        X_final = features[-1].cpu().numpy()
        X_final_torch = features[-1]

        print(f"Feature shape: {X_final.shape}")

        # K-means clustering
        print(f"\n[K-means Clustering]")
        y_pred_kmeans = kmeans_baseline(X_final, num_clusters)
        acc_kmeans = cluster_accuracy(labels.numpy(), y_pred_kmeans)
        nmi_kmeans = nmi(labels.numpy(), y_pred_kmeans)
        print(f"ACC = {acc_kmeans:.4f}, NMI = {nmi_kmeans:.4f}")

        # Log K-means metrics to wandb
        wandb.log({
            f"{dataset}/kmeans/acc": acc_kmeans,
            f"{dataset}/kmeans/nmi": nmi_kmeans
        })

        # Subspace clustering
        print(f"\n[Subspace Clustering]")
        if checkpoint_epochs:
            print(f"Training C matrix (lambda={config['lambda_c']}, lr={config['lr']}, max_epochs={config['epochs']})")
            print(f"Checkpoints: {checkpoint_epochs}")
        else:
            print(f"Training C matrix (lambda={config['lambda_c']}, lr={config['lr']}, epochs={config['epochs']})...")

        # Prepare save path for visualization
        save_path = None
        if save_viz_path:
            save_path = os.path.join(save_viz_path,
                                     f"{dataset}_C_matrix_lambda{config['lambda_c']}_lr{config['lr']}.png")

        # Run subspace clustering with checkpoints
        results = subspace_clustering_baseline(
            X_final_torch, num_clusters, labels,
            lambda_c=config['lambda_c'],
            lr=config['lr'],
            epochs=config['epochs'],
            checkpoint_epochs=checkpoint_epochs,
            save_path=save_path
        )

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"Summary for {dataset}:")
        print(f"{'=' * 80}")
        print(f"  K-means:      ACC = {acc_kmeans:.4f}, NMI = {nmi_kmeans:.4f}")
        print(f"\n  Subspace Clustering:")
        for epoch in sorted(results.keys()):
            y_pred, acc_subspace, nmi_subspace = results[epoch]
            print(f"    Epoch {epoch:5d}:  ACC = {acc_subspace:.4f}, NMI = {nmi_subspace:.4f}")

        # Log best checkpoint metrics to wandb
        best_epoch = max(results.keys(), key=lambda e: results[e][1])  # Best by ACC
        best_acc = results[best_epoch][1]
        best_nmi = results[best_epoch][2]
        wandb.log({
            f"{dataset}/subspace/best_epoch": best_epoch,
            f"{dataset}/subspace/best_acc": best_acc,
            f"{dataset}/subspace/best_nmi": best_nmi
        })

    print("\n" + "=" * 80)
    print("Baseline evaluation completed!")
    print("=" * 80)

    wandb.finish()


if __name__ == "__main__":
    main()
