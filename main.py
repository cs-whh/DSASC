import os.path
import hydra
from omegaconf import DictConfig, OmegaConf
from myutil import *
from model import *
from post_clustering import *
from data import *
from tqdm import tqdm
import wandb
from sklearn.cluster import KMeans


def evaluate_clustering(C, label, num_cluster):
    """Evaluate clustering performance using the self-expression matrix C"""
    C_np = C.detach().to('cpu').numpy()
    y_pred = sklearn_spectral_clustering(C_np, num_cluster)
    acc = cluster_accuracy(label, y_pred)
    nmi_score = nmi(label, y_pred)
    return acc, nmi_score


def structured_regularization_loss(C, pseudo_labels, alpha, beta):
    """
    Structured regularization loss for self-expression matrix C.

    Encourages:
    - Large C_ij for samples i, j from the same class
    - Small C_kl for samples k, l from different classes

    Args:
        C: (N, N) self-expression matrix
        pseudo_labels: (N,) pseudo labels from K-means
        alpha: weight for same-class samples (encourage large values)
        beta: weight for different-class samples (encourage small values)

    Returns:
        loss: scalar loss value
    """
    # Create same/different class masks
    label_mask = (pseudo_labels[:, None] == pseudo_labels[None, :])  # (N, N)

    # Exclude diagonal (self-connections)
    diag_mask = torch.eye(C.shape[0], device=C.device, dtype=torch.bool)
    same_mask = label_mask & ~diag_mask
    diff_mask = ~label_mask & ~diag_mask

    # Same class: encourage large C_ij (use negative sign)
    loss_same = -torch.mean(C[same_mask]**2)

    # Different class: encourage C_kl -> 0
    loss_diff = torch.mean(C[diff_mask]**2)

    return alpha * loss_same + beta * loss_diff


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Set seed
    setup_seed(cfg.seed)

    # Initialize Weights & Biases
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    # Update config based on dataset
    num_cluster = cfg.graph.num_cluster
    epochs = cfg.training.epochs
    if cfg.dataset == 'cifar100':
        num_cluster = 100
        epochs = 4001

    # Load data
    saved_features = torch.load(os.path.join(features_save_dir, cfg.dataset + features_suffix))
    features = saved_features['data']
    label = saved_features['label']

    # ========== Generate Pseudo Labels using K-means ==========
    print("\nGenerating pseudo labels using K-means...")
    device_feat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dino = features[-1].to(device_feat).cpu().numpy()  # Use DINO features
    y_true_np = label.numpy()

    # K-means clustering
    kmeans = KMeans(n_clusters=num_cluster, random_state=cfg.seed, n_init=10)
    pseudo_labels = kmeans.fit_predict(features_dino)
    pseudo_labels = torch.from_numpy(pseudo_labels).long().to(device_feat)

    # Compute initial accuracy
    initial_acc = cluster_accuracy(y_true_np, pseudo_labels.cpu().numpy())
    print(f"Initial K-means accuracy: {initial_acc:.4f}")

    # Remove samples farthest from cluster centers (optional - high confidence samples only)
    centroids = kmeans.cluster_centers_
    remove_indices = []

    # Calculate number of samples to remove per cluster
    total_samples = len(pseudo_labels)
    samples_per_cluster = total_samples // num_cluster
    n_remove_per_cluster = max(1, min(samples_per_cluster // 2, 500))  # Remove at most half, or 500 if larger

    for cluster_id in range(num_cluster):
        cluster_mask = (pseudo_labels == cluster_id)
        cluster_features = features_dino[cluster_mask.cpu().numpy()]
        cluster_indices = torch.where(cluster_mask)[0]

        distances = np.linalg.norm(cluster_features - centroids[cluster_id], axis=1)
        sorted_indices = np.argsort(distances)

        n_remove = min(n_remove_per_cluster, len(sorted_indices) - 1)  # Keep at least 1 sample
        farthest_indices = sorted_indices[-n_remove:]
        remove_indices.extend(cluster_indices[farthest_indices].cpu().numpy())

    remove_indices = torch.tensor(remove_indices)
    keep_mask = torch.ones(len(pseudo_labels), dtype=bool)
    keep_mask[remove_indices] = False

    print(f"Removed {len(remove_indices)} low-confidence samples, kept {keep_mask.sum()} high-confidence samples")
    print(f"High-confidence samples accuracy: {cluster_accuracy(y_true_np[keep_mask], pseudo_labels[keep_mask].cpu().numpy()):.4f}")

    print(f"\nDynamic structured regularization:")
    print(f"  lambda6: {cfg.loss.lambda6}")
    print(f"  alpha: {cfg.loss.alpha}")
    print(f"  beta: {cfg.loss.beta}")
    print(f"  warmup_ratio: {cfg.loss.warmup_ratio}")
    print(f"  Will keep full weight for {int(epochs * cfg.loss.warmup_ratio)} epochs")
    print(f"  Then linearly decay to 0 by epoch {epochs}")

    # Build graph using features
    sparse_adj = bulid_graph(features, K=cfg.graph.num_neighbor)
    datas = bulid_pyg_data(features, sparse_adj)

    # Extract dimensions
    num_samples = features[0].shape[0]
    dino_dim = features[-1].shape[-1]  # DINO features (last in list)
    layer_dims = [features[0].shape[-1], features[1].shape[-1], features[2].shape[-1]]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoupledSelfRepresentation(
        num_samples=num_samples,
        dino_dim=dino_dim,
        layer_dims=layer_dims,
        hidden_dim=cfg.model.hidden_dim,
        gcn_hidden=cfg.model.gcn_hidden
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Forward pass
        C, recon_dino, recon_layers, H_c, H_s, H_c_recon, H_s_recon = model(
            dino_features=datas[-1].x,
            layer_features=[datas[0].x, datas[1].x, datas[2].x],
            edge_index=datas[0].edge_index  # Use same graph for all layers
        )

        # ========== Loss Computation ==========
        # Original space reconstruction loss
        loss_content_recon = F.mse_loss(datas[-1].x, recon_dino)
        loss_structure_recon = (
            F.mse_loss(datas[0].x, recon_layers[0]) +
            F.mse_loss(datas[1].x, recon_layers[1]) +
            F.mse_loss(datas[2].x, recon_layers[2])
        )

        # Hidden space self-expression loss
        loss_content_self = F.mse_loss(H_c, H_c_recon)
        loss_structure_self = F.mse_loss(H_s, H_s_recon)

        # Sparse regularization (L1 norm on C)
        loss_sparse = torch.linalg.matrix_norm(C, 1)

        # Dynamic weight adjustment for structured regularization
        # Early training: use full weight to guide learning
        # Late training: gradually reduce weight to let model learn freely
        progress = epoch / epochs
        if progress < cfg.loss.warmup_ratio:
            current_lambda6 = cfg.loss.lambda6
        else:
            # Linear decay from lambda6 to 0
            decay_progress = (progress - cfg.loss.warmup_ratio) / (1.0 - cfg.loss.warmup_ratio)
            current_lambda6 = cfg.loss.lambda6 * (1.0 - decay_progress)

        # Structured regularization (pseudo-label constraint)
        loss_structured = structured_regularization_loss(
            C,
            pseudo_labels,
            cfg.loss.alpha,
            cfg.loss.beta
        )

        # Total loss
        total_loss = (
            cfg.loss.lambda1 * loss_content_recon +
            cfg.loss.lambda2 * loss_structure_recon +
            cfg.loss.lambda3 * loss_sparse +
            cfg.loss.lambda4 * loss_content_self +
            cfg.loss.lambda5 * loss_structure_self +
            current_lambda6 * loss_structured
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": total_loss.item()})

        # ========== Wandb Logging ==========
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "loss/total": total_loss.item(),
                "loss/content_recon": loss_content_recon.item(),
                "loss/structure_recon": loss_structure_recon.item(),
                "loss/content_self": loss_content_self.item(),
                "loss/structure_self": loss_structure_self.item(),
                "loss/sparse": loss_sparse.item(),
                "loss/structured": loss_structured.item(),
                "loss/lambda6_current": current_lambda6,
            })

        # ========== Checkpoint Evaluation ==========
        if epoch in cfg.training.checkpoints_epochs:
            acc, nmi_score = evaluate_clustering(C, label, num_cluster)
            print(f"\n[Checkpoint Epoch {epoch}] ACC = {acc:.4f}, NMI = {nmi_score:.4f}")

            # Log checkpoint metrics to wandb
            if cfg.wandb.enabled:
                wandb.log({
                    "epoch": epoch,
                    "checkpoint/ACC": acc,
                    "checkpoint/NMI": nmi_score,
                })

    # ========== Final Clustering Results ==========
    print("\nFinal coupled self-representation clustering results:")
    acc, nmi_score = evaluate_clustering(C, label, num_cluster)
    print(f"ACC = {acc:.4f}, NMI = {nmi_score:.4f}")

    # Log final results to wandb
    if cfg.wandb.enabled:
        wandb.log({
            "final/ACC": acc,
            "final/NMI": nmi_score,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
