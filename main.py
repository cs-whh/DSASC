import os.path
import hydra
from omegaconf import DictConfig, OmegaConf
from myutil import *
from model import *
from post_clustering import *
from data import *
from tqdm import tqdm
import wandb


def evaluate_clustering(C, label, num_cluster):
    """Evaluate clustering performance using the self-expression matrix C"""
    C_np = C.detach().to('cpu').numpy()
    y_pred = sklearn_spectral_clustering(C_np, num_cluster)
    acc = cluster_accuracy(label, y_pred)
    nmi_score = nmi(label, y_pred)
    return acc, nmi_score


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

        # Total loss
        total_loss = (
            cfg.loss.lambda1 * loss_content_recon +
            cfg.loss.lambda2 * loss_structure_recon +
            cfg.loss.lambda3 * loss_sparse +
            cfg.loss.lambda4 * loss_content_self +
            cfg.loss.lambda5 * loss_structure_self
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
