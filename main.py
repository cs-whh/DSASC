import os.path
import hydra
from omegaconf import DictConfig, OmegaConf
from myutil import *
from model import *
from post_clustering import *
from data import *
from tqdm import tqdm
import wandb
import torch_geometric as torchgeo


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

    # Build graph using only the last layer (DINO features)
    # Use DINO features to construct k-NN graph edges
    dino_features_only = [features[-1]]  # Only last layer
    sparse_adj = bulid_graph(dino_features_only, K=cfg.graph.num_neighbor)

    # Create PyG data structures - all layers share the same edge_index from DINO
    # But each layer has its own features
    edge_index = torch.from_numpy(
        np.transpose(np.stack(sparse_adj[0].nonzero(), axis=1))
    )
    edge_index = torchgeo.utils.to_undirected(edge_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas = []
    for i in range(len(features)):
        pyg_graph = torchgeo.data.Data()
        pyg_graph.x = features[i].to(device)
        pyg_graph.edge_index = edge_index.to(device)  # All layers share same edges
        pyg_graph.num_nodes = features[i].shape[0]
        datas.append(pyg_graph)

    # Extract dimensions
    num_samples = features[0].shape[0]
    dino_dim = features[-1].shape[-1]  # DINO features (last in list)
    layer_dims = [features[0].shape[-1], features[1].shape[-1], features[2].shape[-1]]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleCoupledSelfRepresentation(
        num_ensembles=cfg.model.num_ensembles,
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
        # Forward pass: get averaged C and all ensemble outputs
        C_avg, Cs, all_outputs = model(
            dino_features=datas[-1].x,
            layer_features=[datas[0].x, datas[1].x, datas[2].x],
            edge_index=datas[0].edge_index  # Use same graph for all layers
        )

        # ========== Loss Computation ==========
        # Compute loss for each ensemble member and average
        total_loss = torch.tensor(0.0, device=device)
        sum_content_recon = torch.tensor(0.0, device=device)
        sum_structure_recon = torch.tensor(0.0, device=device)
        sum_content_self = torch.tensor(0.0, device=device)
        sum_structure_self = torch.tensor(0.0, device=device)
        sum_sparse = torch.tensor(0.0, device=device)

        for output in all_outputs:
            C, recon_dino, recon_layers, H_c, H_s, H_c_recon, H_s_recon = output

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

            # Accumulate for averaging
            sum_content_recon += loss_content_recon
            sum_structure_recon += loss_structure_recon
            sum_content_self += loss_content_self
            sum_structure_self += loss_structure_self
            sum_sparse += loss_sparse

            # Total loss for this ensemble member
            loss_i = (
                cfg.loss.lambda1 * loss_content_recon +
                cfg.loss.lambda2 * loss_structure_recon +
                cfg.loss.lambda3 * loss_sparse +
                cfg.loss.lambda4 * loss_content_self +
                cfg.loss.lambda5 * loss_structure_self
            )
            total_loss += loss_i

        # Average over all ensemble members
        total_loss = total_loss / len(all_outputs)
        avg_content_recon = sum_content_recon / len(all_outputs)
        avg_structure_recon = sum_structure_recon / len(all_outputs)
        avg_content_self = sum_content_self / len(all_outputs)
        avg_structure_self = sum_structure_self / len(all_outputs)
        avg_sparse = sum_sparse / len(all_outputs)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": total_loss.item()})

        # ========== Wandb Logging ==========
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "loss/total": float(total_loss.item()),
                "loss/content_recon": float(avg_content_recon.item()),
                "loss/structure_recon": float(avg_structure_recon.item()),
                "loss/content_self": float(avg_content_self.item()),
                "loss/structure_self": float(avg_structure_self.item()),
                "loss/sparse": float(avg_sparse.item()),
            })

        # ========== Checkpoint Evaluation ==========
        if epoch in cfg.training.checkpoints_epochs:
            acc, nmi_score = evaluate_clustering(C_avg, label, num_cluster)
            print(f"\n[Checkpoint Epoch {epoch}] ACC = {acc:.4f}, NMI = {nmi_score:.4f}")

            # Log checkpoint metrics to wandb
            if cfg.wandb.enabled:
                wandb.log({
                    "epoch": epoch,
                    "checkpoint/ACC": acc,
                    "checkpoint/NMI": nmi_score,
                })

    # ========== Final Clustering Results ==========
    print("\nFinal ensemble coupled self-representation clustering results:")
    acc, nmi_score = evaluate_clustering(C_avg, label, num_cluster)
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
