import os.path
import re
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
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


SCORE_BOUNDS_PATH = os.path.join(os.path.dirname(__file__), "docs", "sweep.md")


def load_score_bounds(path):
    bounds = {}
    current_dataset = None
    line_re = re.compile(r'^(acc|nmi)\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*$')
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = line_re.match(line)
            if match:
                if current_dataset is None:
                    raise ValueError(f"Invalid sweep bounds format in {path}: metric before dataset name.")
                metric = match.group(1)
                floor = float(match.group(2))
                target = float(match.group(3))
                bounds[current_dataset][metric] = (floor, target)
                continue
            current_dataset = line
            if current_dataset in bounds:
                raise ValueError(f"Duplicate dataset entry in {path}: {current_dataset}")
            bounds[current_dataset] = {}
    return bounds


def normalize_score(value, floor, target):
    if target <= floor:
        return 0.0
    score = (value - floor) / (target - floor)
    return max(0.0, min(1.0, float(score)))


def dataset_score(dataset, acc_value, nmi_value, bounds):
    if dataset not in bounds:
        raise ValueError(f"Missing score bounds for dataset '{dataset}' in {SCORE_BOUNDS_PATH}.")
    if "acc" not in bounds[dataset] or "nmi" not in bounds[dataset]:
        raise ValueError(f"Incomplete score bounds for dataset '{dataset}' in {SCORE_BOUNDS_PATH}.")
    acc_floor, acc_target = bounds[dataset]["acc"]
    nmi_floor, nmi_target = bounds[dataset]["nmi"]
    score_acc = normalize_score(acc_value, acc_floor, acc_target)
    score_nmi = normalize_score(nmi_value, nmi_floor, nmi_target)
    return 0.5 * (score_acc + score_nmi)


def resolve_dataset_settings(cfg, dataset):
    num_cluster = cfg.graph.num_cluster
    epochs = cfg.training.epochs
    if dataset == 'cifar100':
        num_cluster = 100
        epochs = 4001
    return num_cluster, epochs


def as_list(value):
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def aggregate_metrics(dataset_results, datasets):
    common_checkpoints = None
    for dataset in datasets:
        checkpoints = set(dataset_results[dataset]["checkpoints"].keys())
        common_checkpoints = checkpoints if common_checkpoints is None else (common_checkpoints & checkpoints)
    common_checkpoints = sorted(common_checkpoints) if common_checkpoints else []

    aggregated = {"checkpoints": {}, "final": {}}
    for epoch in common_checkpoints:
        acc_mean = float(np.mean([dataset_results[d]["checkpoints"][epoch]["acc"] for d in datasets]))
        nmi_mean = float(np.mean([dataset_results[d]["checkpoints"][epoch]["nmi"] for d in datasets]))
        score_mean = float(np.mean([dataset_results[d]["checkpoints"][epoch]["score"] for d in datasets]))
        aggregated["checkpoints"][epoch] = {
            "acc": acc_mean,
            "nmi": nmi_mean,
            "score": score_mean,
        }

    acc_final = float(np.mean([dataset_results[d]["final"]["acc"] for d in datasets]))
    nmi_final = float(np.mean([dataset_results[d]["final"]["nmi"] for d in datasets]))
    score_final = float(np.mean([dataset_results[d]["final"]["score"] for d in datasets]))
    aggregated["final"] = {"acc": acc_final, "nmi": nmi_final, "score": score_final}
    return aggregated


def run_single_dataset(cfg, dataset, score_bounds, multi_dataset):
    num_cluster, epochs = resolve_dataset_settings(cfg, dataset)
    checkpoints_epochs = [e for e in cfg.training.checkpoints_epochs if e <= epochs]

    # Load data
    saved_features = torch.load(os.path.join(features_save_dir, dataset + features_suffix))
    features = saved_features['data']
    label = saved_features['label']

    # Build graph using only the last layer (DINO features)
    dino_features_only = [features[-1]]  # Only last layer
    sparse_adj = bulid_graph(dino_features_only, K=cfg.graph.num_neighbor)

    # Create PyG data structures - all layers share the same edge_index from DINO
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

    dataset_result = {"checkpoints": {}, "final": {}, "epochs": epochs}
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Forward pass: get averaged C and all ensemble outputs
        C_avg, Cs, all_outputs = model(
            dino_features=datas[-1].x,
            layer_features=[datas[0].x, datas[1].x, datas[2].x],
            edge_index=datas[0].edge_index  # Use same graph for all layers
        )

        # ========== Loss Computation ==========
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
        if epoch in checkpoints_epochs:
            acc, nmi_score = evaluate_clustering(C_avg, label, num_cluster)
            score_value = dataset_score(dataset, acc, nmi_score, score_bounds) if score_bounds else None
            print(f"\n[{dataset}][Checkpoint Epoch {epoch}] ACC = {acc:.4f}, NMI = {nmi_score:.4f}")

            dataset_result["checkpoints"][epoch] = {
                "acc": acc,
                "nmi": nmi_score,
                "score": score_value if score_value is not None else 0.0,
            }

            # Log checkpoint metrics to wandb
            if cfg.wandb.enabled:
                log_payload = {"epoch": epoch}
                if multi_dataset:
                    log_payload[f"dataset/{dataset}/checkpoint/ACC"] = acc
                    log_payload[f"dataset/{dataset}/checkpoint/NMI"] = nmi_score
                    if score_value is not None:
                        log_payload[f"dataset/{dataset}/checkpoint/score"] = score_value
                else:
                    log_payload["checkpoint/ACC"] = acc
                    log_payload["checkpoint/NMI"] = nmi_score
                wandb.log(log_payload)

    # ========== Final Clustering Results ==========
    print(f"\n[{dataset}] Final ensemble coupled self-representation clustering results:")
    acc, nmi_score = evaluate_clustering(C_avg, label, num_cluster)
    score_value = dataset_score(dataset, acc, nmi_score, score_bounds) if score_bounds else None
    print(f"ACC = {acc:.4f}, NMI = {nmi_score:.4f}")

    dataset_result["final"] = {
        "acc": acc,
        "nmi": nmi_score,
        "score": score_value if score_value is not None else 0.0,
    }

    # Log final results to wandb
    if cfg.wandb.enabled:
        log_payload = {"epoch": epochs}
        if multi_dataset:
            log_payload[f"dataset/{dataset}/final/ACC"] = acc
            log_payload[f"dataset/{dataset}/final/NMI"] = nmi_score
            if score_value is not None:
                log_payload[f"dataset/{dataset}/final/score"] = score_value
        else:
            log_payload["final/ACC"] = acc
            log_payload["final/NMI"] = nmi_score
        wandb.log(log_payload)

    return dataset_result


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

    if "dataset_list" in cfg and cfg.dataset_list:
        datasets = as_list(cfg.dataset_list)
    else:
        datasets = as_list(cfg.dataset)

    multi_dataset = len(datasets) > 1
    score_bounds = load_score_bounds(SCORE_BOUNDS_PATH) if multi_dataset else None
    if score_bounds:
        missing = [dataset for dataset in datasets if dataset not in score_bounds]
        if missing:
            raise ValueError(f"Missing score bounds for datasets in {SCORE_BOUNDS_PATH}: {missing}")

    dataset_results = {}
    for dataset in datasets:
        print(f"\n========== Running dataset: {dataset} ==========")
        dataset_results[dataset] = run_single_dataset(cfg, dataset, score_bounds, multi_dataset)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(datasets) > 1:
        aggregated = aggregate_metrics(dataset_results, datasets)
        print("\nAggregate results across datasets:")
        print(f"Final ACC(mean) = {aggregated['final']['acc']:.4f}, "
              f"NMI(mean) = {aggregated['final']['nmi']:.4f}, "
              f"Score(mean) = {aggregated['final']['score']:.4f}")

        if cfg.wandb.enabled:
            for epoch, values in aggregated["checkpoints"].items():
                wandb.log({
                    "epoch": epoch,
                    "aggregate/ACC_mean": values["acc"],
                    "aggregate/NMI_mean": values["nmi"],
                    "aggregate/score": values["score"],
                })
            final_epoch = max(result["epochs"] for result in dataset_results.values())
            wandb.log({
                "epoch": final_epoch,
                "aggregate/ACC_mean": aggregated["final"]["acc"],
                "aggregate/NMI_mean": aggregated["final"]["nmi"],
                "aggregate/score": aggregated["final"]["score"],
            })

            if aggregated["checkpoints"]:
                best_score = max(v["score"] for v in aggregated["checkpoints"].values())
                wandb.run.summary["aggregate/score"] = best_score
            else:
                wandb.run.summary["aggregate/score"] = aggregated["final"]["score"]

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
