stl10
acc [0.92, 0.98]
nmi [0.90, 0.96]

cifar10
acc [0.82, 0.88]
nmi [0.75, 0.82]

fashion_mnist
acc [0.70, 0.73]
nmi [0.65, 0.69]

cifar100
acc [0.55, 0.58]
nmi [0.70, 0.75]


单数据集（完全兼容旧方式）

  python main.py dataset=cifar10 training.lr=1e-5

  多数据集（聚合评分）

  python main.py dataset_list="[cifar10,cifar100,fashion_mnist,stl10]" training.lr=1e-5

  （Hydra 也支持 dataset="[cifar10,cifar100,...]"，但我新增了 dataset_list 作为显式入口。）

  W&B 记录项（多数据集）

  - 每个数据集：
      - dataset/<name>/checkpoint/ACC
      - dataset/<name>/checkpoint/NMI
      - dataset/<name>/final/ACC
      - dataset/<name>/final/NMI
  - 聚合指标：
      - aggregate/score
      - aggregate/ACC_mean
      - aggregate/NMI_mean

  Sweep（优化聚合分数）

  wandb sweep sweep_config.yaml
  wandb agent <SWEEP_ID> --dataset_list="[cifar10,cifar100,fashion_mnist,stl10]"

  如果你希望把聚合分数改成“加权平均”或“最差值”，告诉我权重或规则即可。