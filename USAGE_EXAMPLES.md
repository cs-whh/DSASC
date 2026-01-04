# Hydra Baseline - 使用说明

## 核心优化：训练一次，多检查点评估

代码已优化：训练一次到最大 epoch（10000），在中间检查点（1000, 3000, 5000, 7000, 10000）进行聚类评估。
- **之前**：150 次实验（6 lambda × 5 lr × 5 epochs），每次从头训练
- **现在**：30 次实验（6 lambda × 5 lr），每次训练一次，在 5 个检查点评估

## 基本运行

### 使用默认配置运行
```bash
python baseline.py
```
输出会在 5 个检查点（1000, 3000, 5000, 7000, 10000）分别评估聚类性能。

### 自定义检查点
```bash
python baseline.py checkpoint_epochs=[500,1000,2000,5000]
```

### 更改数据集
```bash
python baseline.py dataset=[cifar10,cifar100]
python baseline.py dataset=fashion_mnist
```

## 参数搜索

### 1. 完整网格搜索（推荐）

搜索 lambda_c 和 lr 的所有组合（6 × 5 = 30 次实验）：
```bash
python baseline.py --multirun \
  lambda_c=0.1,0.5,1.0,2.0,5.0,10.0 \
  lr=0.0001,0.0005,0.001,0.005,0.01
```

每次实验会自动在 5 个检查点评估，最终得到 150 个数据点（30 × 5）。

### 2. 单参数搜索

只搜索 lambda_c（6 次实验）：
```bash
python baseline.py --multirun lambda_c=0.1,0.5,1.0,2.0,5.0,10.0
```

只搜索学习率（5 次实验）：
```bash
python baseline.py --multirun lr=0.0001,0.0005,0.001,0.005,0.01
```

### 3. 针对特定数据集搜索
```bash
python baseline.py --multirun \
  lambda_c=0.5,1.0,2.0 \
  lr=0.001,0.005 \
  dataset=cifar10
```

## 常用搜索策略

### 粗搜索（快速探索）
```bash
python baseline.py --multirun \
  lambda_c=0.1,1.0,10.0 \
  lr=0.0001,0.001,0.01 \
  dataset=cifar10
```
9 次实验，每次 5 个检查点 = 45 个数据点

### 细粒度搜索（在粗搜索基础上）
假设发现 lambda_c=1.0, lr=0.001 较好：
```bash
python baseline.py --multirun \
  lambda_c=0.5,0.8,1.0,1.2,1.5 \
  lr=0.0005,0.0008,0.001,0.0012,0.0015
```
25 次实验，每次 5 个检查点 = 125 个数据点

## 结果查看

所有实验结果会自动上传到 wandb，可以在 wandb dashboard 中比较不同参数组合的性能。

查看 wandb 项目：
```bash
wandb dashboard  # 在浏览器中打开
```

## Hydra 输出目录

每次运行会在 `outputs/` 目录下创建一个带时间戳的文件夹，包含：
- `.hydra/` - 配置文件和日志
- 控制台输出重定向到文件

例如：
```
outputs/
├── 2024-01-15/
│   ├── 10-30-45/
│   │   ├── .hydra/
│   │   │   ├── config.yaml
│   │   │   ├── hydra.yaml
│   │   │   └── overrides.yaml
│   │   └── baseline.log
```

## 配置文件说明

- `configs/config.yaml` - 主配置文件（默认参数）
- `configs/db/default.yaml` - Hydra 配置组占位符

修改 `config.yaml` 中的默认值来改变基准参数。
