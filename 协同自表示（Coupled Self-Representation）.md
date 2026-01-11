
# 理论部分

## 1. 动机与核心思想

### 1.1 原方法的局限性
原始DSASC方法采用**独立自表示 + 简单相加**的策略：
- 内容特征自表示系数：$\mathbf{C}_c \in \mathbb{R}^{n \times n}$
- 结构特征自表示系数：$\mathbf{C}_s \in \mathbb{R}^{n \times n}$
- 融合方式：$\mathbf{C}_{fusion} = \mathbf{C}_c + \mathbf{C}_s$

**问题：**
1. 两个系数矩阵独立优化，缺乏耦合约束
2. 简单相加未考虑特征间的深层关系
3. 可能学到矛盾的样本关系（如$C_c[i,j]$大但$C_s[i,j]$小）

### 1.2 协同自表示策略
**核心思想：** 内容特征和结构特征**共享单一自表示系数矩阵** $\mathbf{C} \in \mathbb{R}^{n \times n}$，通过联合优化强制两种模态学习一致的样本亲和关系。

$$
\begin{cases}
\mathbf{H}_c = f_{enc}^c(\mathbf{X}_{DINO}) \\
\mathbf{H}_s = f_{enc}^s(\mathbf{X}_{GCN}, \mathcal{G}) \\
\tilde{\mathbf{H}}_c = \mathbf{C} \mathbf{H}_c \\
\tilde{\mathbf{H}}_s = \mathbf{C} \mathbf{H}_s \\
\hat{\mathbf{X}}_c = f_{dec}^c(\tilde{\mathbf{H}}_c) \\
\hat{\mathbf{X}}_s = f_{dec}^s(\tilde{\mathbf{H}}_s, \mathcal{G})
\end{cases}
$$

---

## 2. 数学公式推导

### 2.1 符号定义
- $\mathbf{X}_{DINO} \in \mathbb{R}^{n \times d_c}$：DINO提取的内容特征（$n$个样本，$d_c$维）
- $\mathbf{X}_{layer}^{(l)} \in \mathbb{R}^{n \times d_l}$：第$l$层中间特征（$l=1,2,3$）
- $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：从特征构建的KNN图
- $\mathbf{C} \in \mathbb{R}^{n \times n}$：**共享自表示系数矩阵**
- $\mathbf{H}_c \in \mathbb{R}^{n \times d_h}$：内容特征的隐表示
- $\mathbf{H}_s \in \mathbb{R}^{n \times d_h}$：结构特征的隐表示（维度对齐）

### 2.2 结构特征分支

#### 2.2.1 GCN Encoder
对三层中间特征分别进行GCN编码并融合：
$$
\mathbf{H}_s^{(l)} = \text{GCN}_{enc}^{(l)}(\mathbf{X}_{layer}^{(l)}, \mathcal{G}), \quad l=1,2,3
$$

GCN的传播规则：
$$
\mathbf{H}_s^{(l)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{X}_{layer}^{(l)} \mathbf{W}^{(l)}\right)
$$
其中$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$（加自环），$\tilde{\mathbf{D}}$为度矩阵。

**融合策略（线性投影 + 求和）：**
$$
\mathbf{H}_s = \sum_{l=1}^{3} \phi_l(\mathbf{H}_s^{(l)}) = \sum_{l=1}^{3} \mathbf{H}_s^{(l)} \mathbf{W}_{proj}^{(l)}
$$
其中$\mathbf{W}_{proj}^{(l)} \in \mathbb{R}^{d_l \times d_h}$将不同维度投影到统一隐空间。

#### 2.2.2 共享自表示层
$$
\tilde{\mathbf{H}}_s = \mathbf{C} \mathbf{H}_s
$$
含义：每个样本的隐表示由**所有样本的线性组合**重构。

#### 2.2.3 GCN Decoder
重建原始特征维度：
$$
\hat{\mathbf{X}}_{layer}^{(l)} = \text{GCN}_{dec}^{(l)}(\tilde{\mathbf{H}}_s, \mathcal{G})
$$

具体形式：
$$
\hat{\mathbf{X}}_{layer}^{(l)} = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{H}}_s \mathbf{W}_{dec}^{(l)}
$$
其中$\mathbf{W}_{dec}^{(l)} \in \mathbb{R}^{d_h \times d_l}$。

### 2.3 内容特征分支

#### 2.3.1 MLP Encoder
$$
\mathbf{H}_c = \sigma(\mathbf{X}_{DINO} \mathbf{W}_{enc}^c + \mathbf{b}_{enc}^c)
$$
其中$\mathbf{W}_{enc}^c \in \mathbb{R}^{d_c \times d_h}$，$\sigma$为激活函数（ReLU）。

#### 2.3.2 共享自表示层
$$
\tilde{\mathbf{H}}_c = \mathbf{C} \mathbf{H}_c
$$
**关键：与结构分支使用相同的$\mathbf{C}$！**

#### 2.3.3 MLP Decoder
$$
\hat{\mathbf{X}}_{DINO} = \tilde{\mathbf{H}}_c \mathbf{W}_{dec}^c + \mathbf{b}_{dec}^c
$$
其中$\mathbf{W}_{dec}^c \in \mathbb{R}^{d_h \times d_c}$。

---

## 3. 损失函数设计

### 3.1 总体损失
$$
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{recon}^c + \lambda_2 \mathcal{L}_{recon}^s + \lambda_3 \mathcal{L}_{reg} + \lambda_4 \mathcal{L}_{consist}
$$

### 3.2 重建损失

#### 内容特征重建
$$
\mathcal{L}_{recon}^c = \|\mathbf{X}_{DINO} - \hat{\mathbf{X}}_{DINO}\|_F^2 = \sum_{i=1}^n \sum_{j=1}^{d_c} (X_{ij}^{DINO} - \hat{X}_{ij}^{DINO})^2
$$

#### 结构特征重建
对三层特征分别计算重建误差：
$$
\mathcal{L}_{recon}^s = \sum_{l=1}^{3} \|\mathbf{X}_{layer}^{(l)} - \hat{\mathbf{X}}_{layer}^{(l)}\|_F^2
$$

### 3.3 正则化损失

#### $\ell_1$稀疏正则（核心）
$$
\mathcal{L}_{reg} = \|\mathbf{C}\|_1 = \sum_{i=1}^n \sum_{j=1}^n |C_{ij}|
$$
**作用：** 促使自表示系数稀疏，符合子空间聚类假设（每个样本仅由同子空间的少数样本表示）。

#### 对角抑制（可选）
$$
\mathcal{L}_{diag} = \sum_{i=1}^n C_{ii}^2
$$
**作用：** 防止自循环（样本用自己表示），鼓励用其他样本表示。

### 3.4 一致性损失（可选）
强制两种模态的隐表示对齐：
$$
\mathcal{L}_{consist} = \|\mathbf{H}_c - \mathbf{H}_s\|_F^2
$$
**权衡：** 过强的约束可能损失模态特异性，建议设置较小的$\lambda_4$（如0.01-0.1）。

---

## 4. 优化目标

完整优化问题：
$$
\min_{\mathbf{C}, \Theta} \quad \lambda_1 \|\mathbf{X}_{DINO} - f_{dec}^c(\mathbf{C} f_{enc}^c(\mathbf{X}_{DINO}))\|_F^2 + \lambda_2 \sum_{l=1}^3 \|\mathbf{X}_{layer}^{(l)} - f_{dec}^{s,(l)}(\mathbf{C} f_{enc}^s(\{\mathbf{X}_{layer}^{(l)}\}))\|_F^2 + \lambda_3 \|\mathbf{C}\|_1
$$

其中$\Theta = \{\mathbf{W}_{enc}^c, \mathbf{W}_{dec}^c, \mathbf{W}_{proj}^{(l)}, \mathbf{W}_{dec}^{(l)}, \text{GCN参数}\}$。

**求解方法：** 端到端梯度下降（Adam优化器）。

---

## 5. 聚类推断

训练完成后，使用**共享系数矩阵$\mathbf{C}$** 进行谱聚类：

### 5.1 构造亲和矩阵
$$
\mathbf{W} = \frac{|\mathbf{C}| + |\mathbf{C}|^T}{2}
$$
**原因：**
1. 取绝对值：保留关系强度，忽略符号
2. 对称化：谱聚类需要对称矩阵

### 5.2 谱聚类
1. 计算归一化拉普拉斯矩阵：
$$
\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}
$$
其中$\mathbf{D}$为度矩阵，$D_{ii} = \sum_j W_{ij}$。

2. 求前$k$个最小特征值对应的特征向量：
$$
\mathbf{L} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad i=1,\ldots,k
$$

3. 构造嵌入矩阵$\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{n \times k}$

4. 对$\mathbf{V}$的行向量进行K-means聚类

---

# AI coding指导手册

## 架构设计

### 模型类结构

```python
class CoupledSelfRepresentation(nn.Module):
    """
    协同自表示深度聚类模型
    """
    def __init__(self, num_samples, dino_dim, layer_dims, hidden_dim=384, gcn_hidden=16):
        """
        Args:
            num_samples: 样本数量 n
            dino_dim: DINO特征维度
            layer_dims: 三层特征维度列表 [d1, d2, d3]
            hidden_dim: 共享隐空间维度
            gcn_hidden: GCN中间层维度
        """

    def forward(self, dino_features, layer_features, edge_index):
        """
        Args:
            dino_features: (n, dino_dim)
            layer_features: [(n, d1), (n, d2), (n, d3)]
            edge_index: (2, num_edges) KNN图边索引

        Returns:
            C: (n, n) 共享自表示系数
            recon_dino: (n, dino_dim) 重建的DINO特征
            recon_layers: [(n, d1), (n, d2), (n, d3)] 重建的层特征
            H_c, H_s: 隐表示（用于一致性损失）
        """
```

### 子模块设计

#### 1. GCN Encoder
```python
class GCNEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dim, gcn_hidden):
        """
        处理3层特征 → 统一隐空间
        """
        # 每层独立GCN
        self.gcn1 = GCN(input_dims[0], gcn_hidden)
        self.gcn2 = GCN(input_dims[1], gcn_hidden)
        self.gcn3 = GCN(input_dims[2], gcn_hidden)

        # 投影到统一维度
        self.proj1 = nn.Linear(gcn_hidden, hidden_dim)
        self.proj2 = nn.Linear(gcn_hidden, hidden_dim)
        self.proj3 = nn.Linear(gcn_hidden, hidden_dim)
```

#### 2. GCN Decoder
```python
class GCNDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dims, gcn_hidden):
        """
        隐空间 → 重建3层特征
        """
        # 每层独立解码路径
        self.gcn_dec1 = GCN(hidden_dim, output_dims[0])
        self.gcn_dec2 = GCN(hidden_dim, output_dims[1])
        self.gcn_dec3 = GCN(hidden_dim, output_dims[2])
```

#### 3. MLP Encoder/Decoder
```python
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        self.decoder = nn.Linear(hidden_dim, output_dim)
```

#### 4. 共享自表示层
```python
class SharedSelfExpression(nn.Module):
    def __init__(self, num_samples):
        self.C = nn.Parameter(
            1.0e-8 * torch.ones(num_samples, num_samples),
            requires_grad=True
        )

    def forward(self, x):
        return torch.matmul(self.C, x)
```

---

## 训练细节

### 超参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `hidden_dim` | 384 | 隐空间维度（与原代码保持一致） |
| `gcn_hidden` | 16 | GCN中间层维度 |
| `λ1` | 1.0 | 内容重建权重 |
| `λ2` | 1.0 | 结构重建权重 |
| `λ3` | 1.0 | 稀疏正则权重 |
| `λ4` | 0.1 | 一致性损失权重（可选，建议较小） |
| `lr` | 1e-5 | 学习率 |
| `epochs` | 2001 | 训练轮数 |

### 训练流程伪代码

```python
for epoch in range(epochs):
    # ========== 前向传播 ==========
    # 1. 编码
    H_s = gcn_encoder(layer_features, edge_index)  # (n, hidden_dim)
    H_c = mlp_encoder(dino_features)               # (n, hidden_dim)

    # 2. 共享自表示
    H_s_recon = C @ H_s  # (n, hidden_dim)
    H_c_recon = C @ H_c  # (n, hidden_dim)

    # 3. 解码
    recon_layers = gcn_decoder(H_s_recon, edge_index)  # 3层特征
    recon_dino = mlp_decoder(H_c_recon)                 # DINO特征

    # ========== 损失计算 ==========
    # 重建损失
    loss_content = MSE(dino_features, recon_dino)
    loss_structure = sum([MSE(layer_features[i], recon_layers[i]) for i in range(3)])

    # 稀疏正则
    loss_sparse = torch.norm(C, p=1)

    # 一致性损失（可选）
    loss_consist = MSE(H_c, H_s)

    # 总损失
    total_loss = λ1*loss_content + λ2*loss_structure + λ3*loss_sparse + λ4*loss_consist

    # ========== 反向传播 ==========
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## 实现注意事项

### 1. 维度对齐
- **关键：** `H_c`和`H_s`必须有相同维度`hidden_dim`，才能共享系数矩阵`C`
- 通过线性投影层确保对齐

### 2. GCN Decoder设计
- **方案A（简单）：** 单层GCN直接从`hidden_dim`映射到原始维度
  ```python
  self.gcn_dec1 = GCN(hidden_dim, output_dims[0])
  ```

- **方案B（深层）：** 多层GCN逐步上采样
  ```python
  self.gcn_dec1 = nn.Sequential(
      GCN(hidden_dim, gcn_hidden),
      GCN(gcn_hidden, output_dims[0])
  )
  ```
  推荐**方案A**（简洁高效）

### 3. 自表示矩阵初始化
- 使用极小值初始化（`1e-8`）而非零，避免梯度消失
- 对角线可设为0（强制用其他样本表示）

### 4. 一致性损失权衡
- `λ4`不宜过大（建议≤0.1），过强约束会抑制模态特异性
- 可以先不加一致性损失训练，观察`H_c`和`H_s`的自然对齐程度

### 5. 批处理考虑
- 当前实现为全批量（所有样本一起训练）
- 对于大规模数据，可考虑采样策略（如GraphSAINT）

---

## 预期改进点

1. **理论一致性**：共享`C`强制跨模态学习统一的样本关系
2. **端到端优化**：Autoencoder + 自表示联合训练，无需分阶段
3. **降维效率**：在低维隐空间做自表示，减少参数量（从$n \times d_c$降到$n \times d_h$）
4. **简化聚类**：直接用单一`C`做谱聚类，无需设计融合策略

---

## 消融实验建议

为验证各组件有效性，可进行以下对比：
1. **仅内容分支**：只用DINO + MLP Autoencoder + 自表示
2. **仅结构分支**：只用GCN Autoencoder + 自表示
3. **独立自表示**：两个分支各自有`C_c`和`C_s`，最后相加
4. **无Autoencoder**：直接在原始特征上做共享自表示（验证降维必要性）
5. **有/无一致性损失**：对比`λ4=0`和`λ4>0`的效果
