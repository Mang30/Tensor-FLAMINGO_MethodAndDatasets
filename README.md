# Tensor-FLAMINGO scHi-C 插补增强方法使用和FLAMINGO模拟数据构建代码实现
具体使用详见两个文件夹中的md文档介绍

#### PD，IF转换公式
IF = PD^(-4) 

PD = IF^(-0.25) 

1. 从距离转换为IF

# R语言
sparse_pd <- read.table("consensus_1_slice_1.txt")
sparse_if <- pmax(abs(sparse_pd), 1e-10)^(-4)

# Python
import numpy as np
sparse_pd = np.loadtxt("consensus_1_slice_1.txt")
sparse_if = np.maximum(np.abs(sparse_pd), 1e-10)**(-4)
2. FLAMINGO重建

library(tFlamingorLite)

pred <- flamingo.reconstruct_structure_worker(
  input_if = sparse_if,
  pd = sparse_pd,
  sw = 0.75,        # 使用75%观测值
  lambda = 1,       # 正则化参数
  max_dist = 0.01,  # 最大距离
  nThread = 4
)

# 保存结果
write.table(pred@coordinates, "predicted_coords.txt", 
            row.names=FALSE, col.names=FALSE)
3. 质量评估

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# 加载GT和预测
gt_coords = np.loadtxt("benchmark_consensus_structure/consensus_1.txt")
pred_coords = np.loadtxt("predicted_coords.txt")

# 计算距离矩阵
gt_dist = cdist(gt_coords, gt_coords)
pred_dist = cdist(pred_coords, pred_coords)

# Spearman相关性
n = gt_dist.shape[0]
triu_idx = np.triu_indices(n, k=1)
spearman_corr, _ = spearmanr(gt_dist[triu_idx], pred_dist[triu_idx])

# RMSD
rmsd = np.sqrt(np.mean((pred_coords - gt_coords)**2))

print(f"Spearman: {spearman_corr:.4f}")
print(f"RMSD: {rmsd:.4f}")
✅ 成功标准（论文要求）
指标	模拟数据	10kb分辨率
Spearman	> 0.60	> 0.40
RMSD	< 0.08	< 0.16
📁 生成的文件结构

simulation_pipeline_output/
├── simulation_input_for_tensor/
│   ├── consensus_1_slice_1_IF.txt
│   └── consensus_1_slice_1_PD.txt
│
├── reconstructed_coords/
│   ├── Cell_1_coords.txt
│   ├── Cell_2_coords.txt
│   ...
│   └── Cell_10_coords.txt
│
└── evaluation_results/
    └── quality_metrics.json
