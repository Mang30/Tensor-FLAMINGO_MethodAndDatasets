# Tensor-FLAMINGO补全算法 t-SVD ADMM的使用

Tensor-FLAMINGO补全算法 t-SVD 是一种非模型训练方法，主要是通过对scHi-C 距离矩阵/接触频率矩阵数据进行 **t-SVD 低秩张量补全**，实现数据质量增强。其针对模拟数据集具体实现过程如下：

## 数据流概览

```
稀疏距离矩阵 (30个细胞，500 beads长度) 
    ↓
组装 3阶张量 T ∈ R^(500×500×30)
    ↓
t-SVD 张量补全 (ADMM优化)
    ↓
提取完整 PD 矩阵
    ↓
与 GT 对比计算评价指标
```

注意⚠️：Tensor-FLAMINGO 下构建的模拟数据集的数据矩阵为距离（PD）矩阵，因此Tensor-FLAMINGO补全算法 t-SVD ADMM 对于模拟数据集的补全/插补，是直接在 PD 矩阵上进行的，并使用增强后的 PD 矩阵完成最后的 3D 重建。而对于真实测序所得的 scHi-C 接触频率（IF）矩阵，Tensor-FLAMINGO补全算法 t-SVD ADMM 则会对 IF 矩阵进行预处理，即 linear_transformation 预处理，然后在 IF 矩阵上进行插补增强，完成插补后，再根据 IF to PD的转换公式：pd <- if_matrix^(-0.25)获得 PD 矩阵，并最后进行 3D重建。




<img width="1580" height="928" alt="image-20260613204258608" src="https://github.com/user-attachments/assets/04bcaab8-4df5-457c-9daa-e68c0e9906b8" />




## 步骤 1: 数据准备

### 输入数据结构

- **稀疏距离矩阵**: simulation/downsampled_data/consensus_X_slice_Y.txt
  - 尺寸: 500×500
  - 格式: 制表符分隔，0 表示缺失值
  - 数量: 30 个文件 (3 consensus × 10 cells)
  - 稀疏度: ~99.5% 缺失
- **完整 GT 坐标**: simulation/benchmark_consensus_structure/consensus_X.txt
  - 尺寸: 500×3 (3D 坐标)
  - 数量: 3 个文件



### 关键转换

从**距离矩阵**到**IF矩阵**的转换公式：

```
# 对于模拟数据，使用论文标准公式
IF = PD^(-4)  # alpha = 0.25 的逆运算

# 处理零值（缺失位置）
IF[PD == 0] = 0  # 保持为观测集标记
```



#### 1. t-SVD ADMM 在FLAMINGO模拟数据上的使用


<img width="2538" height="1788" alt="image-20260613211638004" src="https://github.com/user-attachments/assets/bc4c3b4c-c1a9-4507-91c9-d8d980419d16" />


<img width="1724" height="1786" alt="image-20260613211725105" src="https://github.com/user-attachments/assets/4b20be55-808b-4a00-9272-e3d6fcfc8dea" />





#### 2. t-SVD ADMM 在HiCImpute模拟数据上的使用



<img width="3316" height="1556" alt="image-20260613210328258" src="https://github.com/user-attachments/assets/8225832b-ca36-400a-b5b5-89dc9cf12666" />


## 步骤 2: 张量组装脚本

创建文件 assemble_tensor.py:

```
import numpy as np
import pandas as pd
import os
import glob

def load_sparse_distance_matrices(input_dir):
    """加载所有稀疏距离矩阵"""
    files = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    print(f"找到 {len(files)} 个距离矩阵文件")
    
    distance_matrices = []
    for f in files:
        df = pd.read_csv(f, header=None, sep='\t')
        mat = df.values
        distance_matrices.append(mat)
        print(f"加载 {os.path.basename(f)}: shape={mat.shape}, "
              f"非零元素={np.count_nonzero(mat)}, "
              f"稀疏度={1-np.count_nonzero(mat)/mat.size:.4f}")
    
    return np.array(distance_matrices), files

def convert_distance_to_if(distance_matrix):
    """将距离矩阵转换为 IF 矩阵
    
    使用公式: IF = PD^(-4)
    零值位置保持为 0（表示未观测）
    """
    # 避免除零错误
    if_matrix = np.zeros_like(distance_matrix)
    nonzero_mask = distance_matrix > 0
    if_matrix[nonzero_mask] = distance_matrix[nonzero_mask] ** (-4)
    
    return if_matrix

def assemble_tensor(distance_matrices):
    """组装 3阶张量
    
    张量维度: (cells, beads, beads)
    即 T[i,j,k] 表示第 i 个细胞的第 j,k 个 bead 之间的 IF
    """
    n_cells, n_beads, _ = distance_matrices.shape
    tensor = np.zeros((n_cells, n_beads, n_beads))
    
    for i in range(n_cells):
        tensor[i] = convert_distance_to_if(distance_matrices[i])
    
    print(f"张量组装完成: shape={tensor.shape}")
    print(f"总非零元素: {np.count_nonzero(tensor)}")
    print(f"整体稀疏度: {1-np.count_nonzero(tensor)/tensor.size:.6f}")
    
    return tensor

def compute_ground_truth_pd(gt_coords_file):
    """从 GT 坐标计算完整距离矩阵用于评估"""
    from scipy.spatial.distance import cdist
    
    coords = pd.read_csv(gt_coords_file, header=None, sep='\t').values
    gt_distance = cdist(coords, coords)
    
    print(f"GT 距离矩阵: min={gt_distance.min():.4f}, "
          f"max={gt_distance.max():.4f}, mean={gt_distance.mean():.4f}")
    
    return gt_distance

if __name__ == '__main__':
    # 配置路径
    input_dir = '/path/to/simulation/downsampled_data'
    output_dir = '/path/to/output/tensor_assembly'
    gt_dir = '/path/to/simulation/benchmark_consensus_structure'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载稀疏距离矩阵
    distance_matrices, file_list = load_sparse_distance_matrices(input_dir)
    
    # 2. 组装张量（自动转换为 IF）
    tensor = assemble_tensor(distance_matrices)
    
    # 3. 保存张量
    np.save(os.path.join(output_dir, 'sparse_if_tensor.npy'), tensor)
    print(f"张量已保存至: {output_dir}/sparse_if_tensor.npy")
    
    # 4. 保存文件索引
    with open(os.path.join(output_dir, 'file_index.txt'), 'w') as f:
        for fname in file_list:
            f.write(os.path.basename(fname) + '\n')
    
    # 5. 计算 GT 距离矩阵（用于后续评估）
    for i in range(1, 4):
        gt_file = os.path.join(gt_dir, f'consensus_{i}.txt')
        if os.path.exists(gt_file):
            gt_dist = compute_ground_truth_pd(gt_file)
            np.save(os.path.join(output_dir, f'gt_distance_consensus_{i}.npy'), gt_dist)
    
    print("数据准备完成！")
```

------

## 步骤 3: 运行 t-SVD 张量补全

### 使用现有代码

Tensor-FLAMINGO 已提供完整的 t-SVD 实现：

```
cd /Users/wuhaoliu/Downloads/02_First_Review/First_Review/00_compare_methods/02_Tensor-FLAMINGO/src

python Paralized_Low_rank_tensor_completion_FFTW.py \
    -i /path/to/output/tensor_assembly \
    -o /path/to/output/tensor_completion \
    -s completed_tensor \
    -t 1e-4 \
    -max_iter 500 \
    -mu 1e-4 \
    -max_mu 1e10 \
    -rho 1.1 \
    -n_core 10
```

### 参数说明

| 参数      | 含义                             | 推荐值           |
| :-------- | :------------------------------- | :--------------- |
| -i        | 输入目录（包含稀疏 IF 矩阵文件） | -                |
| -o        | 输出目录                         | -                |
| -s        | 输出文件前缀                     | completed_tensor |
| -t        | ADMM 收敛容差                    | 1e-4             |
| -max_iter | 最大迭代次数                     | 500              |
| -mu       | 初始对偶变量步长                 | 1e-4             |
| -max_mu   | 最大步长                         | 1e10             |
| -rho      | 步长增长因子 (≥1)                | 1.1              |
| -n_core   | 并行核心数                       | 10               |

### 注意事项

1. **输入文件格式**: 代码期望读取的是 **IF 矩阵**而非距离矩阵
2. **文件命名**: 确保输入目录中只有需要处理的矩阵文件
3. **内存需求**: 500×500×30 的张量约需 ~180MB (float64)
4. **运行时间**: 取决于收敛速度，通常 10-50 次迭代

------

## 步骤 4: 提取补全结果

使用现有的 Extract_matrix_from_LRTC.py:

```
python Extract_matrix_from_LRTC.py \
    -i /path/to/output/tensor_completion/completed_tensor.npy \
    -o /path/to/output/completed_matrices \
    -t 1 \
    -alpha 0.25
```

### 输出文件

- IF_Cell_X.txt: 补全后的 IF 矩阵
- PD_Cell_X.txt: 通过 PD = IF^(-0.25) 转换的距离矩阵

### 阈值说明

- -t 1: 仅保留 IF ≥ 1 的值为有效数据（过滤噪声）
- -alpha 0.25: 使用论文标准转换公式

------

## 步骤 5: 质量评价

创建评价脚本 evaluate_completion.py:

```
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import os
import glob

def load_completed_matrices(output_dir):
    """加载补全后的 PD 矩阵"""
    pd_files = sorted(glob.glob(os.path.join(output_dir, 'PD_Cell_*.txt')))
    matrices = []
    for f in pd_files:
        mat = pd.read_csv(f, header=None, sep='\t').values
        matrices.append(mat)
    return np.array(matrices), pd_files

def load_ground_truth(gt_dir, n_consensus=3):
    """加载 GT 完整距离矩阵"""
    gt_matrices = []
    for i in range(1, n_consensus+1):
        gt_file = os.path.join(gt_dir, f'consensus_{i}.txt')
        coords = pd.read_csv(gt_file, header=None, sep='\t').values
        gt_dist = cdist(coords, coords)
        gt_matrices.append(gt_dist)
    return gt_matrices

def evaluate_spearman_correlation(completed_pd, gt_pd, mask):
    """计算 Spearman 相关系数（基于距离矩阵）
    
    Args:
        completed_pd: 补全后的距离矩阵
        gt_pd: GT 距离矩阵
        mask: 布尔掩码，指定要比较的位置（通常是原缺失位置）
    
    Returns:
        spearman_corr: Spearman 相关系数
        p_value: p 值
    """
    comp_vals = completed_pd[mask]
    gt_vals = gt_pd[mask]
    
    corr, p_value = spearmanr(comp_vals, gt_vals)
    return corr, p_value

def evaluate_rmse(completed_pd, gt_pd, mask):
    """计算 RMSE（均方根误差）
    
    Args:
        completed_pd: 补全后的距离矩阵
        gt_pd: GT 距离矩阵
        mask: 布尔掩码
    
    Returns:
        rmse: 均方根误差
    """
    comp_vals = completed_pd[mask]
    gt_vals = gt_pd[mask]
    rmse = np.sqrt(np.mean((comp_vals - gt_vals) ** 2))
    return rmse

def evaluate_relative_error(completed_pd, gt_pd, mask):
    """计算相对误差
    
    Returns:
        rel_error: 相对误差 = ||X_comp - X_gt|| / ||X_gt||
    """
    comp_vals = completed_pd[mask]
    gt_vals = gt_pd[mask]
    rel_error = np.linalg.norm(comp_vals - gt_vals) / np.linalg.norm(gt_vals)
    return rel_error

def main():
    # 配置路径
    completed_dir = '/path/to/output/completed_matrices'
    gt_dir = '/path/to/simulation/benchmark_consensus_structure'
    sparse_dir = '/path/to/simulation/downsampled_data'
    output_eval = '/path/to/output/evaluation_results'
    
    os.makedirs(output_eval, exist_ok=True)
    
    # 1. 加载补全结果
    completed_pds, comp_files = load_completed_matrices(completed_dir)
    print(f"加载 {len(comp_files)} 个补全矩阵")
    
    # 2. 加载 GT
    gt_distances = load_ground_truth(gt_dir)
    print(f"加载 {len(gt_distances)} 个 GT 距离矩阵")
    
    # 3. 加载原始稀疏数据（确定哪些位置是缺失的）
    sparse_files = sorted(glob.glob(os.path.join(sparse_dir, '*.txt')))
    
    # 4. 逐细胞评估
    results = []
    
    for cell_idx in range(len(completed_pds)):
        # 确定该细胞属于哪个 consensus
        consensus_id = cell_idx // 10  # 每 10 个细胞共享一个 GT
        gt_pd = gt_distances[consensus_id]
        
        # 读取原始稀疏矩阵
        sparse_df = pd.read_csv(sparse_files[cell_idx], header=None, sep='\t')
        sparse_mat = sparse_df.values
        
        # 定义评估掩码
        observed_mask = sparse_mat > 0  # 已观测位置
        missing_mask = sparse_mat == 0  # 缺失位置（需要插补的位置）
        
        # 排除对角线
        n = completed_pds[cell_idx].shape[0]
        diag_mask = np.eye(n, dtype=bool)
        valid_missing_mask = missing_mask & (~diag_mask)
        
        # 计算指标
        # (a) 缺失位置的 Spearman 相关
        spearman_corr, p_val = evaluate_spearman_correlation(
            completed_pds[cell_idx], gt_pd, valid_missing_mask
        )
        
        # (b) 缺失位置的 RMSE
        rmse = evaluate_rmse(
            completed_pds[cell_idx], gt_pd, valid_missing_mask
        )
        
        # (c) 缺失位置的相对误差
        rel_err = evaluate_relative_error(
            completed_pds[cell_idx], gt_pd, valid_missing_mask
        )
        
        # (d) 所有位置的 Spearman 相关（包括已观测）
        all_mask = ~diag_mask
        spearman_all, p_val_all = evaluate_spearman_correlation(
            completed_pds[cell_idx], gt_pd, all_mask
        )
        
        results.append({
            'cell_idx': cell_idx,
            'consensus_id': consensus_id,
            'spearman_missing': spearman_corr,
            'p_value_missing': p_val,
            'rmse_missing': rmse,
            'relative_error_missing': rel_err,
            'spearman_all': spearman_all,
            'p_value_all': p_val_all
        })
        
        print(f"Cell {cell_idx} (Consensus {consensus_id}):")
        print(f"  Spearman (missing): {spearman_corr:.4f} (p={p_val:.2e})")
        print(f"  RMSE (missing): {rmse:.6f}")
        print(f"  Relative Error: {rel_err:.4f}")
        print(f"  Spearman (all): {spearman_all:.4f}")
    
    # 5. 汇总统计
    results_df = pd.DataFrame(results)
    summary = results_df.groupby('consensus_id').agg({
        'spearman_missing': ['mean', 'std'],
        'rmse_missing': ['mean', 'std'],
        'relative_error_missing': ['mean', 'std'],
        'spearman_all': ['mean', 'std']
    })
    
    print("\n=== 按 Consensus 分组的统计摘要 ===")
    print(summary)
    
    # 6. 保存结果
    results_df.to_csv(os.path.join(output_eval, 'cell_level_evaluation.csv'), index=False)
    summary.to_csv(os.path.join(output_eval, 'consensus_summary.csv'))
    
    # 7. 全局平均
    global_avg = {
        'avg_spearman_missing': results_df['spearman_missing'].mean(),
        'avg_rmse_missing': results_df['rmse_missing'].mean(),
        'avg_relative_error': results_df['relative_error_missing'].mean(),
        'avg_spearman_all': results_df['spearman_all'].mean()
    }
    
    print("\n=== 全局平均指标 ===")
    for k, v in global_avg.items():
        print(f"{k}: {v:.4f}")
    
    pd.DataFrame([global_avg]).to_csv(
        os.path.join(output_eval, 'global_average.csv'), index=False
    )

if __name__ == '__main__':
    main()
```

------

## 评价指标说明

### 主要指标（论文明确使用）

1. **Spearman 相关系数**
   - **计算对象**: 距离矩阵的下三角部分
   - **意义**: 衡量 rank ordering 的一致性
   - **范围**: [-1, 1]，越接近 1 越好
   - **论文报告**: 通常在 0.8-0.95 之间
2. **RMSD (Root Mean Square Deviation)**
   - **计算对象**: 3D 坐标（本指南不涉及，因跳过重建）
   - **如需计算**: 需先进行 3D 重建

### 补充指标（适合 IF/PD 矩阵评估）

1. **RMSE (Root Mean Square Error)**
   - **计算公式**: √(mean((PD_comp - PD_gt)²))
   - **适用场景**: 缺失位置的绝对误差
   - **单位**: 与距离相同
2. **相对误差**
   - **计算公式**: ||PD_comp - PD_gt|| / ||PD_gt||
   - **意义**: 归一化的整体误差
   - **范围**: [0, 1]，越小越好
3. **矩阵恢复精度**
   - **计算公式**: 1 - relative_error
   - **意义**: 恢复的准确度百分比

------

## 完整工作流程

```
# 1. 数据准备：转换并组装张量
python assemble_tensor.py

# 2. 运行 t-SVD 补全
cd /path/to/Tensor-FLAMINGO/src
python Paralized_Low_rank_tensor_completion_FFTW.py \
    -i /path/to/output/tensor_assembly \
    -o /path/to/output/tensor_completion \
    -s completed_tensor \
    -t 1e-4 -max_iter 500 -mu 1e-4 -max_mu 1e10 -rho 1.1 -n_core 10

# 3. 提取补全矩阵
python Extract_matrix_from_LRTC.py \
    -i /path/to/output/tensor_completion/completed_tensor.npy \
    -o /path/to/output/completed_matrices \
    -t 1 -alpha 0.25

# 4. 评估质量
python evaluate_completion.py
```

------

## 预期结果示例

```
=== 全局平均指标 ===
avg_spearman_missing: 0.8742
avg_rmse_missing: 0.0234
avg_relative_error: 0.1156
avg_spearman_all: 0.9123
```

### 解读

- **Spearman (missing) = 0.87**: 补全的缺失位置与 GT 有强正相关
- **RMSE = 0.023**: 平均距离误差约 0.023 单位
- **相对误差 = 0.12**: 整体误差约 12%，即恢复精度 88%
- **Spearman (all) = 0.91**: 包含已观测位置后相关性更高



### Q1: 如何判断补全是否收敛？

**A**: 查看日志中的 Error 值，应随迭代递减。如果达到 tol=1e-4 或 max_iter=500 则停止。

### Q2: Spearman 相关很低怎么办？

**A**: 可能原因：

1. 稀疏度过高（>99.5%），信息不足
2. W 参数设置不当，consensus 间差异过大
3. 噪声水平过高（Level-2）
4. 尝试调整 -mu 和 -rho 参数



------

## 参考文献

- Tensor-FLAMINGO Methods, Line 1200-1205: 评价指标基于距离矩阵的 Spearman 相关和 RMSD
- Paper Formula: IF = PD^(-α), 其中 α=0.25 为标准值
- t-SVD Algorithm: 基于 ADMM 优化的低秩张量补全
