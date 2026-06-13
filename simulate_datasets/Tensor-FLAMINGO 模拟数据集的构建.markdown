# Tensor-FLAMINGO 模拟数据集的构建

Tensor-FLAMINGO 是一个用于从 scHiC 距离（PD）矩阵/接触频率（IF）矩阵重建染色体三维结构的数学算法模型。其提供了一个模拟数据集的构建方法，即从共识3D结构生成模拟scHi-C距离矩阵数据，具体流程如下：

``````
Step 0: 参数配置
  ↓
Step 1: 生成/加载共识3D坐标 (GT) # N种细胞类型下有 N 个共识3D 坐标
  ↓
Step 2: 计算完整欧氏距离矩阵 (GT距离) # N 个 PD 矩阵
  ↓
Step 3: 下采样（模拟实验稀疏性）# 可产生 N*M 个细胞
  ↓
Step 4: 添加高斯噪声（可选，多级别）
  ↓
Step 5: 保存模拟数据 # N 个 GT距离矩阵+N*M个降采样加噪距离矩阵
``````

具体代码：`generate_simulation_data.py`，实现从共识3D结构生成模拟scHi-C距离矩阵数据的完整流程。

#### 1. FLAMIHNGO 模拟数据集的实际构建

![image-20260613212432721](/Users/wuhaoliu/Library/Application Support/typora-user-images/image-20260613212432721.png)





## 🔧 关键参数定义（保留可配置空间）

### 1️⃣ 基础结构参数

```python
STRUCTURE_PARAMS = {
    # 位点数量（基因组分辨率）
    'n_beads': 500,              # 可选: 300, 400, 500, 600, ..., 1000
    
    # 共识结构数量（细胞类型数）
    'n_consensus': 3,            # 可选: 1-5
    
    # 每个共识结构的单细胞变体数
    'n_cells_per_consensus': 10, # 可选: 5-20
    
    # 3D坐标生成方法
    'coord_generation_method': 'sarw',  # 可选: 'sarw' (自回避随机行走), 
                                         #      'chromatin_polymer', 
                                         #      'load_from_file'
}
```

### 2️⃣ 相似性控制参数（W参数）

```python
SIMILARITY_PARAMS = {
    # W参数：控制不同consensus之间的结构相似度
    # 0 < W < 1
    # W越小 → 不同consensus差异越大 → 细胞类型异质性越高
    'W_values': [0.6, 0.7, 0.8],  # 可配置多个值进行对比实验
    
    # 实现方式说明：
    # consensus_2 = W * consensus_1 + (1-W) * random_perturbation
    # consensus_3 = W * consensus_1 + (1-W) * different_random_perturbation
}
```

### 3️⃣ 噪声参数

```python
NOISE_PARAMS = {
    # 噪声级别定义（参考论文Methods第1200行）
    'noise_levels': {
        'level_0': {
            'enabled': True,
            'distribution': None,  # 无噪声
            'description': 'No noise - pure benchmark'
        },
        'level_1': {
            'enabled': True,
            'mean_multiplier': 1.0,   # Normal(δ, δ)
            'std_multiplier': 1.0,
            'description': 'Light noise: Normal(δ, δ)'
        },
        'level_2': {
            'enabled': True,
            'mean_multiplier': 2.0,   # Normal(2δ, δ)
            'std_multiplier': 1.0,
            'description': 'Heavy noise: Normal(2δ, δ)'
        }
    },
    
    # δ的计算方式
    'delta_calculation': 'min_nonzero_distance',  # δ = min(下采样后的非零距离)
}
```

### 4️⃣ 下采样参数

```python
DOWNSAMPLING_PARAMS = {
    # 下采样率（保留比例）
    'retention_rates': [0.005],  # 可选: 0.005 (0.5%), 0.01, 0.05, 0.1, ...
                                 # 论文中使用 ~0.5% 模拟真实scHi-C稀疏性
    
    # 下采样策略
    'strategy': 'random_uniform',  # 可选: 'random_uniform', 
                                   #      'distance_dependent',
                                   #      'realistic_scHiC_pattern'
    
    # 是否保持对称性
    'preserve_symmetry': True,
    
    # 是否保持对角线为0
    'zero_diagonal': True,
}
```

### 5️⃣ 输出参数

```python
OUTPUT_PARAMS = {
    # 输出目录结构
    'output_base_dir': './simulation_generated',
    
    # 子目录命名
    'benchmark_dir': 'benchmark_consensus_structure',
    'downsampled_dir': 'downsampled_data',
    
    # 文件命名格式
    'consensus_filename': 'consensus_{idx}.txt',           # idx: 1, 2, 3...
    'sparse_filename': 'consensus_{c_idx}_slice_{s_idx}.txt', # c_idx: consensus编号, s_idx: 样本编号
    
    # 数据格式
    'matrix_format': 'dense_txt',  # 可选: 'dense_txt', 'sparse_npz', 'hdf5'
    'delimiter': '\t',
    'precision': '%.6f',
}
```

---

## 💻 必须实现的核心函数

### 函数1：生成共识3D结构

```python
def generate_consensus_structure(
    n_beads: int = 500,
    method: str = 'sarw',
    seed: int = None,
    **kwargs
) -> np.ndarray:
    """
    生成单个共识结构的3D坐标
    
    Args:
        n_beads: 位点数量（染色体beads数）
        method: 生成方法
            - 'sarw': Self-Avoiding Random Walk（自回避随机行走）
            - 'chromatin_polymer': 染色质聚合物模型
            - 'load_from_file': 从文件加载已有坐标
        seed: 随机种子（用于复现）
        kwargs: 其他方法特定参数
    
    Returns:
        coords: 3D坐标数组，shape=(n_beads, 3)
    
    Example:
        >>> coords = generate_consensus_structure(n_beads=500, seed=42)
        >>> print(coords.shape)
        (500, 3)
    """
    # TODO: 实现代码
    pass
```

---

### 函数2：生成多个具有可控相似度的共识结构

```python
def generate_multiple_consensus_structures(
    n_consensus: int = 3,
    n_beads: int = 500,
    W_values: list = None,
    base_seed: int = 42
) -> list:
    """
    生成多个共识结构，通过W参数控制它们之间的相似度
    
    Args:
        n_consensus: 共识结构数量
        n_beads: 每个结构的位点数
        W_values: 相似度参数列表 (0 < W < 1)
                  - W接近1 → 结构非常相似
                  - W接近0 → 结构差异很大
        base_seed: 基础随机种子
    
    Returns:
        consensus_list: 包含n_consensus个坐标数组的列表
                       每个数组shape=(n_beads, 3)
    
    实现逻辑:
        1. 生成第一个consensus作为基准
        2. 对于后续的consensus i:
           consensus_i = W * consensus_1 + (1-W) * perturbation_i
           其中perturbation_i是随机扰动
    
    Example:
        >>> structures = generate_multiple_consensus_structures(
        ...     n_consensus=3, n_beads=500, W_values=[0.6, 0.7, 0.8]
        ... )
        >>> len(structures)
        3
    """
    # TODO: 实现代码
    pass
```

---

### 函数3：从3D坐标计算距离矩阵

```python
def coords_to_distance_matrix(
    coords: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    从3D坐标计算成对欧氏距离矩阵
    
    Args:
        coords: 3D坐标数组，shape=(n_beads, 3)
        normalize: 是否归一化距离到[0, 1]范围
    
    Returns:
        dist_matrix: 对称距离矩阵，shape=(n_beads, n_beads)
                    - 对角线为0
                    - 对称: dist[i,j] == dist[j,i]
                    - 所有值 >= 0
    
    Example:
        >>> coords = np.random.randn(500, 3)
        >>> dist = coords_to_distance_matrix(coords)
        >>> print(dist.shape)
        (500, 500)
        >>> print(np.allclose(dist, dist.T))
        True
    """
    from scipy.spatial.distance import cdist
    # TODO: 实现代码
    pass
```

---

### 函数4：添加高斯噪声

```python
def add_noise_to_distance_matrix(
    dist_matrix: np.ndarray,
    noise_level: str = 'level_1',
    delta: float = None
) -> np.ndarray:
    """
    向距离矩阵添加高斯噪声（参考论文Methods第1200行）
    
    Args:
        dist_matrix: 原始距离矩阵，shape=(n, n)
        noise_level: 噪声级别
            - 'level_0': 无噪声
            - 'level_1': Normal(δ, δ)
            - 'level_2': Normal(2δ, δ)
        delta: 噪声尺度参数
               如果为None，则 delta = min(dist_matrix[dist_matrix > 0])
    
    Returns:
        noisy_dist: 添加噪声后的距离矩阵
                   （需要确保所有值 >= 0）
    
    实现逻辑:
        if noise_level == 'level_0':
            return dist_matrix
        elif noise_level == 'level_1':
            if delta is None:
                delta = np.min(dist_matrix[dist_matrix > 0])
            noise = np.random.normal(loc=delta, scale=delta, size=dist_matrix.shape)
        elif noise_level == 'level_2':
            if delta is None:
                delta = np.min(dist_matrix[dist_matrix > 0])
            noise = np.random.normal(loc=2*delta, scale=delta, size=dist_matrix.shape)
        
        noisy_dist = dist_matrix + noise
        noisy_dist = np.maximum(noisy_dist, 0)  # 确保非负
        
        return noisy_dist
    
    Example:
        >>> dist = np.random.rand(100, 100)
        >>> noisy = add_noise_to_distance_matrix(dist, noise_level='level_1')
    """
    # TODO: 实现代码
    pass
```

---

### 函数5：下采样（模拟实验稀疏性）

```python
def downsample_distance_matrix(
    dist_matrix: np.ndarray,
    retention_rate: float = 0.005,
    strategy: str = 'random_uniform',
    preserve_symmetry: bool = True,
    zero_diagonal: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    对距离矩阵进行下采样，模拟scHi-C实验的极端稀疏性
    
    Args:
        dist_matrix: 完整距离矩阵，shape=(n, n)
        retention_rate: 保留比例 (0 < rate < 1)
                       - 0.005 = 0.5% (论文设置)
                       - 0.01 = 1%
                       - 0.1 = 10%
        strategy: 下采样策略
            - 'random_uniform': 均匀随机保留
            - 'distance_dependent': 基于距离的概率（近距离更易保留）
            - 'realistic_scHiC_pattern': 使用真实scHi-C的稀疏模式
        preserve_symmetry: 是否保持矩阵对称性
        zero_diagonal: 是否强制对角线为0
        seed: 随机种子
    
    Returns:
        sparse_dist: 稀疏距离矩阵
                    - 大部分元素为0
                    - 稀疏度 ≈ (1 - retention_rate) * 100%
    
    实现逻辑 (random_uniform):
        np.random.seed(seed)
        mask = np.random.random(dist_matrix.shape) < retention_rate
        
        if zero_diagonal:
            np.fill_diagonal(mask, False)
        
        if preserve_symmetry:
            mask = mask | mask.T  # 保持对称
        
        sparse_dist = dist_matrix * mask
        
        return sparse_dist
    
    Example:
        >>> dist = np.random.rand(100, 100)
        >>> sparse = downsample_distance_matrix(dist, retention_rate=0.005)
        >>> sparsity = 100 * (1 - np.count_nonzero(sparse) / sparse.size)
        >>> print(f"稀疏度: {sparsity:.2f}%")
        稀疏度: 99.50%
    """
    # TODO: 实现代码
    pass
```

---

### 函数6：完整生成流程

```python
def generate_full_simulation_dataset(
    structure_params: dict = None,
    similarity_params: dict = None,
    noise_params: dict = None,
    downsampling_params: dict = None,
    output_params: dict = None,
    verbose: bool = True
) -> dict:
    """
    生成完整的模拟数据集（主函数）
    
    Args:
        structure_params: 结构参数（见STRUCTURE_PARAMS定义）
        similarity_params: 相似性参数（见SIMILARITY_PARAMS定义）
        noise_params: 噪声参数（见NOISE_PARAMS定义）
        downsampling_params: 下采样参数（见DOWNSAMPLING_PARAMS定义）
        output_params: 输出参数（见OUTPUT_PARAMS定义）
        verbose: 是否打印详细日志
    
    Returns:
        dataset_info: 包含生成数据集信息的字典
            {
                'n_consensus': 生成的共识结构数,
                'n_beads': 位点数,
                'n_total_cells': 总细胞数 (n_consensus * n_cells_per_consensus),
                'n_noise_levels': 噪声级别数,
                'n_downsample_rates': 下采样率数,
                'output_dir': 输出目录路径,
                'files_generated': 生成的文件列表,
            }
    
    完整流程:
        Step 1: 生成多个共识3D结构
                ↓
        Step 2: 对每个consensus计算完整距离矩阵 (GT)
                ↓
        Step 3: 对每个consensus生成n_cells个变体:
                - 选择噪声级别 (level_0/1/2)
                - 添加噪声
                - 下采样（指定保留率）
                ↓
        Step 4: 保存所有数据
                - GT坐标: benchmark_consensus_structure/consensus_X.txt
                - 稀疏距离: downsampled_data/consensus_X_slice_Y.txt
    
    Example:
        >>> info = generate_full_simulation_dataset(
        ...     structure_params={'n_beads': 500, 'n_consensus': 3},
        ...     similarity_params={'W_values': [0.6, 0.7, 0.8]},
        ...     noise_params={'noise_levels': ['level_0', 'level_1', 'level_2']},
        ...     downsampling_params={'retention_rates': [0.005]},
        ...     output_params={'output_base_dir': './my_simulation'}
        ... )
        >>> print(f"生成了 {info['n_total_cells']} 个细胞的模拟数据")
    """
    # TODO: 实现代码
    pass
```

---

## 📁 期望的输出目录结构

```
simulation_generated/
├── README.md                          # 自动生成，描述数据集信息
├── params.json                        # 记录所有使用的参数
│
├── benchmark_consensus_structure/     # GT数据目录
│   ├── consensus_1.txt                # 共识结构1的3D坐标 (500×3)
│   ├── consensus_2.txt                # 共识结构2的3D坐标 (500×3)
│   └── consensus_3.txt                # 共识结构3的3D坐标 (500×3)
│
└── downsampled_data/                  # 下采样数据目录
    ├── consensus_1_slice_1.txt        # consensus_1的第1个细胞（稀疏距离矩阵 500×500）
    ├── consensus_1_slice_2.txt        # consensus_1的第2个细胞
    ...
    ├── consensus_1_slice_10.txt       # consensus_1的第10个细胞
    │
    ├── consensus_2_slice_1.txt        # consensus_2的第1个细胞
    ...
    └── consensus_3_slice_10.txt       # consensus_3的第10个细胞

总计: 3个GT文件 + 30个稀疏矩阵文件 = 33个文件
```

---

## 📝 文件格式规范

### GT坐标文件格式

```txt
# consensus_1.txt
# 格式: 纯文本，每行一个bead的3D坐标
# 行数: n_beads (如500)
# 列数: 3 (X, Y, Z)
# 分隔符: 空格或制表符

0.376  0.421  0.503
0.382  0.428  0.511
0.391  0.435  0.519
...
```

### 稀疏距离矩阵格式

```txt
# consensus_1_slice_1.txt
# 格式: 纯文本，方阵
# 行列数: n_beads × n_beads (如500×500)
# 值: 浮点数，6位小数精度
# 缺失值: 0（表示未观测到）
# 特性: 对称矩阵，对角线为0

0.000000  0.000000  0.023456  0.000000  ...
0.000000  0.000000  0.000000  0.031245  ...
0.023456  0.000000  0.000000  0.000000  ...
...
```

---

## ✅ 验证要求

生成的数据必须通过以下验证：

```python
def validate_generated_data(output_dir: str):
    """验证生成的模拟数据是否符合规范"""
    
    # 验证1: GT坐标形状正确
    gt = np.loadtxt(f'{output_dir}/benchmark_consensus_structure/consensus_1.txt')
    assert gt.shape == (n_beads, 3), f"GT形状错误: {gt.shape}"
    
    # 验证2: 距离矩阵对称且对角线为0
    sparse = np.loadtxt(f'{output_dir}/downsampled_data/consensus_1_slice_1.txt')
    assert sparse.shape == (n_beads, n_beads), "距离矩阵形状错误"
    assert np.allclose(sparse, sparse.T, equal_nan=True), "矩阵不对称"
    assert np.all(np.diag(sparse) == 0), "对角线不为0"
    
    # 验证3: 稀疏度符合预期
    non_zero = np.count_nonzero(sparse)
    total = sparse.size
    actual_retention = non_zero / total
    expected_retention = retention_rate  # 考虑对称性可能需要调整
    assert abs(actual_retention - expected_retention) < 0.001, \
        f"稀疏度不符: 实际{actual_retention:.4f}, 期望{expected_retention:.4f}"
    
    # 验证4: 数值范围合理（无负值）
    assert np.all(sparse >= 0), "存在负值"
    
    # 验证5: 从GT计算的距離与存储的距离一致（在噪声范围内）
    from scipy.spatial.distance import cdist
    calc_dist = cdist(gt, gt)
    # ... 比较逻辑
    
    print("✅ 所有验证通过！")
```

---

## 🚀 使用示例

### 示例1：基本用法（复现论文设置）

```python
from generate_simulation_data import generate_full_simulation_dataset

# 使用默认参数（复现Tensor-FLAMINGO论文设置）
info = generate_full_simulation_dataset()

print(f"生成了 {info['n_total_cells']} 个细胞")
print(f"输出目录: {info['output_dir']}")
```

### 示例2：自定义参数

```python
# 自定义参数配置
params = {
    'structure_params': {
        'n_beads': 1000,          # 更高分辨率
        'n_consensus': 5,         # 5种细胞类型
        'n_cells_per_consensus': 15,  # 每种15个细胞
    },
    'similarity_params': {
        'W_values': [0.5, 0.6, 0.7, 0.8, 0.9],  # 不同的相似度
    },
    'noise_params': {
        'noise_levels': ['level_0', 'level_1', 'level_2'],
    },
    'downsampling_params': {
        'retention_rates': [0.005, 0.01, 0.05],  # 多种稀疏度
    },
    'output_params': {
        'output_base_dir': './simulation_high_res',
    }
}

info = generate_full_simulation_dataset(**params)
```

### 示例3：只生成单个consensus用于快速测试

```python
# 快速测试配置
info = generate_full_simulation_dataset(
    structure_params={'n_beads': 300, 'n_consensus': 1, 'n_cells_per_consensus': 5},
    noise_params={'noise_levels': ['level_0']},  # 无噪声
    downsampling_params={'retention_rates': [0.01]},  # 1%保留
    output_params={'output_base_dir': './test_simulation'}
)
```

---

## 📚 参考文献和依据

### 论文原文引用

**Methods 第1200行：**

> "The l by l pairwise spatial distance matrix of each consensus structure is used to generate N different distance matrices of single cells, based on random down-sampling of entries and adding random noise."

> "The added white noise takes three different levels as suggested by previous studies, including level-zero: no noise; level-one: generated by the normal distribution Normal(δ, δ); and level-two: generated by Normal(2δ, δ); where δ represents the minimum value from the down-sampled pairwise distances."

### 参数设置依据

| 参数                  | 论文值                      | 来源                      |
| --------------------- | --------------------------- | ------------------------- |
| n_beads               | 500 (简化版), 1000 (完整版) | 论文Fig.2说明             |
| n_consensus           | 3                           | simulation目录实际数据    |
| n_cells_per_consensus | 10                          | simulation目录实际数据    |
| retention_rate        | ~0.005 (0.5%)               | 实际数据统计: 99.5%稀疏度 |
| noise_level_1         | Normal(δ, δ)                | Methods第1200行           |
| noise_level_2         | Normal(2δ, δ)               | Methods第1200行           |

---

## 