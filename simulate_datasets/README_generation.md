# Tensor-FLAMINGO Simulation Data Generator

`generate_simulation_data.py` generates Tensor-FLAMINGO-style simulated scHi-C distance matrices.

The implemented data flow is:

```text
consensus 3D coordinates
-> dense GT Euclidean distance matrix
-> dense raw IF/contact matrix, contact = distance^(-1/alpha)
-> downsample observed entries Omega
-> delta = min distance on Omega
-> add Normal(delta, delta) or Normal(2*delta, delta) only on Omega
-> sparse noisy distance matrix
-> sparse raw IF/contact matrix
```

No IF/contact matrix is normalized, scaled, or calibrated.

## Outputs

```text
simulation_generated/
├── README.md
├── params.json
├── metadata.csv
├── benchmark_consensus_structure/
│   └── consensus_1.txt
├── gt_distance_matrices/
│   └── consensus_1_distance.txt
├── gt_contact_matrices/
│   └── consensus_1_contact.txt
├── downsampled_data/
│   └── consensus_1_slice_1.txt
└── downsampled_contact/
    └── consensus_1_slice_1.txt
```

`downsampled_data/` is the primary imputation input: dense text matrices with unobserved entries set to `0`.

## Recommended Environment

Use the requested micromamba environment:

```bash
micromamba run -p /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env python generate_simulation_data.py --validate
```

## Small Test Dataset

```bash
micromamba run -p /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env \
  python generate_simulation_data.py \
  --out ./simulation_generated_example \
  --n_beads 50 \
  --n_consensus 2 \
  --n_cells_per_consensus 3 \
  --retention_rate 0.02 \
  --validate
```

Then validate an existing output directory:

```bash
micromamba run -p /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env \
  python test_generation.py ./simulation_generated_example
```

## Python API

```python
from generate_simulation_data import generate_full_simulation_dataset

info = generate_full_simulation_dataset(
    structure_params={"n_beads": 500, "n_consensus": 3, "n_cells_per_consensus": 10},
    similarity_params={"W_values": [0.6, 0.7, 0.8]},
    noise_params={"noise_levels": ["level_0", "level_1", "level_2"]},
    downsampling_params={"retention_rates": [0.005]},
    output_params={"output_base_dir": "./simulation_generated"},
)
```
