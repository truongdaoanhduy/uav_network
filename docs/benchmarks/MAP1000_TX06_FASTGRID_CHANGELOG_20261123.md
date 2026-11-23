# FASTGRID MAP1000 x MAP1000, TX0.6W — 23 Nov 2026

Source parent `a3ba05e03bf4ca4e7667cf6a037c6eaea9115050` (MAP800 TX0.8 FASTGRID). Independent branch `experiment/map1000-tx06-posterioronly-fastgrid-20261123`; old variants unmodified.

Only science changes:

| Key | Previously | New |
|---|---:|---:|
| `world.map_size_m` | 800 | **1000** (square 1000m×1000m) |
| `network.tx_power_max_w` | 0.8 | **0.6** |
| `sensor.grid_cell_m` | 20 | **25** (maintains 40×40 cells) |

`network.tx_power_min_w=0.001`, `reward.communication_reference_tx_power_w=0.4`, Posterior-only camera-positive threshold0.99, sensor Pd/Pf by altitude, No LR4/50m confirmation gate, target count35, obstacles25, six UAVs and learned radio power remain unchanged. Identical H256x256 orthogonal actorBN/plaincritic, torch.compile, E4096/U4/B8192, replay75%CPU/25%GPU, 20k episode budget, periodic Actor every100 vector steps, Val100 per Actor on two GPUs, Test1000, HF forensic, W&B and FASTGRID performance optimizations. User asked for a new map scenario; old checkpoints and run names are NOT reused.

Notebook: 3-cell, fully embedded source with SHA256, mandatory cloud preflight tests retained, Kaggle size under1MB, unique output directory and new W&B run ID. Must import on Kaggle, T4×2 with Internet and `WANDB_API_KEY`, `HF_TOKEN` Secrets. GPU train and Eval/Test were NOT performed locally. Unit/regression tests: **22 passed** on local CPU before committing the new notebook. This does NOT predict model quality on the larger map.
