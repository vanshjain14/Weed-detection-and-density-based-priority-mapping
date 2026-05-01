# 🌿 WeedGuard AI — Intelligent Weed Detection & Autonomous Spray Control

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PPO-Stable--Baselines3-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?style=for-the-badge&logo=kaggle"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> An end-to-end AI pipeline that detects weeds using computer vision, maps weed density across a field, and trains a Reinforcement Learning agent to autonomously control herbicide spray levels — simulating precision UAV-based agriculture.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Datasets](#-datasets)
- [Model Details](#-model-details)
- [Reinforcement Learning Design](#-reinforcement-learning-design)
- [Novelty](#-novelty)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Requirements](#-requirements)
- [Future Work](#-future-work)

---

## 🔍 Overview

Traditional weed management applies herbicide uniformly across entire fields, wasting chemicals and harming soil health. This project proposes a **three-stage intelligent system**:

1. **Detect** weeds in field images using a fine-tuned YOLOv8 model (binary: Weed vs Non-Weed)
2. **Map** the detected weed density across a synthetic 20×20 field grid using YOLO inference + DBSCAN spatial clustering
3. **Control** spray levels intelligently using a PPO Reinforcement Learning agent trained on the density map

The RL agent learns to traverse the entire field in a boustrophedon (snake) path — visiting every cell exactly once — and choose the optimal herbicide spray level (0 = none, 1 = low, 2 = medium, 3 = high) based on the weed density at each location.

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│              STAGE 1 — Dataset Preparation               │
│  DeepWeeds (9 classes)   +   MH-Weed16 (16 classes)     │
│        ↓ binary remap              ↓ binary remap        │
│         Weed (1)  /  Non-Weed (0)                        │
│              70% train / 20% val / 10% test              │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              STAGE 2 — YOLOv8 Training                   │
│  Model: yolov8l  │  Epochs: 80  │  Optimizer: AdamW     │
│  Augmentation: mosaic, mixup, HSV, flips, scale          │
│  Confidence threshold sweep → best F1 threshold          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│           STAGE 3 — Synthetic Field Mapping              │
│  400 validation images → 20×20 grid                      │
│  YOLO counts weed detections per patch                   │
│  Gaussian smoothing (σ=1.2) → raw + smoothed maps saved  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            STAGE 4 — DBSCAN Clustering                   │
│  Threshold: cells above p75 weed density                 │
│  DBSCAN (eps=2.0, min_samples=2) → cluster centroids     │
│  Centroid = argmax cell (peak density, not mean coords)  │
│  Isolated hotspot recovery: noise cells above p90        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│         STAGE 5 — RL Environment (SnakeSprayEnv)         │
│  Observation: normalised weed density (single float)     │
│  Action space: Discrete(4) — spray levels 0 to 3         │
│  Traversal: boustrophedon (snake), 400 cells / episode   │
│  Reward: +15 correct spray, penalty for error,           │
│          +20 bonus at DBSCAN cluster centroids           │
│  Thresholds: percentile-adaptive (p25/p50/p75)           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│             STAGE 6 — PPO Training                       │
│  Policy: MlpPolicy  │  Envs: 4 parallel                  │
│  Timesteps: 300 000  │  gamma: 0.99  │  lr: 1e-3         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│           STAGE 7 — Evaluation & Visualization           │
│  Spray accuracy, cluster hit rate, spray distribution    │
│  DBSCAN heatmap + snake traversal overlay                │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Datasets

### DeepWeeds
- **Source:** [Kaggle — DeepWeeds](https://www.kaggle.com/datasets/imsparsh/deepweeds)
- **Classes:** 8 weed species + 1 negative (non-weed)
- **Remapping:** Labels 0–7 → Weed (1), Label 8 → Non-Weed (0)
- **Sampling:** Balanced 1500 per class, stratified across species

### MH-Weed16
- **Source:** [Kaggle — MH-Weed16](https://www.kaggle.com/datasets/psycho358/mh-weed16)
- **Sensor:** Intel RealSense Depth Camera
- **Classes:** 16 weed species with YOLO bounding box annotations
- **Remapping:** Class 0 (crop) → Non-Weed (0), all others → Weed (1)
- **Split:** 70% train / 20% val / 10% test

Both datasets are structured into YOLO-compatible directories (`images/` + `labels/`) with a `data.yaml` config file.

---

## 🤖 Model Details

### Object Detector: YOLOv8 Large

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | `yolov8l.pt` | Large variant — better capacity for small dataset (~3k images) |
| Input size | 640 × 640 | Standard YOLO input; preserves spatial detail |
| Optimizer | AdamW | Adaptive LR + weight decay; better than SGD for small datasets |
| Learning rate | 0.001 with cosine decay | Smooth annealing prevents oscillation near convergence |
| Label smoothing | 0.1 | Prevents overconfidence, improves generalization |
| Early stopping | patience=20 | Stops training if no val improvement for 20 epochs |
| Mosaic augmentation | 1.0 | Merges 4 images — critical for small dataset variety |
| Mixup | 0.1 | Blends two images for added regularization |

**Data augmentation used:**

- HSV color jitter (h=0.015, s=0.7, v=0.4) — robustness to field lighting
- Rotation ±15°, translation, scale 0.5 — drone viewpoint variation
- Horizontal/vertical flips — weeds are orientation-invariant

**Confidence threshold sweep:** Post-training sweep from 0.25–0.50, selecting the threshold with the best F1-score to balance precision and recall for spraying decisions.

---

## 🧠 Reinforcement Learning Design

### Environment: `SnakeSprayEnv`

```
Observation space : Box(1,) — normalised weed density at current cell
Action space      : Discrete(4) — spray levels 0, 1, 2, 3
Episode length    : 400 steps (20×20 grid, one traversal)
```

**Traversal — Boustrophedon path:**
The agent moves left→right on even rows, right→left on odd rows (like a plowing ox). This guarantees 100% field coverage with zero revisits — the optimal UAV coverage pattern.

**Spray thresholds (percentile-adaptive):**

| Weed density | Correct spray |
|---|---|
| ≤ 25th percentile | 0 — no spray |
| 25th–50th percentile | 1 — low |
| 50th–75th percentile | 2 — medium |
| > 75th percentile | 3 — high |

**Reward function:**

```python
error = abs(predicted_spray - ideal_spray)
spray_reward   = [+15, -5, -15, -30][min(error, 3)]
hotspot_bonus  = +20  if (row, col) in cluster_centroids else 0
total_reward   = spray_reward + hotspot_bonus
```

### Algorithm: PPO (Proximal Policy Optimization)

| Hyperparameter | Value |
|---|---|
| Policy | MlpPolicy |
| Total timesteps | 300 000 |
| Parallel environments | 4 |
| Learning rate | 1e-3 |
| Discount factor γ | 0.99 |
| Entropy coefficient | 0.005 |
| Batch size | 128 |
| Steps per rollout | 1024 |

---

## ✨ Novelty

This project introduces several contributions not commonly found together in agricultural AI systems:

**1. YOLO → DBSCAN → PPO unified pipeline**
Most weed detection papers stop at detection. This project chains the YOLO detection output directly into an RL training environment, with an unsupervised clustering step in between.

**2. DBSCAN-informed RL reward shaping**
Cluster centroids identified by DBSCAN are used to inject a hotspot bonus (+20) into the PPO reward function. This is a novel application of unsupervised learning to guide RL policy learning — the agent is explicitly taught to prioritize statistically identified high-infestation zones.

**3. Percentile-adaptive spray thresholds**
Spray level boundaries are computed from the field's actual data distribution (quartiles of the raw density map), not fixed values. The controller adapts to any field or season with no manual re-tuning.

**4. Isolated hotspot recovery**
DBSCAN naturally marks isolated high-density cells as noise. A custom recovery step promotes any noise cell above the 90th percentile back to a cluster centroid — preventing extreme infestation zones from being missed.

**5. Argmax centroid placement**
Cluster centroids are snapped to the peak-density cell in each cluster (not the geometric mean), ensuring the RL bonus always lands on the most critical tile in each zone.

---

## 📊 Results

After 300 000 training timesteps:

| Metric | Value |
|---|---|
| Cells visited | 400 / 400 (100%) |
| Spray accuracy | ~90%+ correct spray decisions |
| Cluster centroids hit | All identified centroids visited |
| Spray distribution | Proportional to field density quartiles |

Visualizations saved:
- `raw_density_heatmap.png` — raw YOLO-detected weed counts per cell
- `smoothed_density_heatmap.png` — Gaussian-smoothed version
- `final_v4_result.png` — DBSCAN map + snake traversal with spray decisions

---

## 📁 Project Structure

```
weedguard-ai/
│
├── weedmodeltrain.ipynb        # YOLO training pipeline (DeepWeeds + MH-Weed16)
├── finalreview__1_.ipynb       # Full end-to-end pipeline (field map + RL)
│
├── models/
│   └── final_v4/
│       └── spray_model_final.zip   # Trained PPO model
│
├── outputs/
│   ├── synthetic_density_map.npy
│   ├── synthetic_density_map_smoothed.npy
│   ├── raw_density_heatmap.png
│   ├── smoothed_density_heatmap.png
│   └── final_v4_result.png
│
├── data/                       # YOLO-format datasets (created at runtime)
│   ├── deepweeds_yolo/
│   └── mhweed_binary_yolo/
│
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/weedguard-ai.git
cd weedguard-ai
```

### 2. Install dependencies

```bash
pip install ultralytics==8.0.196 opencv-python scikit-learn scipy seaborn pandas \
            matplotlib pillow gymnasium stable-baselines3 torch torchvision
```

### 3. Run on Kaggle (recommended)

Both notebooks are designed to run on **Kaggle with GPU (T4 or P4)**.

- Upload `weedmodeltrain.ipynb` and add the DeepWeeds + MH-Weed16 datasets as Kaggle inputs
- Run all cells to train the YOLO model
- Upload `finalreview__1_.ipynb`, add the trained weights as a Kaggle dataset input
- Run all cells to generate the field map, train PPO, and visualize results

### 4. Key paths to configure

```python
# In weedmodeltrain.ipynb
src = Path("/kaggle/input/datasets/imsparsh/deepweeds")

# In finalreview__1_.ipynb
model = YOLO("/kaggle/input/datasets/vanshj14/model2/mhweed16_best_model.pt")
```

---

## 📋 Requirements

| Package | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5.1 |
| Ultralytics | 8.0.196 |
| Stable-Baselines3 | latest |
| Gymnasium | latest |
| OpenCV | latest |
| scikit-learn | latest |
| scipy | latest |
| NumPy | latest |
| Matplotlib | latest |

> GPU recommended (NVIDIA T4 or P4 minimum). CPU training for PPO will work but is slow.

---

## 🔭 Future Work

- **Real UAV integration** — Deploy YOLO model on drone edge hardware (Jetson Nano / Raspberry Pi) for live field inference
- **Multi-agent RL** — Train multiple drones to collaboratively cover large fields
- **GPS-tagged density maps** — Replace synthetic grid with real GPS coordinates for actual farm mapping
- **Weed species classification** — Extend from binary to multi-class detection for species-specific herbicide selection
- **Temporal tracking** — Use multiple flight sessions to track weed spread over time and predict infestation growth

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [DeepWeeds Dataset](https://github.com/AlexOlsen/DeepWeeds)
- [MH-Weed16 Dataset](https://www.kaggle.com/datasets/psycho358/mh-weed16)
