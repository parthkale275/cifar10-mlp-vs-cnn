# CIFAR-10: MLP vs CNN — Architectural Analysis
 
Empirical comparison of Multi-Layer Perceptron and Convolutional Neural Network architectures on CIFAR-10, covering capacity analysis, pixel-shuffling invariance, feature map visualization, and translation robustness — all in PyTorch.
 
## Overview
 
This project investigates *why* CNNs outperform MLPs on image data, not just *that* they do. Beyond a standard train-and-report comparison, it includes four targeted experiments designed to isolate specific architectural properties:
 
- **Capacity study** — does adding more layers/neurons to an MLP actually help, or just overfit?
- **Pixel-shuffling test** — proof that MLPs are spatially invariant (they don't care about pixel order), while CNNs depend on it
- **Feature map visualization** — how representations evolve across convolutional layers
- **Translation robustness** — quantifying how much each architecture's accuracy degrades under exact pixel shifts
## Results Summary
 
| Model | Parameters | Test Accuracy |
|---|---|---|
| MLP (3-layer baseline) | 3,805,450 | 57.40% |
| MLP (pixel-shuffled) | 3,805,450 | 57.42% |
| **CNN (with pooling)** | **1,342,122** | **82.01%** |
 
The CNN achieves **~2.8× fewer parameters** while outperforming the MLP baseline by **~24.6 percentage points** — strong empirical evidence that convolutional inductive biases (local connectivity, weight sharing, translation equivariance) are far more parameter-efficient for image data than fully-connected layers.
 
## Key Findings
 
### 1. MLP Capacity Study
 
| Architecture | Params | Train Acc | Val Acc | Test Acc |
|---|---|---|---|---|
| 1L-64 | 197,322 | 53.20% | 51.62% | 51.58% |
| 1L-256 | 789,258 | 65.33% | 55.60% | 55.22% |
| 2L-512x256 | 1,707,274 | 67.74% | 57.20% | 57.07% |
| 2L-1024x512 | 3,676,682 | 73.15% | 57.36% | 57.85% |
| 3L-1024x512x256 | 3,805,450 | 70.59% | 56.88% | 57.86% |
 
Accuracy improves with capacity up to a point, then plateaus — adding a third hidden layer (3.8M params) produces no meaningful improvement over two layers (3.7M params), while the train/val gap widens, indicating overfitting rather than better generalization.
 
### 2. Pixel Shuffling — MLPs Are Spatially Invariant
 
Applying a single fixed pixel permutation (consistent across train and test) and retraining gives **57.42% accuracy** — virtually identical to the unshuffled baseline (57.40%). Since the MLP flattens images into 1D vectors before any computation, it has no notion of which pixels were originally adjacent; it only matters that the *same* scrambled order is used consistently. This is direct evidence that MLPs lack the spatial inductive bias that makes CNNs effective for image data.
 
### 3. Feature Map Evolution (CNN)
 
Visualizing intermediate activations across the three convolutional blocks shows a clear hierarchy:
- **Block 1** (32 ch, 16×16) — simple edge and color-contrast detectors, close to raw pixel patterns
- **Block 2** (64 ch, 8×8) — more abstract structural regions
- **Block 3** (128 ch, 4×4) — high-level, class-discriminative representations feeding the classifier
### 4. Translation Robustness
 
Both models were evaluated on test images shifted by an exact, fixed number of pixels (not random augmentation):
 
| Shift | MLP Accuracy | CNN Accuracy |
|---|---|---|
| 0px | 57.40% | 82.01% |
| 1px | 53.14% | 80.46% |
| 2px | 46.72% | 76.49% |
| 4px | 32.37% | 61.86% |
| 6px | 21.75% | 47.25% |
| 8px | 16.81% | 32.46% |
 
The CNN degrades far more gracefully under small shifts (82.01% → 76.49% at 2px, a ~5.5-point drop) than the MLP (57.40% → 46.72%, a ~10.7-point drop) — direct evidence of the local translation invariance provided by convolution and pooling. At extreme shifts (8px), both architectures converge toward information-limited performance as the shift pushes real content out of frame.
 
## Repository Structure
 
```
cifar10-mlp-vs-cnn/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Loading, normalization, sample & flatten visualization
│   ├── 02_mlp_experiments.ipynb        # MLP baseline, capacity sweep, pixel shuffling, misclassified images
│   ├── 03_cnn_experiments.ipynb        # CNN training, MLP comparison, feature map visualization
│   └── 04_mlp_vs_cnn_robustness.ipynb  # Translation robustness comparison
├── src/
│   ├── data.py         # Dataloaders, ShuffleDataset, apply_shift (fixed pixel translation)
│   ├── models.py        # MLP and CNN architecture definitions
│   ├── train.py         # Training loop, evaluation, misclassified-image extraction
│   └── visualize.py     # Plotting utilities shared across all notebooks
├── results/
│   ├── figures/          # All saved plots (training curves, capacity sweep, feature maps, etc.)
│   ├── metrics.json      # All final numbers in one machine-readable file
│   ├── mlp_model.pt      # Trained MLP weights
│   └── cnn_model.pt      # Trained CNN weights
├── requirements.txt
└── .gitignore
```
 
Each notebook imports shared logic from `src/` rather than redefining it inline, and is independently runnable — notebook 04 is the only one with a dependency (it loads the trained weights saved by notebooks 02 and 03).
 
## Architectures
 
**MLP** — flattens 32×32×3 images into 3072-d vectors. Baseline: 3 hidden layers (1024 → 512 → 256), ReLU activations, Dropout (0.3), trained with cross-entropy loss.
 
**CNN** — 3 convolutional blocks (Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool(stride 2) → Dropout2d), channel progression 32 → 64 → 128, followed by adaptive average pooling and a fully-connected classifier head.
 
Both models trained with Adam (lr=1e-3, weight decay 1e-4) and cosine annealing, for 20 epochs, on a 90/10 train/validation split of the CIFAR-10 training set.
 
## Setup & Usage
 
```bash
pip install -r requirements.txt
```
 
Run the notebooks in order (01 → 02 → 03 → 04) — CIFAR-10 downloads automatically via `torchvision` on first run. A GPU is strongly recommended for notebooks 02 and 03 (CPU training is feasible but considerably slower, especially for the CNN).
 
## Notes
 
- All pixel-shift and pixel-shuffle experiments use **fixed, deterministic transformations** (not random augmentation), so that accuracy at a specific shift amount or under a specific permutation is precisely reproducible.
- The pooling-ablation experiment (CNN without `MaxPool2d` layers) was scoped out of this version to keep training time manageable, but the architecture in `src/models.py` already supports it via `CNN(use_pooling=False)` for anyone who wants to extend this comparison.