# Distributed Multi-View Vision-Only RSSI Estimation

Official implementation of **"Distributed Multi-View Vision-Only RSSI Estimation"**

For details on the proposed framework, model architecture, training procedure, and experimental results, please refer to the paper.

---

## ⚠️ Note on Updated Results

**The results in this repository differ from those reported in the arXiv preprint and in earlier versions of this repository.** The current version reflects the following changes:

- **Shared encoder**: MulViT-TF and MulViT-TWDNN use a single ViT encoder shared across cameras, instead of camera-specific encoders.
- **Single model across scenes and cameras**: every model is trained once on the pooled data of both scenes (and, for the single-view baselines, on the images of both cameras), without any scene-specific component.
- **Validation-guarded time-division split**: for each recording, the samples are divided into consecutive 10 s segments. Within each segment, the first 8 s form the training set, and the remaining 2 s are split into a 1 s test block flanked by two 0.5 s validation blocks (8:1:1). Every test frame is thus separated from the nearest training frame by at least 0.5 s of validation data, while all three sets span the entire user trajectory.
- **Additional baselines**: fusion models built on the ResNet-18 encoder of prior work (ResNet18-TF) and on DeiT-Tiny (DeiT-Tiny-TF).

| Metric (vs. best single-view baseline) | arXiv preprint | This version |
|---|---|---|
| Scene 1 — RMSE reduction | 26.3% | **21.6%** |
| Scene 1 — ±3 dB accuracy gain | +13.8 pp (70.7% → 84.5%) | **+10.7 pp** (64.6% → 75.3%) |
| Scene 2 — RMSE reduction | 16.6% | **14.3%** |
| Scene 2 — ±3 dB accuracy gain | +7.5 pp (70.9% → 78.4%) | **+7.9 pp** (64.7% → 72.6%) |

All numbers reported below follow the current protocol.

---

## Key Results

Compared to the best-performing single-view baseline **for each metric**, MulViT-TF achieves:

| Scene | RMSE | MAE | R² | ±3 dB Accuracy |
|---|---|---|---|---|
| Scene 1 | **21.6%** ↓ | **23.3%** ↓ | **+0.18** | 64.6% → **75.3%** (+10.7 pp) |
| Scene 2 | **14.3%** ↓ | **16.3%** ↓ | **+0.14** | 64.7% → **72.6%** (+7.9 pp) |

while requiring **16% fewer FLOPs** and **67% fewer parameters** than SinViT-W.

Replacing the token-wise fusion blocks with the fusion Transformer reduces the RMSE by **11.2%** (Scene 1) and **7.7%** (Scene 2), isolating the contribution of cross-view attention.

### Full Comparison

| Configuration | Model | FLOPs (G) | Params (M) | S1 RMSE (dB) | S1 MAE (dB) | S1 R² | S1 ±3 dB (%) | S2 RMSE (dB) | S2 MAE (dB) | S2 R² | S2 ±3 dB (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single-View | SinViT-D (Cam1) | 1.26 | 1.46 | 3.637 | 2.933 | 0.493 | 57.8 | 3.731 | 2.791 | 0.380 | 64.0 |
| Single-View | SinViT-D (Cam2) | 1.26 | 1.46 | 3.892 | 3.050 | 0.419 | 59.3 | 3.696 | 2.865 | 0.392 | 61.8 |
| Single-View | SinViT-W (Cam1) | 2.10 | 2.90 | 3.439 | 2.712 | 0.547 | 64.6 | 3.658 | 2.758 | 0.404 | 64.7 |
| Single-View | SinViT-W (Cam2) | 2.10 | 2.90 | 3.687 | 2.831 | 0.479 | 63.2 | 3.420 | 2.652 | 0.479 | 63.9 |
| Multi-View (Backbone) | ResNet18-TF† | 7.41 | 11.93 | 2.862 | 2.224 | 0.686 | 70.7 | **2.811** | **2.071** | **0.648** | **77.9** |
| Multi-View (Backbone) | DeiT-Tiny-TF | 5.72 | 6.17 | **2.300** | **1.793** | **0.797** | **80.7** | 2.938 | 2.255 | 0.616 | 73.6 |
| Multi-View | **MulViT-TF** | 1.76 | 0.95 | **2.695** | **2.079** | **0.722** | **75.3** | **2.931** | **2.221** | **0.617** | **72.6** |
| Multi-View | MulViT-TWDNN | 1.66 | 1.10 | 3.034 | 2.297 | 0.647 | 70.4 | 3.174 | 2.386 | 0.551 | 68.6 |

Bold denotes the best result within each multi-view group. † Camera encoder adopted from Zhang *et al.*, "Vision aided channel prediction for vehicular communications: A case study of received power prediction using RGB images," *IEEE Trans. Veh. Technol.*, 2025.

MulViT-TF requires 3.2× and 4.2× fewer FLOPs and 6.5× and 12.6× fewer parameters than DeiT-Tiny-TF and ResNet18-TF, respectively; it outperforms ResNet18-TF in Scene 1 and remains within 0.12 dB of both models in Scene 2.

FLOPs are counted with fvcore as 2 × MACs per inference (both cameras for multi-view models), including the attention operations.

---

## Hardware

For hardware setup and synchronized data collection, see [esp32-csi-raspi-cam-sync-collector](https://github.com/Kim-JungBeom/esp32-csi-raspi-cam-sync-collector).

---

## Repository Structure

```
.
├── Foundation Model/          # Pre-trained ViT weights (Places365)
│   ├── sinvit_d_places365_320x240_best.pth
│   ├── sinvit_w_places365_320x240_best.pth
│   └── mulvit_backbone_places365_320x240_best.pth
└── Fine-Tuning/               # Fine-tuning notebooks
    ├── SinViT-D.ipynb         # Single-view baseline (embed_dim=96, depth=12)
    ├── SinViT-W.ipynb         # Single-view baseline (embed_dim=192, depth=6)
    ├── MulViT-TF.ipynb        # Proposed: shared ViT + Transformer fusion
    ├── MulViT-TWDNN.ipynb     # Baseline: shared ViT + token-wise DNN fusion
    ├── ResNet18-TF.ipynb      # Baseline: shared ResNet-18 + Transformer fusion
    └── DeiT-Tiny-TF.ipynb     # Baseline: shared DeiT-Tiny + Transformer fusion
```

---

## Models

| Model | Encoder (shared across cameras) | Fusion | Input | FLOPs (G) | Params (M) | Pre-trained Weights |
|---|---|---|---|---|---|---|
| SinViT-D | ViT, D=96, L=12, H=3 | — (single CLS token) | 320 × 240 | 1.26 | 1.46 | `sinvit_d_places365_320x240_best.pth` |
| SinViT-W | ViT, D=192, L=6, H=3 | — (single CLS token) | 320 × 240 | 2.10 | 2.90 | `sinvit_w_places365_320x240_best.pth` |
| **MulViT-TF** | ViT, D=96, L=6, H=3 | 2 fusion Transformer blocks (D=96, H=3, MLP ratio 2) | 320 × 240 | 1.76 | 0.95 | `mulvit_backbone_places365_320x240_best.pth` |
| MulViT-TWDNN | ViT, D=96, L=6, H=3 | 4 token-wise residual blocks (no cross-view interaction) | 320 × 240 | 1.66 | 1.10 | `mulvit_backbone_places365_320x240_best.pth` |
| ResNet18-TF | ResNet-18 | 2 fusion Transformer blocks (D=192, H=3, MLP ratio 2) | 224 × 224 | 7.41 | 11.93 | ImageNet-1k (torchvision) |
| DeiT-Tiny-TF | DeiT-Tiny | 2 fusion Transformer blocks (D=192, H=3, MLP ratio 2) | 224 × 224 | 5.72 | 6.17 | ImageNet-1k (timm) |

The MLP head, shared across all models, consists of a single hidden layer of 128 units with GELU activation.

In the multi-view models, **a single encoder is shared across all cameras**, so that the number of encoder parameters does not grow with the number of cameras. Before fusion, a **learnable segment embedding** is added to the concatenated token sequence to distinguish tokens by their originating camera. The RSSI estimate is produced from the concatenation of the fused CLS tokens.

---

## Pre-training

The provided backbone weights in `Foundation Model/` are pre-trained on the **Places365** dataset using an image classification task following the **DeiT training recipe**. Places365 is a scene-centric dataset that requires holistic understanding of the entire environment, which aligns well with the indoor RSSI estimation task where the overall spatial configuration governs signal propagation. The DeiT recipe is adopted for its data-efficient training strategy, enabling effective ViT optimization without large-scale data.

---

## Fine-Tuning Procedure

The pre-trained backbone is fine-tuned on the target RSSI estimation task in **two phases**:

- **Phase 1 (40 epochs)**: Backbone frozen. Only the CLS token, positional embedding, segment embeddings, fusion module, and MLP head are trained.
- **Phase 2 (60 epochs)**: End-to-end joint optimization, with a reduced learning rate applied to the backbone to preserve pre-trained representations.

In both phases, the model is trained by minimizing the **MSE loss**. Each model is trained once on the pooled data of both scenes.

### Input Specification

- **Image resolution**: `320 × 240` (preserves the full FoV); `224 × 224` for ResNet18-TF and DeiT-Tiny-TF
- **Patch size**: `16 × 16` → **300 patch tokens** per image
- **RSSI labels**: z-score standardization computed from the training set pooled over both scenes

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Backbone LR scale (Phase 2) | 0.01 |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Batch size | 16 |
| Loss | MSE |
| Phase 1 epochs | 40 |
| Phase 2 epochs | 60 |
| Random seed | 1 (deterministic cuDNN settings) |

---

## Usage

1. Set `PRETRAINED_PATH` and `SAVE_DIR` in the Hyperparameters cell
2. Prepare your dataloader (`train_loader`, `val_loader`) returning `(image, rssi)` pairs for single-view models or `(image1, image2, rssi)` for multi-view models
   - Images must be resized to `320 × 240` (`224 × 224` for ResNet18-TF and DeiT-Tiny-TF)
   - RSSI labels should be z-score normalized using statistics from the training set
   - For single-view models, the images of both cameras are used as independent samples
3. Run the notebook (Phase 1 → Phase 2 fine-tuning)

---

## Citation

If you used this system and it is relevant to your research, 
please consider citing:
```bibtex
@misc{kim2026distributedmultiviewvisiononlyrssi,
      title={Distributed Multi-View Vision-Only RSSI Estimation}, 
      author={Jung-Beom Kim and Woongsup Lee},
      year={2026},
      eprint={2604.26738},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2604.26738}, 
}
```

## Contact

For any questions or inquiries, please contact:
**Jung-Beom Kim** ([kjung99@yonsei.ac.kr](mailto:kjung99@yonsei.ac.kr))

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
