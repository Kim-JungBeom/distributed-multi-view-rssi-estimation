# Distributed Multi-View Vision-Only RSSI Estimation

Official implementation of **"Distributed Multi-View Vision-Only RSSI Estimation"**

For details on the proposed framework, model architecture, training procedure, and experimental results, please refer to the paper.

---

## ⚠️ Note on Updated Results

**The results in this repository differ from those reported in the arXiv preprint.**

The current results are obtained under a **time-division (temporally segmented) train/validation/test split**. For each scene, the samples are divided into consecutive 10 s segments, within each of which the first 8 s, the next 1 s, and the last 1 s form the training, validation, and test sets, respectively. Since images are captured at 20 Hz, temporally adjacent frames are nearly identical; a random split would therefore place near-duplicate frames on both sides of the split. The time-division split prevents this while still allowing all three sets to span the entire user trajectory.

| Metric (vs. best single-view baseline) | arXiv preprint | This version |
|---|---|---|
| Scene 1 — RMSE reduction | 26.3% | **22.8%** |
| Scene 1 — ±3 dB accuracy gain | +13.8 pp (70.7% → 84.5%) | **+10.8 pp** (71.7% → 82.5%) |
| Scene 2 — RMSE reduction | 16.6% | **18.5%** |
| Scene 2 — ±3 dB accuracy gain | +7.5 pp (70.9% → 78.4%) | **+7.4 pp** (71.2% → 78.6%) |

All numbers reported below follow the current (time-division) protocol.

---

## Key Results

Compared to the best-performing single-view baseline **for each metric**, MulViT-TF achieves:

| Scene | RMSE | MAE | R² | ±3 dB Accuracy |
|---|---|---|---|---|
| Scene 1 | **22.8%** ↓ | **24.7%** ↓ | **+0.15** | 71.7% → **82.5%** (+10.8 pp) |
| Scene 2 | **18.5%** ↓ | **19.3%** ↓ | **+0.15** | 71.2% → **78.6%** (+7.4 pp) |

while requiring **16% fewer FLOPs** and **41% fewer parameters** than SinViT-W.

### Full Comparison

| Configuration | Model | FLOPs (G) | Params (M) | S1 RMSE (dB) | S1 MAE (dB) | S1 R² | S1 ±3 dB (%) | S2 RMSE (dB) | S2 MAE (dB) | S2 R² | S2 ±3 dB (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single-View | SinViT-D (Cam1) | 1.26 | 1.46 | 3.154 | 2.451 | 0.604 | 66.7 | 3.289 | 2.476 | 0.507 | 68.9 |
| Single-View | SinViT-D (Cam2) | 1.26 | 1.46 | 3.879 | 2.983 | 0.401 | 61.0 | 3.485 | 2.623 | 0.446 | 64.4 |
| Single-View | SinViT-W (Cam1) | 2.10 | 2.90 | 3.019 | 2.317 | 0.637 | 71.7 | 3.246 | 2.378 | 0.520 | 71.2 |
| Single-View | SinViT-W (Cam2) | 2.10 | 2.90 | 3.569 | 2.700 | 0.493 | 66.3 | 3.128 | 2.418 | 0.554 | 67.9 |
| Multi-View | **MulViT-TF** | 1.76 | 1.72 | **2.330** | **1.745** | **0.784** | **82.5** | **2.550** | **1.919** | **0.704** | **78.6** |
| Multi-View | MulViT-TWDNN | 1.66 | 1.87 | 2.602 | 1.979 | 0.730 | 77.4 | 2.879 | 2.163 | 0.622 | 73.1 |

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
    ├── MulViT-TF.ipynb        # Proposed: Dual ViT + Transformer Fusion
    └── MulViT-TWDNN.ipynb     # Baseline: Dual ViT + Token-wise DNN Fusion
```

---

## Models

| Model | Class | Config | Fusion | FLOPs (G) | Params (M) | Pre-trained Weights |
|---|---|---|---|---|---|---|
| SinViT-D | `SinViTD` | D=96, L=12, H=3 | — (single CLS token) | 1.26 | 1.46 | `sinvit_d_places365_320x240_best.pth` |
| SinViT-W | `SinViTW` | D=192, L=6, H=3 | — (single CLS token) | 2.10 | 2.90 | `sinvit_w_places365_320x240_best.pth` |
| MulViT-TF | `MulViTTF` | D=96, L=6, H=3 ─ ×2 encoders | L'=2 fusion Transformer blocks (D=96, H=3) | 1.76 | 1.72 | `mulvit_backbone_places365_320x240_best.pth` |
| MulViT-TWDNN | `MulViTTWDNN` | D=96, L=6, H=3 ─ ×2 encoders | 4 token-wise residual blocks (no cross-view interaction) | 1.66 | 1.87 | `mulvit_backbone_places365_320x240_best.pth` |

The MLP head, shared across all models, consists of a single hidden layer of 128 units with GELU activation.

The two camera encoders share the same architecture and pre-trained initialization but **do not share parameters** during fine-tuning, allowing each encoder to specialize to its assigned viewpoint. Before fusion, a **learnable segment embedding** is added to the concatenated token sequence to distinguish tokens by their originating camera. The RSSI estimate is produced from the concatenation of the fused CLS tokens.

---

## Pre-training

The provided backbone weights in `Foundation Model/` are pre-trained on the **Places365** dataset using an image classification task following the **DeiT training recipe**. Places365 is a scene-centric dataset that requires holistic understanding of the entire environment, which aligns well with the indoor RSSI estimation task where the overall spatial configuration governs signal propagation. The DeiT recipe is adopted for its data-efficient training strategy, enabling effective ViT optimization without large-scale data.

---

## Fine-Tuning Procedure

The pre-trained backbone is fine-tuned on the target RSSI estimation task in **two phases**:

- **Phase 1 (40 epochs)**: Backbone frozen. Only the CLS tokens, positional embeddings, segment embeddings, fusion module, and MLP head are trained.
- **Phase 2 (60 epochs)**: End-to-end joint optimization, with a reduced learning rate applied to the backbone to preserve pre-trained representations.

In both phases, the model is trained by minimizing the **MSE loss**.

### Input Specification

- **Image resolution**: `320 × 240` (preserves the full FoV)
- **Patch size**: `16 × 16` → **300 patch tokens** per image
- **RSSI labels**: z-score standardization computed from the training set

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

---

## Usage

1. Set `PRETRAINED_PATH` and `SAVE_DIR` in the Hyperparameters cell
2. Prepare your dataloader (`train_loader`, `val_loader`) returning `(image, rssi)` pairs for SinViT or `(image1, image2, rssi)` for MulViT
   - Images must be resized to `320 × 240`
   - RSSI labels should be z-score normalized using statistics from the training set
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
