# Distributed Multi-View Vision-Only RSSI Estimation

Official implementation of **"Distributed Multi-View Vision-Only RSSI Estimation"**

For details on the proposed framework, model architecture, training procedure, and experimental results, please refer to the paper.

---

## Key Results

Compared to the best-performing single-view baseline (SinViT-W), MulViT-TF achieves:

| Scene | RMSE Reduction | 3dB Error Coverage |
|---|---|---|
| Scene 1 | **26.3%** ↓ | 70.7% → **84.5%** (+13.8pp) |
| Scene 2 | **16.6%** ↓ | 70.9% → **78.4%** (+7.5pp) |

while using fewer FLOPs and parameters than SinViT-W.

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

| Model | Class | Config | FLOPs (G) | Params (M) | Pre-trained Weights |
|---|---|---|---|---|---|
| SinViT-D | `SinViTD` | D=96, L=12, H=3 | 1.26 | 1.46 | `sinvit_d_places365_320x240_best.pth` |
| SinViT-W | `SinViTW` | D=192, L=6, H=3 | 2.10 | 2.90 | `sinvit_w_places365_320x240_best.pth` |
| MulViT-TF | `MulViTTF` | D=96, L=6, H=3 ─ ×2 encoders | 1.76 | 1.72 | `mulvit_backbone_places365_320x240_best.pth` |
| MulViT-TWDNN | `MulViTTWDNN` | D=96, L=6, H=3 ─ ×2 encoders | 1.66 | 1.87 | `mulvit_backbone_places365_320x240_best.pth` |

The MLP head, shared across all models, consists of a single hidden layer of 128 units with GELU activation.

---

## Pre-training

The provided backbone weights in `Foundation Model/` are pre-trained on the **Places365** dataset using an image classification task following the **DeiT training recipe**. Places365 is a scene-centric dataset that requires holistic understanding of the entire environment, which aligns well with the indoor RSSI estimation task where the overall spatial configuration governs signal propagation. The DeiT recipe is adopted for its data-efficient training strategy, enabling effective ViT optimization without large-scale data.

---

## Fine-Tuning Procedure

The pre-trained backbone is fine-tuned on the target RSSI estimation task in **two phases**:

- **Phase 1 (80 epochs)**: Backbone frozen. Only the CLS token, positional embeddings, fusion module, and MLP head are trained.
- **Phase 2 (120 epochs)**: End-to-end joint optimization, with a reduced learning rate applied to the backbone to preserve pre-trained representations.

In both phases, a **cosine annealing** learning rate schedule is applied, and the model is trained by minimizing the **MSE loss**.

### Input Specification

- **Image resolution**: `320 × 240` (preserves the full FoV)
- **Patch size**: `16 × 16` → **300 patch tokens** per image
- **RSSI labels**: z-score standardization computed from the training set

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Backbone LR scale (Phase 2) | 0.1 |
| Weight decay | 0.1 |
| Dropout | 0.1 |
| Batch size | 32 |
| LR schedule | Cosine annealing |
| Loss | MSE |
| Phase 1 epochs | 80 |
| Phase 2 epochs | 120 |

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
