# Distributed Multi-View Vision-Only RSSI Estimation

Official implementation of **"Distributed Multi-View Vision-Only RSSI Estimation"**

> Jung-Beom Kim and Woongsup Lee  
> Graduate School of Information, Yonsei University

For details on the proposed framework, model architecture, training procedure, and experimental results, please refer to the paper.

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

| Model | Class | Config | Params (M) | Pre-trained Weights |
|---|---|---|---|---|
| SinViT-D | `SinViTD` | D=96, L=12, H=3 | 1.46 | `sinvit_d_places365_320x240_best.pth` |
| SinViT-W | `SinViTW` | D=192, L=6, H=3 | 2.90 | `sinvit_w_places365_320x240_best.pth` |
| MulViT-TF | `MulViTTF` | D=96, L=6, H=3 ─ ×2 encoders | 1.72 | `mulvit_backbone_places365_320x240_best.pth` |
| MulViT-TWDNN | `MulViTTWDNN` | D=96, L=6, H=3 ─ ×2 encoders | 1.87 | `mulvit_backbone_places365_320x240_best.pth` |

---

## Usage

1. Set `PRETRAINED_PATH` and `SAVE_DIR` in the Hyperparameters cell
2. Prepare your dataloader (`train_loader`, `val_loader`) returning `(image, rssi)` pairs for SinViT or `(image1, image2, rssi)` for MulViT
3. Run the notebook

---
## Citation

If you used this system and it is relevant to your research, 
please consider citing:
@misc{kim2026distributedmultiviewvisiononlyrssi,
      title={Distributed Multi-View Vision-Only RSSI Estimation}, 
      author={Jung-Beom Kim and Woongsup Lee},
      year={2026},
      eprint={2604.26738},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2604.26738}, 
}
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
