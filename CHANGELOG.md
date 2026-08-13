# Changelog — plant-disease-vision-transformers

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [SemVer](https://semver.org/).

---

## [Unreleased]

### Planned
- Weights & Biases (W&B) integration for experiment tracking
- Custom dataset support via YAML config
- FastAPI REST endpoint for remote inference
- GitHub Pages with interactive results (Plotly)

---

## [1.0.0] — 2026-08-13

### Added
- Benchmark of 8 architectures on DatasetCOLEAF (9-class coffee leaf disease dataset)
- **CNN baselines:** VGG16, Inception_v3, ResNet200d
- **Vision Transformers:** ViT-B/16 (`google/vit-base-patch16-224`), ViT-B/16 (`nateraw/vit-base-patch16-224-cifar10`), Swin-B (`swin_base_patch4_window7_224`), BEiT (`microsoft/beit-base-patch16-224-pt22k-ft22k`)
- **State Space Model:** MambaVision-L (`nvidia/MambaVision-L-21K`)
- Preprocessing pipeline: Otsu thresholding + regionprops feature extraction
- Training config: AdamW, ReduceLROnPlateau, AMP GradScaler, CrossEntropyLoss
- Evaluation: per-class F1-Score, normalized confusion matrix, loss/accuracy curves
- GitHub Actions CI: flake8 lint + black format check

### Technical Details
- **Framework:** PyTorch 2.x + Hugging Face Transformers + `timm`
- **Dataset:** DatasetCOLEAF — 9 classes of coffee leaf diseases
- **Hardware:** NVIDIA GPU with AMP mixed precision
- **Metrics:** Validation Accuracy, F1-Score (macro), per-class precision/recall
