# plant-disease-vision-transformers

[![CI](https://github.com/FernandoMcv/plant-disease-vision-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/FernandoMcv/plant-disease-vision-transformers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co)

Comparative benchmark of 8 deep learning architectures — CNN baselines, Vision Transformers, and State Space Models — for multi-class plant disease classification. Fine-tuned on DatasetCOLEAF (9-class coffee leaf disease dataset) using PyTorch and Hugging Face.

---

## Architecture

```
Dataset
────────────────────────────────────────────────────────────────
  DatasetCOLEAF · 9 classes · RGB images resized to 224x224

Preprocessing Pipeline
────────────────────────────────────────────────────────────────
  Raw image
      |
      +-- Otsu thresholding          (unsupervised segmentation)
      |       |
      |       +-- regionprops        (region feature extraction)
      |
      +-- Tensor normalization       (mean/std per channel)

Benchmark Models (PyTorch)
────────────────────────────────────────────────────────────────
  CNN Baselines
  +---------------+  +---------------+  +---------------+
  |    VGG16      |  | Inception_v3  |  |  ResNet200d   |
  +---------------+  +---------------+  +---------------+

  Vision Transformers
  +--------------------+  +--------------------+
  | ViT-B/16           |  | ViT-B/16           |
  | google/vit-base    |  | nateraw/vit-base   |
  | patch16-224        |  | patch16-224-cifar10|
  +--------------------+  +--------------------+

  +--------------------+  +--------------------+
  | Swin-B             |  | BEiT               |
  | swin_base_patch4   |  | microsoft/beit-    |
  | _window7_224       |  | base-pt22k-ft22k   |
  +--------------------+  +--------------------+

  State Space Model
  +--------------------+
  | MambaVision-L      |
  | nvidia/MambaVision |
  | -L-21K             |
  +--------------------+

  Training Config (all models)
  Optimizer : AdamW
  Scheduler : ReduceLROnPlateau
  Precision : Mixed precision — AMP GradScaler
  Loss      : CrossEntropyLoss

Evaluation
────────────────────────────────────────────────────────────────
  Accuracy · F1-Score (macro + per-class) · Confusion matrix · Loss curves
```

---

## Installation

```bash
git clone https://github.com/FernandoMcv/plant-disease-vision-transformers.git
cd plant-disease-vision-transformers

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook notebooks/vision_transformers_finetuning.ipynb
```

---

## Project Structure

```
plant-disease-vision-transformers/
├── .github/
│   └── workflows/
│       └── ci.yml              GitHub Actions — lint and format check
├── notebooks/
│   └── vision_transformers_finetuning.ipynb
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Stack

| Category | Details |
|:---|:---|
| Deep Learning | PyTorch 2.1+, Hugging Face Transformers, `timm` |
| CNN Baselines | VGG16, Inception_v3, ResNet200d |
| Vision Transformers | ViT-B/16 (`google`, `nateraw`), Swin-B, BEiT (`microsoft`) |
| State Space Models | MambaVision-L (`nvidia/MambaVision-L-21K`) |
| Preprocessing | OpenCV, scikit-image (`threshold_otsu`, `regionprops`) |
| Optimization | AdamW, ReduceLROnPlateau, AMP GradScaler |
| Evaluation | scikit-learn (F1-Score, confusion matrix) |
| Visualization | Matplotlib, Seaborn |
| CI | GitHub Actions (flake8, black) |

---

## Results

| Architecture | Model | Val. Accuracy | Val. F1-Score (macro) | Parameters |
|:---|:---|:---:|:---:|:---:|
| CNN | VGG16 | — | — | 138M |
| CNN | Inception_v3 | — | — | 27M |
| CNN | ResNet200d | — | — | 65M |
| Vision Transformer | ViT-B/16 `google/vit-base-patch16-224` | — | — | 86M |
| Vision Transformer | ViT-B/16 `nateraw/vit-base-patch16-224-cifar10` | — | — | 86M |
| Vision Transformer | Swin-B `swin_base_patch4_window7_224` | — | — | 88M |
| Vision Transformer | BEiT `microsoft/beit-base-patch16-224-pt22k-ft22k` | — | — | 86M |
| State Space Model | MambaVision-L `nvidia/MambaVision-L-21K` | — | — | — |

> Training in progress on GCP (Google Colab + GPU). Results and plots will be updated after each run.
> Full outputs — confusion matrices, learning curves, and per-class metrics — are visible directly in the [notebook on GitHub](notebooks/vision_transformers_finetuning.ipynb).

---

## 📱 Mobile Deployment

The best performing models from this benchmark were quantized using **PyTorch Mobile Lite** (`.ptl`) and deployed to an Edge AI Android application for real-time inference without cloud dependency.

> **View the companion repository:** [cafe-diagnostic-tool-android](https://github.com/FernandoMcv/cafe-diagnostic-tool-android)

---

## Dataset

**DatasetCOLEAF** — 9-class coffee leaf disease dataset.
Not included in this repository. See the notebook for data loading and preparation steps.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Fernando Castro · [LinkedIn](https://www.linkedin.com/in/fernandocastrov/) · [GitHub](https://github.com/FernandoMcv)
