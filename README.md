# plant-disease-vision-transformers

[![CI](https://github.com/FernandoMcv/plant-disease-vision-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/FernandoMcv/plant-disease-vision-transformers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co)

Comparative benchmark of Vision Transformers, Mamba-SSM, and CNN baselines for plant disease classification. Fine-tuned on DatasetCOLEAF (9-class coffee leaf disease dataset) using PyTorch and Hugging Face.

---

## Results

| Architecture | Model | Val. Accuracy | Val. F1-Score (macro) | Parameters |
|:---|:---|:---:|:---:|:---:|
| Vision Transformer | ViT-B/16 `google/vit-base-patch16-224` | — | — | 86M |
| Vision Transformer | BEiT `microsoft/beit-base-patch16-224` | — | — | 86M |
| State Space Model | Mamba-SSM | — | — | — |
| CNN | ResNet-50 | — | — | 25M |
| CNN | VGG16 | — | — | 138M |

> Results pending. Will be updated after training. Confusion matrices and learning curves available in `/results/`.

---

## Architecture

```
Dataset
──────────────────────────────────────────────────────────────
  DatasetCOLEAF · 9 classes · RGB images resized to 224×224

Preprocessing Pipeline
──────────────────────────────────────────────────────────────
  Raw image
      │
      ├─ Otsu thresholding          (unsupervised segmentation)
      │       │
      │       └─ regionprops        (region feature extraction)
      │
      └─ Tensor normalization       (mean/std per channel)

Model Training
──────────────────────────────────────────────────────────────
  ┌──────────────────────────────────────────────────────────┐
  │                     PyTorch Training Loop                │
  │                                                          │
  │  ┌─────────────────┐  ┌─────────┐  ┌───────────────┐   │
  │  │   ViT-B/16      │  │  BEiT   │  │  Mamba-SSM    │   │
  │  │  (HuggingFace)  │  │  (HF)   │  │  (SSM-based)  │   │
  │  │  Progressive    │  │  Full   │  │  Alternative  │   │
  │  │  unfreezing     │  │  fine-  │  │  architecture │   │
  │  │  (30% deep      │  │  tuning │  │               │   │
  │  │   layers)       │  │         │  │               │   │
  │  └─────────────────┘  └─────────┘  └───────────────┘   │
  │                                                          │
  │  ┌─────────────────┐  ┌──────────────────────────────┐  │
  │  │   ResNet-50     │  │          VGG16               │  │
  │  │   (baseline)    │  │         (baseline)           │  │
  │  └─────────────────┘  └──────────────────────────────┘  │
  │                                                          │
  │  Optimizer  : AdamW (weight decay = 1e-4)               │
  │  Scheduler  : ReduceLROnPlateau (patience = 3)          │
  │  Precision  : Mixed precision — AMP GradScaler          │
  │  Loss       : CrossEntropyLoss                          │
  └──────────────────────────────────────────────────────────┘

Evaluation
──────────────────────────────────────────────────────────────
  F1-Score (macro + per-class)
  Confusion matrix (normalized)
  Loss and accuracy curves (train vs. validation)
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
├── results/
│   ├── confusion_matrix_vit.png
│   ├── confusion_matrix_beit.png
│   └── learning_curves.png
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
| Vision Models | ViT-B/16, BEiT (`microsoft/beit-base-patch16-224`) |
| State Space Models | Mamba-SSM |
| CNN Baselines | ResNet-50, VGG16 (via `timm`) |
| Preprocessing | OpenCV, scikit-image (`threshold_otsu`, `regionprops`) |
| Optimization | AdamW, ReduceLROnPlateau, AMP GradScaler |
| Evaluation | scikit-learn (F1-Score, confusion matrix) |
| Visualization | Matplotlib, Seaborn |
| CI | GitHub Actions (flake8, black) |

---

## Dataset

**DatasetCOLEAF** — 9-class coffee leaf disease dataset.
Not included in this repository. See the notebook for data loading and preparation steps.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Fernando Castro · [LinkedIn](https://linkedin.com/in/FernandoMcv) · [GitHub](https://github.com/FernandoMcv)
