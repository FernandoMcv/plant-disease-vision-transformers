# Changelog — plant-disease-vision-transformers

Todos los cambios notables en este proyecto están documentados en este archivo.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).  
Versionado según [SemVer](https://semver.org/lang/es/).

---

## [Unreleased]

### Planeado
- Integración con Weights & Biases (W&B) para tracking de experimentos
- Soporte para dataset personalizado vía configuración YAML
- API REST con FastAPI para inferencia remota
- GitHub Pages con resultados interactivos (Plotly)

---

## [1.0.0] — 2026-08-12

### Added
- Fine-tuning de **ViT-B/16** (`google/vit-base-patch16-224`) con descongelamiento progresivo del 30% de capas profundas
- Fine-tuning de **BEiT** (`microsoft/beit-base-patch16-224`) sobre DatasetCOLEAF (9 clases)
- Fine-tuning de arquitecturas CNN baseline: **ResNet-50** y **VGG16** para comparación SOTA
- Experimentos con **Mamba-SSM** como arquitectura alternativa basada en State Space Models
- Pipeline de preprocesamiento no supervisado con **segmentación Otsu** y extracción de regiones (`regionprops`)
- Optimización con `AdamW`, scheduler `ReduceLROnPlateau` y Mixed Precision Training (`AMP GradScaler`)
- Evaluación multiclase: F1-Score por clase, Confusion Matrix normalizada, Curvas de Loss/Accuracy
- Exportación de resultados a `.csv` y visualizaciones a `.png`

### Technical Details
- **Framework:** PyTorch 2.x + Hugging Face Transformers + `timm`
- **Dataset:** DatasetCOLEAF — 9 clases de enfermedades en hojas de café
- **Hardware:** GPU NVIDIA (entrenamiento con AMP para optimización de memoria)
- **Métricas:** Validation F1-Score (macro), Validation Loss, per-class precision/recall
