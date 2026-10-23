# CropCare AI — PlantVillage Classification

> MobileNetV3-Large fine-tuned on the 38 verified, populated local PlantVillage classes · RTX 3050 GPU · Flask web app

---

## Project Overview

CropCare AI is a college mini-project that demonstrates end-to-end transfer learning:
- Torchvision MobileNetV3-Large initialized with **ImageNet pretrained weights** and fine-tuned on PlantVillage
- Trained on the [PlantVillage dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5032846/) (Hughes & Salehe 2015)
- Flask web application for image upload, prediction, history, and documentation
- SQLite prediction history
- GPU-accelerated training with CUDA 12.8 (RTX 3050)

**This is not a wrapper around a pretrained model.** The CNN learns plant disease features from scratch.

---

## Verified Dataset

The authorized local dataset contains 38 populated class directories and 54,305 readable images (census: 2026-10-23). The class mapping is the sorted, verified list in `config/settings.py`; `reports/class_mapping.json` and `data/processed/class_mapping.json` record its indices and image counts after audit/splitting.

The observed classes include Apple (4), Blueberry (1), Cherry (2), Corn (4), Grape (4), Orange (1), Peach (2), Bell Pepper (2), Potato (3), Raspberry (1), Soybean (1), Squash (1), Strawberry (2), and Tomato (10). No empty directory is admitted. The current image-count ratio is 36.23:1 (5,507 maximum, 152 minimum).

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.14 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA RTX 3050 6GB Laptop |
| Flask | 3.1.x |
| Pillow | 12.x |

---

## Project Structure

```
CropCareAI/
├── app.py                      # Flask web application
├── config/
│   └── settings.py             # All configuration (paths, hyperparams, disease info)
├── src/
│   ├── models.py               # 3 custom CNN architectures (S/M/L)
│   ├── dataset.py              # DataLoader, augmentations, exact/perceptual grouped splits
│   ├── train.py                # Training loop (GPU, AMP, OneCycleLR, early stopping)
│   ├── evaluate.py             # Test-set evaluation + reports
│   ├── inference.py            # Prediction engine (singleton model loader)
│   └── database.py             # SQLite prediction history
├── scripts/
│   ├── download_dataset.py     # Download PlantVillage dataset
│   ├── audit_dataset.py        # Dataset audit: verify all populated classes
│   └── compare_architectures.py# Quick 5-epoch comparison of all 3 CNNs
├── tests/
│   └── test_cropcare.py        # Full test suite (37 tests)
├── data/
│   └── raw/PlantVillage/       # Dataset goes here (auto-downloaded)
├── models/                     # Saved checkpoints (.pth)
├── reports/                    # Audit, training and evaluation JSON reports
├── static/
│   ├── css/style.css           # Dark-mode responsive UI
│   └── uploads/                # Uploaded leaf images
├── templates/                  # Jinja2 HTML templates
└── requirements.txt
```

---

## Setup & Usage

### 1. Install dependencies

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flask pillow pytest
```

### 2. Download and audit dataset

```powershell
# Download PlantVillage (~1 GB from GitHub mirror)
python scripts/download_dataset.py

# Verify the local class mapping and usable image counts
python scripts/audit_dataset.py

# Generate reproducible group-aware manifests; checks exact and perceptual duplicates
python scripts/create_manifests.py
```

The current train/validation/test manifests and ordered 38-class mapping are already generated. The commands above are for explicit dataset audits or split regeneration; training and evaluation use the existing manifests without creating new splits.

### 3. Train

```powershell
# MobileNetV3-Large smoke test (diagnostic only; 2 batches per split)
python src/train_mobilenet.py --smoke-test

# Full training: ImageNet initialization, 224px, 8 images/batch by default
python src/train_mobilenet.py --epochs 20 --head-epochs 2 --batch-size 8

# Resume interrupted MobileNet training
python src/train_mobilenet.py --epochs 20 --head-epochs 2 --batch-size 8 --resume models/mobilenet_v3/last_MobileNetV3_Large.pth
```

### 4. Evaluate on test set

```powershell
python src/evaluate.py --checkpoint models/mobilenet_v3/best_MobileNetV3_Large.pth
```

### 5. Run the web app

```powershell
python app.py
# Open: http://localhost:5000
```

### 6. Run tests

```powershell
python -m pytest tests/test_cropcare.py -v
```

---

## Model Architectures

The primary model is torchvision MobileNetV3-Large. Its classifier is replaced with a 38-output layer. The ImageNet pretrained model is fine-tuned in two phases: classifier head first, then the full backbone at a lower learning rate. Custom CropCareNet implementations and their previous checkpoints remain available but are no longer the primary pipeline.

| Model | Description |
|---|---|
| `MobileNetV3_Large` | Torchvision MobileNetV3-Large with ImageNet initialization and a replacement 38-class classifier |
| `CropCareNet_S/M/L` | Legacy custom CNNs retained for reproducibility |

---

## Training Configuration

- **Optimizer**: AdamW; head phase 1e-3, then backbone 1e-4 and classifier 5e-4 (weight_decay=1e-4)
- **Scheduler**: cosine decay within each phase
- **Early stopping**: patience=7 on validation macro-F1
- **Mixed precision**: FP16 AMP on CUDA; default batch size 8 at 224px
- **Preprocessing**: train RandomResizedCrop(224), horizontal flip; validation/test resize to 256 and center crop to 224; ImageNet normalization
- **Loss**: CrossEntropyLoss with label smoothing and normalized square-root inverse-frequency weights calculated from the training split only
- **Splits**: 70% train / 15% val / 15% test (seed=42; exact duplicates and dHash near-duplicates kept together)
- **Selection**: Best checkpoint selected by validation macro-F1; evaluate the untouched test split only after selection
- **Checkpoint safety**: MobileNet checkpoints go to `models/mobilenet_v3/`; prior checkpoints remain untouched

---

## Important Limitations

⚠️ **Read before using:**

1. Trained on **PlantVillage lab-controlled images** — uniform backgrounds. Real-world field photos may significantly reduce accuracy.
2. **Limited to the 38 verified PlantVillage labels**. Other crops and diseases are outside the model's label space and will be assigned a known class.
3. **ImageNet pretraining is used** — MobileNetV3-Large is fine-tuned on PlantVillage; this is transfer learning, not training from scratch.
4. **Do not use for actual crop management** without consulting a qualified agronomist.

---

## Dataset Reference

Hughes, D., & Salehe, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
[https://pmc.ncbi.nlm.nih.gov/articles/PMC5032846/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5032846/)
