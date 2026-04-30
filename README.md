# Colon Cancer Detection Using Deep Learning


## Overview

This project presents a deep learning framework for automated colon cancer analysis from histopathological images. It addresses three tasks in a single unified pipeline built on a shared **EfficientNetV2-S** backbone pretrained on ImageNet:

| Objective | Task | Dataset(s) | Result |
|-----------|------|-----------|--------|
| 1 | Binary Classification (Normal vs Tumour) | LC25000 + NCT-CRC-HE-100K + CRC-VAL-HE-7K | **99.72% accuracy, ROC-AUC 0.9999** |
| 2 | Nine-Class Tissue Recognition | NCT-CRC-HE-100K + CRC-VAL-HE-7K | **99.67% accuracy, F1 0.9967** |
| 3 | Four-Class Severity Grading | CRC-HGD-V1 | **89% accuracy, F1 0.89** |

---

---

## Environment Setup

**Python 3.9+** is recommended.

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn pillow tqdm
```

Or with conda:

```bash
conda create -n colon_cancer python=3.9
conda activate colon_cancer
conda install pytorch torchvision -c pytorch
pip install scikit-learn matplotlib seaborn pillow tqdm
```

---

## Datasets

Download the following datasets and place them in a `data/` folder:

| Dataset | Classes | Size | Link |
|---------|---------|------|------|
| LC25000 | 5 (colon + lung) | 25,000 images | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) |
| NCT-CRC-HE-100K | 9 tissue types | 100,000 images | [Zenodo](https://zenodo.org/record/1214456) |
| CRC-VAL-HE-7K | 9 tissue types | 7,180 images | [Zenodo](https://zenodo.org/record/1214456) |
| CRC-HGD-V1 | 4 severity grades | ~1,000 images | [Mendeley Data](https://data.mendeley.com) |

---

## Training

### Objective 1 — Binary Classification

```bash
python objective1_binary.py --data_dir data/ --epochs 30 --batch_size 32
```

### Objective 2 — Nine-Class Tissue Recognition

```bash
python objective2_multiclass.py --data_dir data/ --epochs 30 --batch_size 32
```

### Objective 3 — Four-Class Severity Grading

```bash
python objective3_grading.py --data_dir data/CRC-HGD-V1/ --epochs 50 --batch_size 16
```

Trained model weights are saved automatically as `.pth` files in the working directory.

---

## Evaluation / Inference

To evaluate using the pretrained weights:

```bash
python objective1_binary.py --eval --weights best_model_binary.pth
python objective2_multiclass.py --eval --weights best_multiclass_model.pth
python objective3_grading.py --eval --weights best_model_severity.pth
```

To generate confusion matrices and performance plots:

```bash
python visualise.py
```

---

## Model Architecture

All three objectives share the same backbone:

- **Backbone:** EfficientNetV2-S pretrained on ImageNet (1,280-dimensional feature vector)
- **Classification head:** Fully connected layers → task-specific output
- **Loss:** Cross-entropy (with class weights for severity grading to handle imbalance)
- **Optimizer:** AdamW with decoupled weight decay
- **Augmentation:** Random horizontal/vertical flip, rotation, colour jitter, normalization

---

## Results Summary

### Binary Classification (combined test set, N = 17,577)
| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|----|---------|
| LC25000 | 100% | 1.00 | 1.00 | 1.00 | 1.0000 |
| NCT-CRC-HE-100K | 99.80% | — | — | — | — |
| CRC-VAL-HE-7K | 99.20% | — | — | — | — |
| **Combined** | **99.72%** | — | — | — | **0.9999** |

### Nine-Class Tissue Classification (combined test set, N = 16,077)
- Overall Accuracy: **99.67%**
- Weighted F1-score: **0.9967**

### Severity Grading (CRC-HGD-V1 test set, N = 286)
- Overall Accuracy: **89%**
- Weighted F1-score: **0.89**

---

## Severity Grading Classes

| Label | Description |
|-------|-------------|
| Grade 1 | Well-Differentiated (low aggression) |
| Grade 2 | Moderately-Differentiated |
| Grade 3 | Poorly-Differentiated (high aggression) |
| Normal | Normal colon tissue |

---

## License

This project is submitted as an academic B.Tech final year project. Code is shared for reproducibility and educational purposes.

Below is Hugging face deployment of Colon Cancer Detection 
  #https://nsree2405-colon-cancer-detection.hf.space
