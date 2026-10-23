# Held-Out Test Set Final Evaluation Report

**Model Architecture**: `MobileNetV3-Large` (Best Checkpoint Epoch: 5)  
**Evaluation Timestamp**: 2026-10-10 20:42:44  
**Evaluation Hardware**: cuda:0  
**Total Held-Out Test Samples**: `8351`  

---

## 1. Global Performance Metrics

| Metric | Result | Benchmark Interpretation |
|---|:---:|---|
| **Top-1 Accuracy** | **99.71%** | Percentage of exact correct diagnoses |
| **Top-3 Accuracy** | **100.0%** | Correct disease within top 3 confidence rankings |
| **Macro-F1 Score** | **0.9961** | Unweighted mean F1 across all 39 classes |
| **Weighted-F1 Score** | **0.9971** | Support-weighted mean F1 score |
| **Micro-F1 Score** | **0.9971** | Global precision-recall balance |

---

## 2. Real-Time Inference Latency Benchmarks

- **Average Latency per Image**: `1.176 ms`
- **Batch Throughput (Batch Size 32)**: `37.64 ms` per batch (~850 images/sec)
- **Total Test Evaluation Duration**: `29.22 seconds`

---

## 3. Negative-Control (`Background_without_leaves`) Analysis

The negative control class verifies that the network does not hallucinate plant diseases when non-foliar or background imagery is supplied:

- **Precision**: `98.29%`
- **Recall**: `100.00%`
- **F1-Score**: `0.9914`
- **Held-Out Test Support**: `172.0` samples

---

## 4. Per-Class Diagnostic Performance (39 Classes)

| No. | Class Name | Precision | Recall | F1-Score | Support |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | `Apple___Apple_scab` | 100.0% | 100.0% | 1.0000 | 95.0 |
| 2 | `Apple___Black_rot` | 100.0% | 100.0% | 1.0000 | 94.0 |
| 3 | `Apple___Cedar_apple_rust` | 100.0% | 100.0% | 1.0000 | 42.0 |
| 4 | `Apple___healthy` | 99.6% | 100.0% | 0.9980 | 248.0 |
| 5 | `Background_without_leaves` | 98.3% | 100.0% | 0.9914 | 172.0 |
| 6 | `Blueberry___healthy` | 100.0% | 100.0% | 1.0000 | 226.0 |
| 7 | `Cherry___Powdery_mildew` | 100.0% | 99.4% | 0.9968 | 159.0 |
| 8 | `Cherry___healthy` | 100.0% | 99.2% | 0.9961 | 129.0 |
| 9 | `Corn___Cercospora_leaf_spot Gray_leaf_spot` | 97.3% | 93.6% | 0.9542 | 78.0 |
| 10 | `Corn___Common_rust` | 100.0% | 99.4% | 0.9972 | 180.0 |
| 11 | `Corn___Northern_Leaf_Blight` | 96.1% | 98.7% | 0.9735 | 149.0 |
| 12 | `Corn___healthy` | 100.0% | 100.0% | 1.0000 | 175.0 |
| 13 | `Grape___Black_rot` | 100.0% | 100.0% | 1.0000 | 177.0 |
| 14 | `Grape___Esca_(Black_Measles)` | 100.0% | 100.0% | 1.0000 | 208.0 |
| 15 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 100.0% | 100.0% | 1.0000 | 162.0 |
| 16 | `Grape___healthy` | 100.0% | 100.0% | 1.0000 | 64.0 |
| 17 | `Orange___Haunglongbing_(Citrus_greening)` | 100.0% | 100.0% | 1.0000 | 827.0 |
| 18 | `Peach___Bacterial_spot` | 100.0% | 100.0% | 1.0000 | 346.0 |
| 19 | `Peach___healthy` | 100.0% | 100.0% | 1.0000 | 55.0 |
| 20 | `Pepper,_bell___Bacterial_spot` | 100.0% | 100.0% | 1.0000 | 151.0 |
| 21 | `Pepper,_bell___healthy` | 100.0% | 100.0% | 1.0000 | 223.0 |
| 22 | `Potato___Early_blight` | 100.0% | 100.0% | 1.0000 | 150.0 |
| 23 | `Potato___Late_blight` | 100.0% | 99.3% | 0.9967 | 150.0 |
| 24 | `Potato___healthy` | 100.0% | 100.0% | 1.0000 | 24.0 |
| 25 | `Raspberry___healthy` | 100.0% | 100.0% | 1.0000 | 57.0 |
| 26 | `Soybean___healthy` | 99.7% | 100.0% | 0.9987 | 764.0 |
| 27 | `Squash___Powdery_mildew` | 100.0% | 100.0% | 1.0000 | 276.0 |
| 28 | `Strawberry___Leaf_scorch` | 100.0% | 100.0% | 1.0000 | 167.0 |
| 29 | `Strawberry___healthy` | 100.0% | 100.0% | 1.0000 | 69.0 |
| 30 | `Tomato___Bacterial_spot` | 99.4% | 99.7% | 0.9953 | 320.0 |
| 31 | `Tomato___Early_blight` | 98.6% | 95.3% | 0.9695 | 150.0 |
| 32 | `Tomato___Late_blight` | 99.0% | 98.3% | 0.9860 | 287.0 |
| 33 | `Tomato___Leaf_Mold` | 100.0% | 100.0% | 1.0000 | 144.0 |
| 34 | `Tomato___Septoria_leaf_spot` | 100.0% | 100.0% | 1.0000 | 267.0 |
| 35 | `Tomato___Spider_mites Two-spotted_spider_mite` | 99.6% | 100.0% | 0.9980 | 252.0 |
| 36 | `Tomato___Target_Spot` | 99.1% | 100.0% | 0.9953 | 212.0 |
| 37 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 100.0% | 100.0% | 1.0000 | 805.0 |
| 38 | `Tomato___Tomato_mosaic_virus` | 100.0% | 100.0% | 1.0000 | 57.0 |
| 39 | `Tomato___healthy` | 100.0% | 100.0% | 1.0000 | 240.0 |

---

## 5. Methodological Strengths & Known Limitations

### Strengths:
1. **Strict Held-Out Separation**: The test set was untouched during hyperparameter tuning and model training.
2. **Zero Cryptographic Hash Leakage**: Cluster-aware grouping guaranteed that identical images never crossed the train/test barrier.
3. **Real-Time Efficiency**: MobileNetV3-Large executes forward passes in single-digit milliseconds, making it suitable for field deployment.

### Known Limitations:
1. **Single-Leaf Focus**: Images in the dataset predominantly feature segmented single leaves rather than full-canopy drone views.
2. **Imbalance Variance**: Minority classes (e.g., `Potato___healthy` with only 24 test samples) exhibit wider confidence intervals than well-represented classes.