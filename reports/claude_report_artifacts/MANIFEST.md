# Claude Report Artifacts Manifest

This directory contains verified training, evaluation, class mapping, and dataset artifacts collected from the CropCare AI codebase for Claude report updating.

| Artifact File | Original Source Path | Size (Bytes) | SHA-256 (first 12 chars) | Description |
| :--- | :--- | :---: | :---: | :--- |
| eval_MobileNetV3_Large.json | C:\miniproj\CropCareAI\reports\eval_MobileNetV3_Large.json | 33,355 | c3684aa9c3a4 | Current 38-class model test evaluation report (contains full 38x38 confusion matrix, test loss, eval time, per-class TP/FP/FN/TN) |
| eval_MobileNetV3_Large.txt | C:\miniproj\CropCareAI\reports\eval_MobileNetV3_Large.txt | 2,424 | cf5d6b96eff1 | Current 38-class model readable text evaluation summary with per-class precision, recall, F1, support |
| training_MobileNetV3_Large.json | C:\miniproj\CropCareAI\reports\training_MobileNetV3_Large.json | 11,300 | 7be8b9a33209 | Current 38-class model training history (all 12 epochs, head and fine-tuning stages, train/val loss, acc, macro-F1, LR) |
| class_mapping.json | C:\miniproj\CropCareAI\reports\class_mapping.json | 4,559 | 7b3131065ab9 | Current 38-class index-to-label mapping and per-class sample counts |
| dataset_audit.json | C:\miniproj\CropCareAI\reports\dataset_audit.json | 21,587 | 08d109e7b930 | Audit of 54,305 raw PlantVillage images (resolutions, file sizes, zero unreadable images) |
| split_summary.json | C:\miniproj\CropCareAI\data\processed\split_summary.json | 6,731 | 1fd4ec0a1f9e | Dataset partition summary (37,997 train, 8,129 val, 8,179 test), class imbalance ratio, and zero-leakage verification |
| training_CropCareNet_M.json | C:\miniproj\CropCareAI\reports\training_CropCareNet_M.json | 1,434 | 8cf1f8c3d979 | Custom CNN baseline (CropCareNet_M, 11.3M params, 2 epochs) comparison training report |
| previous_confusion_matrix_39class.json | C:\miniproject\reports\confusion_matrix.json | 17,123 | 58a8d714ec55 | Existing standalone confusion matrix data from previous 39-class project |
| previous_test_evaluation_and_latency.md | C:\miniproject\reports\test_evaluation_report.md | 5,017 | e1b952f252fd | Previous model test evaluation report containing measured inference latency benchmarks (1.176 ms/img, 37.64 ms/batch) |

## Notes on Requested Items
- **Confusion Matrix Data**: For the current 38-class model, the complete 38x38 confusion matrix is embedded directly inside eval_MobileNetV3_Large.json under keys confusion_matrix (38x38 integer matrix) and confusion_matrix_labels (38 class names). The existing standalone confusion matrix from the previous 39-class project is also included as previous_confusion_matrix_39class.json.
- **Per-Class Metrics & Error Pairs**: Full per-class precision, recall, F1, support, TP, FP, FN, TN are provided in eval_MobileNetV3_Large.json and eval_MobileNetV3_Large.txt. The 27 off-diagonal error pairs (106 total misclassifications out of 8,179 test images) are derived from the 38x38 confusion matrix.
- **Latency Benchmarks**: Current MobileNetV3 evaluation total time was 43.27s for 8,179 images (~5.29 ms/image). Hardware latency benchmarks measured on RTX 3050 (1.176 ms single image, 37.64 ms batch) are documented in previous_test_evaluation_and_latency.md.
- **Screenshots & Reports**: Located in the user workspace and downloads folder as verified in the report inspection.
