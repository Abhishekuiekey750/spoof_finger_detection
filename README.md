# Fingerprint Spoof Detection using LBP

Detect whether a fingerprint image is **real (live)** or **fake (spoof)** using texture-based machine learning.

## Overview

Fingerprint spoofing (using printed, molded, or artificial fingerprints) can bypass weak biometric systems.  
This project helps reduce that risk by analyzing fingerprint texture patterns and classifying images as real or fake.

Why it matters:
- Improves trust in biometric authentication systems
- Adds a lightweight anti-spoofing layer before fingerprint matching
- Uses interpretable image-texture features (LBP + histograms)

## Features

- Image preprocessing pipeline (grayscale, resize, normalization) 🖼️
- Texture feature extraction with Local Binary Patterns (LBP) 🔍
- Histogram-based feature vectors for model input 📊
- Model training using classic ML algorithms (Random Forest, SVM) 🤖
- Evaluation with accuracy and confusion-matrix-style analysis ✅

## Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Project Workflow

1. Collect real and spoof fingerprint images
2. Preprocess each image:
   - Convert to grayscale
   - Resize to a fixed shape
   - Normalize pixel values
3. Extract LBP texture map from each image
4. Convert LBP output into histogram features
5. Split data into train/test sets
6. Train ML model (Random Forest / SVM)
7. Evaluate on unseen test data (accuracy + error analysis)
8. Save trained model for later prediction

## Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/Abhishekuiekey750/spoof_finger_detection.git
cd spoof_finger_detection
```

### 2) Create and activate virtual environment (recommended)

```bash
python -m venv .venv
```

- On Windows:
```bash
.venv\Scripts\activate
```

- On macOS/Linux:
```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

## Usage

### Run evaluation script

```bash
python evaluate_only.py
```

### Predict on sample/input fingerprint images

Use your prediction script:

```bash
python recognize.py
```

If needed, update paths inside the script to point to:
- input fingerprint image
- trained model file (`fingerprint_model.pkl`)

## Results

The model is evaluated using standard classification metrics:
- Accuracy
- Confusion matrix
- Class-level performance (real vs spoof)

Current repository includes:
- Saved model: `fingerprint_model.pkl`
- Evaluation artifact: `test_dataset_confusion_matrix.png`

> Note: Exact accuracy depends on dataset quality, spoof diversity, and train/test split strategy.

## Future Improvements

- Add larger and more diverse spoof datasets
- Try deep learning models (CNN-based anti-spoofing)
- Add cross-validation and hyperparameter tuning
- Build a simple API or web app for real-time testing
- Add automated tests and experiment tracking

## Folder Structure

Typical structure for this project:

```text
spoof_finger_detection/
├── README.md
├── localbinarypatterns.py
├── recognize.py
├── evaluate_only.py
├── fingerprint_model.pkl
└── test_dataset_confusion_matrix.png
```

## Contributing

Contributions are welcome 🙌  
If you want to improve preprocessing, feature engineering, or model performance:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request with a clear description

## License

No license file is currently defined in this repository.  
Consider adding an open-source license (for example, MIT) to clarify usage permissions.