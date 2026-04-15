# DeepFake Image Detection (Binary Classification)

**Team No : 06**

| Member | Roll No |
|--------|---------|
| Ananthan K | AM.SC.P2ARI25006 |
| Amrutha M | AM.SC.P2ARI25004 |
| Mohammed Aariff L | AM.SC.P2ARI25014 |

**Submitted to:** Akshara P Byju

---

## Abstract

The rapid advancement of generative AI has enabled the creation of highly realistic DeepFake images, raising significant concerns about misinformation, digital manipulation, and security. This project develops a deep learning–based system for detecting DeepFake facial images using a transfer learning approach with **EfficientNet-B4**, evaluated using accuracy, precision, recall, F1-score, confusion matrix analysis, and **Grad-CAM** visualizations.

---

## Project Structure

```
deepfake-detection/
├── src/
│   ├── model.py        # EfficientNet-B4 classifier
│   ├── dataset.py      # PyTorch Dataset for Kaggle dataset
│   ├── train.py        # Full training pipeline
│   └── predict.py      # Inference script (single image / folder)
├── utils/
│   ├── metrics.py      # Accuracy, F1, confusion matrix, plots
│   ├── grad_cam.py     # Grad-CAM visualization
│   └── logger.py       # Formatted logging
├── notebooks/
│   └── deepfake_detection.ipynb  # End-to-end Jupyter notebook
├── outputs/            # Generated during training
│   ├── best_model.pth
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── test_metrics.json
│   └── grad_cam/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API

Set your Kaggle credentials so `kagglehub` can download the dataset automatically:

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

Or place `kaggle.json` in `~/.kaggle/kaggle.json`.

---

## Dataset

The project uses the **Deepfake and Real Images** dataset from Kaggle:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
print("Path to dataset files:", path)
```

**Dataset structure expected:**
```
Dataset/
  Train/
    Real/  *.jpg
    Fake/  *.jpg
  Test/
    Real/
    Fake/
  Validation/   (optional)
    Real/
    Fake/
```

---

## Training

```bash
python src/train.py
```

With custom arguments:

```bash
python src/train.py \
  --dataset_path /path/to/dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir outputs
```

### Key Training Features

| Feature | Details |
|---------|---------|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Loss | BCEWithLogitsLoss with class-balanced pos_weight |
| Optimizer | AdamW (weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Augmentation | Random crop, flip, color jitter, rotation |
| Regularization | Dropout 0.4, gradient clipping |
| Mixed Precision | torch.cuda.amp (AMP) |
| Early Stopping | Patience = 5 epochs (by val F1) |

---

## Inference

**Single image:**
```bash
python src/predict.py \
  --image path/to/face.jpg \
  --checkpoint outputs/best_model.pth
```

**Folder of images:**
```bash
python src/predict.py \
  --image_dir path/to/faces/ \
  --checkpoint outputs/best_model.pth \
  --output_csv results.csv
```

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## Grad-CAM Visualization

Grad-CAM highlights the image regions most influential in the model's decision, enabling interpretability. Generated automatically after training in `outputs/grad_cam/`.

---

## Notebook

For an end-to-end walkthrough including dataset exploration, training, evaluation, and Grad-CAM analysis, open:

```bash
jupyter notebook notebooks/deepfake_detection.ipynb
```

---

## Results (Expected)

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~96% |
| Recall | ~97% |
| F1-Score | ~97% |
| ROC-AUC | ~99% |

*Actual results will vary depending on hardware, epochs, and data splits.*

---

## Tech Stack

- **PyTorch** — deep learning framework
- **torchvision** — EfficientNet-B4 pretrained model
- **kagglehub** — dataset download
- **scikit-learn** — evaluation metrics
- **matplotlib / seaborn** — visualizations
- **Grad-CAM** — model interpretability

---

## License

MIT License — free to use for academic and research purposes.
