# Chest X-Ray Diagnosis — Multi-Label Classification

A deep learning portfolio project exploring multi-label classification of chest X-ray images using DenseNet-121, PyTorch, and Streamlit.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Overview

The application predicts scores for 15 labels based on chest X-ray images: 14 findings from the NIH ChestXray14 label set and an additional `No Finding` label.

The project includes a model training pipeline, evaluation scripts, and an interactive Streamlit interface.

## Technologies

- **Programming Language:** Python
- **Deep Learning Frameworks:** PyTorch, Torchvision
- **Model Architecture:** DenseNet-121
- **Data Processing:** Pandas, NumPy
- **Machine Learning Utilities:** Scikit-learn
- **Web Application:** Streamlit
- **Image Processing and Visualization:** OpenCV, Matplotlib

## Dataset

This project uses the NIH ChestXray14 dataset. The dataset and medical images are not included in this repository.

Obtain the dataset from its authorized source and follow its applicable terms of use. Configure the processed CSV files and image paths for your local environment.

## Installation

```bash
git clone https://github.com/mehdi-bakkali/chest-X-ray-diagnosis-project.git
cd chest-X-ray-diagnosis-project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

Run the training script with the path to your processed dataset:

```bash
python -m src.train \
  --data-dir /path/to/processed \
  --epochs 3 \
  --batch-size 8 \
  --workers 0 \
  --output-dir outputs-15-classes
```

## Run the Application

After training a compatible checkpoint and configuring its path in `app.py`, launch the application:

```bash
streamlit run app.py
```

## Evaluation and Explainability

The 15-class DenseNet-121 model will be evaluated on an independent test set using per-label AUROC, precision, recall, and F1-score, together with macro-averaged metrics and class prevalence. Results will be interpreted in the context of class imbalance, data-splitting methodology, uncertainty, and dataset limitations.

Grad-CAM will be used to visualize image regions that may influence the model's predictions. These visualizations are exploratory and do not establish clinical validity.

### Evaluate the Model

```bash
python -m src.evaluate \
  --data-dir data/processed \
  --checkpoint outputs-15-classes/best_model.pt
```

### Generate Grad-CAM Visualizations

```bash
python -m src.gradcam \
  --data-dir data/processed \
  --checkpoint outputs-15-classes/best_model.pt \
  --image data/images/00000001_000.png \
  --label Atelectasis
```

**Note:** These commands assume that `src.evaluate` and `src.gradcam` are implemented and accept the specified arguments. Update the paths to match your dataset and model checkpoint locations. Evaluation results should be reported only after testing is complete.

## Limitations

- Predictions are experimental and are not confirmed diagnoses.
- The dataset may contain label noise and class imbalance.
- Performance may vary across patient populations, imaging equipment, and acquisition settings.
- Prediction scores are not necessarily calibrated probabilities.
- Grad-CAM visualizations are qualitative and should not be interpreted as medical explanations.

## License

A license for the project code will be added after reviewing the applicable dataset terms and third-party dependencies.

