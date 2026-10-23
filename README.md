# Chest X-ray Multi-label Classification (DenseNet-121)

Beginner-friendly research/portfolio starter project using the NIH ChestX-ray14 dataset and PyTorch.

## Important limitations
- This is an educational research project, not a medical device or diagnostic tool.
- Dataset labels are noisy and performance on a small subset is not evidence of clinical accuracy.
- Use patient-level train/validation/test splits to reduce leakage.
- Grad-CAM visualizations are qualitative and do not prove that a model's prediction is correct.

## 1. Environment
Activate your virtual environment, then install requirements:

```bash
python -m pip install -r requirements.txt
```

On Apple Silicon, check MPS support:

```bash
python -c "import torch; print('Apple MPS available:', torch.backends.mps.is_available())"
```

## 2. Download data
Get the NIH ChestX-ray14 files from https://nihcc.app.box.com/v/ChestXray-Nih . Read the dataset terms and instructions. At minimum, download `Data_Entry_2017.csv` and the image archives you plan to use. Extract images into a folder such as `data/images/`.

Expected layout:

```text
data/
  Data_Entry_2017.csv
  images/
    00000001_000.png
    ...
```

Do not commit the dataset to Git.

## 3. Prepare patient-level splits
Run from the project root:

```bash
python -m src.prepare_data --csv data/Data_Entry_2017.csv --images data/images --out data/processed --limit 1000
```

The optional `--limit` is for a small smoke test only. The script creates CSV manifests with patient-level splits and reports missing image files. For serious evaluation, use the full available dataset and a predefined, documented patient-level split.

## 4. Train

```bash
python -m src.train --data-dir data/processed --epochs 2 --batch-size 8
```

Adjust batch size if you run out of memory. Training defaults to a small number of epochs for a smoke test.

## 5. Evaluate

```bash
python -m src.evaluate --data-dir data/processed --checkpoint outputs/best_model.pt
```

## 6. Grad-CAM

```bash
python -m src.gradcam --data-dir data/processed --checkpoint outputs/best_model.pt --image data/images/00000001_000.png --label Atelectasis
```

Results are exploratory. Report per-label AUROC and threshold-dependent precision/recall/F1, along with class prevalence, split method, uncertainty, and limitations.

## Dataset and paper
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-Nih
- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*: https://arxiv.org/abs/1705.02315
