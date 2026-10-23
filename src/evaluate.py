
"""Evaluate a trained chest X-ray classifier on the validation set."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train import XrayDataset, make_model
from src.labels import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where evaluation metrics will be saved",
    )

    args = parser.parse_args()

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    data_dir = Path(args.data_dir)
    dataset = XrayDataset(data_dir / "val.csv", transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["labels"] != LABELS:
        raise ValueError("Checkpoint labels do not match current LABELS.")

    model = make_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating validation set"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets.numpy())


    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    results = []

    for i, label in enumerate(LABELS):
        positives = int(y_true[:, i].sum())
        negatives = len(y_true) - positives

        # AUROC needs both positive and negative examples.
        if positives == 0 or negatives == 0:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])

        # Average Precision is undefined without positive examples.
        if positives == 0:
            ap = float("nan")
        else:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])

        results.append({
            "label": label,
            "positive_cases": positives,
            "negative_cases": negatives,
            "AUROC": auc,
            "Average_Precision": ap,
        })

    report = pd.DataFrame(results)

    print("\nPer-label validation results:")
    print(report.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    print("\nMacro-average metrics:")
    print(f"Mean AUROC: {report['AUROC'].mean():.4f}")
    print(f"Mean Average Precision: {report['Average_Precision'].mean():.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation_metrics.csv"

    report.to_csv(output_path, index=False)
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
