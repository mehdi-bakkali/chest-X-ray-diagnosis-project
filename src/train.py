"""Train a DenseNet-121 multi-label classifier."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from .labels import LABELS

class XrayDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.image_path).convert("RGB")
        if self.transform: image = self.transform(image)
        labels = torch.tensor(row[LABELS].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, labels

def make_model():
    try:
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
    except Exception as exc:
        print(f"Could not load pretrained weights ({exc}); using random initialization.")
        model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds, val_ds = XrayDataset(data_dir / "train.csv", transform), XrayDataset(data_dir / "val.csv", transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    model = make_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        train_images = 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train")
        ):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total += loss.item() * images.size(0)
            train_images += images.size(0)

        train_loss = total / max(train_images, 1)
        model.eval(); val_total = 0.0; val_images = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val")
            ):
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images),labels)
                val_total += loss.item() * images.size(0)
                val_images += images.size(0)
        val_loss = val_total / max(val_images, 1)
    
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "labels": LABELS, "val_loss": val_loss}, output_dir / "best_model.pt")
    print(f"Saved best checkpoint to {output_dir / 'best_model.pt'}")
if __name__ == "__main__": main()
