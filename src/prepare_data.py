
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .labels import LABELS, parse_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of patients for a smoke test",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images_root)
    out_dir = Path(args.out)

    df = pd.read_csv(csv_path)

    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Index PNG files across images_001/images through images_012/images.
    image_lookup = {}
    for image_dir in sorted(images_root.glob("images_*")):
        nested_dir = image_dir / "images"
        if nested_dir.is_dir():
            for path in nested_dir.glob("*.png"):
                image_lookup[path.name] = str(path)

    df["image_path"] = df["Image Index"].map(image_lookup)
    missing_count = int(df["image_path"].isna().sum())
    df = df.dropna(subset=["image_path"]).copy()

    if df.empty:
        raise FileNotFoundError(
            "No CSV filenames matched the PNG files. Check --csv and --images-root."
        )

    if args.limit > 0:
        patients = df["Patient ID"].drop_duplicates().head(args.limit)
        df = df[df["Patient ID"].isin(patients)].copy()

    vectors = df["Finding Labels"].map(parse_labels)
    for i, label in enumerate(LABELS):
        df[label] = vectors.map(lambda vector: vector[i])

    # Keep each patient's images within one split to reduce data leakage.
    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, temp_idx = next(
        split1.split(df, groups=df["Patient ID"])
    )
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    split2 = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=43
    )
    val_idx, test_idx = next(
        split2.split(temp_df, groups=temp_df["Patient ID"])
    )
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "Image Index", "image_path", "Patient ID", "Finding Labels", *LABELS
    ]

    for name, part in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        part[columns].to_csv(out_dir / f"{name}.csv", index=False)
        print(
            f"{name}: {len(part)} images, "
            f"{part['Patient ID'].nunique()} patients"
        )

    print(f"CSV rows without a matching PNG: {missing_count}")
    if args.limit:
        print("SMOKE TEST ONLY: do not use these splits for final evaluation.")


if __name__ == "__main__":
    main()
