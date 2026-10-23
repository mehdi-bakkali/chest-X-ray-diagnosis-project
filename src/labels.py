
"""Label definitions and NIH ChestX-ray14 label parsing."""

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
]


def parse_labels(label_text: str) -> list[float]:
    """Convert NIH Finding Labels into a 15-label multi-hot vector."""
    found = set(str(label_text).split("|"))
    labels = [float(label in found) for label in LABELS]

    # No Finding is positive only when no pathology is listed.
    labels[-1] = float(sum(labels[:-1]) == 0)

    return labels
