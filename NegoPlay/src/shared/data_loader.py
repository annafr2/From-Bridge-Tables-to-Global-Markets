"""
src/shared/data_loader.py
=========================
Load and validate the NegoPlay bridge dataset.

All downstream code should import from here — never read CSVs directly.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must exist in the CSV
REQUIRED_COLUMNS: list[str] = [
    "match_id", "round", "board", "room",
    "contract", "declarer", "tricks",
    "ns_score", "ew_score",
    "open_north", "open_south", "open_east", "open_west",
    "closed_north", "closed_south", "closed_east", "closed_west",
    "has_bidding", "has_cards",
]

# Dtype hints for faster loading
COLUMN_DTYPES: dict[str, str] = {
    "board": "Int16",
    "round": "Int16",
    "tricks": "Int8",
    "ns_score": "Int16",
    "ew_score": "Int16",
    "has_bidding": "boolean",
    "has_cards": "boolean",
}


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load the bridge dataset from a CSV file.

    Args:
        path: Absolute or relative path to all_matches_full.csv.

    Returns:
        DataFrame with validated schema and clean dtypes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Set DATA_PATH in your .env file to the correct location."
        )

    logger.info("Loading dataset from %s ...", path)

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",   # handles BOM from Excel exports
        low_memory=False,
    )

    # --- Schema validation ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # --- Apply dtype hints where column exists ---
    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                logger.warning("Could not cast column '%s' to %s — skipping.", col, dtype)

    logger.info(
        "Loaded %d rows × %d columns. "
        "Bidding coverage: %.1f%% | Cards coverage: %.1f%%",
        len(df),
        len(df.columns),
        df["has_bidding"].mean() * 100 if "has_bidding" in df.columns else 0,
        df["has_cards"].mean() * 100 if "has_cards" in df.columns else 0,
    )

    return df
