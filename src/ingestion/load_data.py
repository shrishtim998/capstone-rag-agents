
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_COLUMNS = [
    "instruction",
    "category",
    "intent",
    "response",
]


def load_dataset() -> pd.DataFrame:
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIR}"
        )

    dataset_path = csv_files[0]
    df = pd.read_csv(dataset_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    initial_rows = len(df)

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[
        df["instruction"].ne("")
        & df["response"].ne("")
    ].copy()

    df = df.drop_duplicates(
        subset=["instruction", "response"]
    ).reset_index(drop=True)

    print(f"Dataset: {dataset_path.name}")
    print(f"Original rows: {initial_rows}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Removed rows: {initial_rows - len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample records:")
    print(
        df[["instruction", "category", "intent", "response"]]
        .head(3)
        .to_string(index=False, max_colwidth=100)
    )

    return df


if __name__ == "__main__":
    load_dataset()
