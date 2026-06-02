import pandas as pd
import re
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            f"Make sure data/raw_skills.csv exists."
        )

    df = pd.read_csv(filepath)

    required_columns = {"job_role", "skills_required"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def clean_skills_text(skills_text: str) -> str:
    if pd.isna(skills_text) or not isinstance(skills_text, str):
        return ""
    lowercased = skills_text.lower()
    normalized = re.sub(r"[^a-z0-9\+\#\.\s]", " ", lowercased)
    collapsed = re.sub(r"\s+", " ", normalized).strip()
    return collapsed


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["job_role"] = df["job_role"].str.strip()
    df["skills_cleaned"] = df["skills_required"].apply(clean_skills_text)
    df = df.dropna(subset=["skills_cleaned"])
    df = df[df["skills_cleaned"] != ""]
    df = df.drop_duplicates(subset=["job_role"])
    df = df.reset_index(drop=True)
    return df


def parse_user_skills(raw_input: list) -> str:
    joined = " ".join([str(s).lower().strip() for s in raw_input])
    cleaned = re.sub(r"[^a-z0-9\+\#\.\s]", " ", joined)
    return re.sub(r"\s+", " ", cleaned).strip()