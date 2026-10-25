"""Carga y evidencia del dataset de Hugging Face."""

from datasets import load_dataset


DATASET_ID = "scikit-learn/auto-mpg"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DATASET_SPLIT = "train"


def load_dataframe():
    """Descarga el split configurado y lo convierte en un DataFrame."""
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    return dataset.to_pandas()
