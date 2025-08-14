import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.packages import *
from src.config.config import PROCESSED_DATA_PATH, MODEL_PATH, VAL_RESULTS_PATH
from src.utils.utils import train_and_validate_models

if __name__ == "__main__":
    preprocessed_train_file = PROCESSED_DATA_PATH / "train_preprocessed.csv"
    top_models, results_df = train_and_validate_models(
        preprocessed_train_file=preprocessed_train_file,
        models_path=MODEL_PATH,
        val_results_path=VAL_RESULTS_PATH,
        top_n=3
    )