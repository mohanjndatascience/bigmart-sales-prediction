import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.packages import *
from src.config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.utils.utils import preprocess_data

if __name__ == "__main__":
    train_file, test_file = preprocess_data(
        raw_train_path=RAW_DATA_PATH / "train.csv",
        raw_test_path=RAW_DATA_PATH / "test.csv",
        processed_data_path=PROCESSED_DATA_PATH
    )