from pathlib import Path

# Base project directory (repo root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / "Data" / "Input_Data"
PROCESSED_DATA_PATH = BASE_DIR / "Data" / "Preprocessed_Data"
MODEL_PATH = BASE_DIR / "Data" / "Models"
VAL_RESULTS_PATH = BASE_DIR / "Data" / "Validation_Results"
OUTPUT_PATH = BASE_DIR / "Data" / "Output_Data"