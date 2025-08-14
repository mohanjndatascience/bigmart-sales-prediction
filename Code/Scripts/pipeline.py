import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.packages import *
from src.config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, VAL_RESULTS_PATH, OUTPUT_PATH
from src.utils.utils import preprocess_data, train_and_validate_models, predict_test_data


if __name__ == "__main__":
    # Step 1: Preprocessing
    train_file, test_file = preprocess_data(
        raw_train_path=RAW_DATA_PATH / "train.csv",
        raw_test_path=RAW_DATA_PATH / "test.csv",
        processed_data_path=PROCESSED_DATA_PATH
    )

    # Step 2: Train & Validate
    top_models, results_df = train_and_validate_models(
        preprocessed_train_file=train_file,
        models_path=MODEL_PATH,
        val_results_path=VAL_RESULTS_PATH,
        top_n=3
    )

    # Step 3: Predict on Test Data
    predict_test_data(
        preprocessed_test_file=test_file,
        top_model_paths=top_models,
        output_path=OUTPUT_PATH
    )