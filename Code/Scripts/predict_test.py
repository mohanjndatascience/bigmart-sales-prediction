import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.packages import *
from src.config.config import PROCESSED_DATA_PATH, OUTPUT_PATH, VAL_RESULTS_PATH
from src.utils.utils import predict_test_data

if __name__ == "__main__":
    preprocessed_test_file = PROCESSED_DATA_PATH / "test_preprocessed.csv"
    results_df = pd.read_csv(VAL_RESULTS_PATH / "model_validation_results.csv")
    top_model_paths = results_df[results_df['Model'] != "Top_3_Combined"].sort_values(by="RMSE").head(3)['Model Path'].tolist()
    predict_test_data(
        preprocessed_test_file=preprocessed_test_file,
        top_model_paths=top_model_paths,
        output_path=OUTPUT_PATH
    )