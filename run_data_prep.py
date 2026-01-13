import os
import pandas as pd
from src.preprocessing import (
    load_metadata, 
    process_and_save_images
)

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')

TRAIN_OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'train_cropped')
TEST_OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'test_cropped')
# ---------------------

def main():
    train_df, test_df = load_metadata(DATA_DIR)

    process_and_save_images(
        df=train_df, 
        source_root=DATA_DIR, 
        target_root=TRAIN_OUTPUT_DIR
    )

    process_and_save_images(
        df=test_df, 
        source_root=DATA_DIR, 
        target_root=TEST_OUTPUT_DIR
    )

if __name__ == "__main__":
    main()