import os
import cv2
import pandas as pd
from tqdm import tqdm  

def load_metadata(data_dir):
    train_csv_path = os.path.join(data_dir, 'Train.csv')
    test_csv_path = os.path.join(data_dir, 'Test.csv')
    
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Could not find Train.csv or Test.csv in {data_dir}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    return train_df, test_df

def process_and_save_images(df, source_root, target_root, target_size=(32, 32)):
    """
    Reads images from source_root based on df['Path'], crops them using ROI,
    resizes them, and saves them to target_root.
    
    Organizes output as: target_root / ClassId / image_name.png
    """
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        
    print(f"Processing {len(df)} images to {target_root}...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # 1. Construct full path
        # Note: In the CSV, Path is like "Train/20/..." or "Test/..."
        original_image_path = os.path.join(source_root, row['Path'])
        
        # 2. Read Image
        img = cv2.imread(original_image_path)
        if img is None:
            print(f"Warning: Could not read image {original_image_path}")
            continue
            
        # 3. Crop using ROI coordinates
        # Coordinates in CSV are: Roi.X1, Roi.Y1, Roi.X2, Roi.Y2
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        cropped_img = img[y1:y2, x1:x2]
        
        # 4. Resize
        try:
            resized_img = cv2.resize(cropped_img, target_size)
        except Exception as e:
            # Fallback if crop is invalid or zero-size
            print(f"Error resizing image {original_image_path}: {e}")
            continue
            
        # 5. Save to processed folder
        # Structure: processed_data/train/20/00020_00000_00000.png
        class_folder = os.path.join(target_root, str(row['ClassId']))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        # Extract original filename
        filename = os.path.basename(row['Path'])
        save_path = os.path.join(class_folder, filename)
        
        cv2.imwrite(save_path, resized_img)