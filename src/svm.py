import os
import cv2
import numpy as np
import pickle
import time
import argparse 
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_images_from_folder(folder_path):
    """
    Loads images and labels from the processed folder structure.
    Returns:
        X: list of images (numpy arrays)
        y: list of labels (integers)
    """
    images = []
    labels = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")
        
    for class_id in sorted(os.listdir(folder_path)):
        class_dir = os.path.join(folder_path, class_id)
        
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                images.append(img)
                labels.append(int(class_id))
                
    return images, np.array(labels)

def extract_hog_features(images):
    """
    Converts a list of raw images into HOG feature vectors.
    """
    hog_features = []
    print("Extracting HOG features...")
    
    for img in tqdm(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       transform_sqrt=True)
        
        hog_features.append(features)
        
    return np.array(hog_features)

def flatten_images(images):
    """
    Converts a list of raw images into flattened raw pixel vectors.
    1. Converts BGR to Grayscale.
    2. Flattens the 2D matrix into a 1D vector.
    """
    flat_features = []
    print("Flattening raw images (No HOG)...")
    
    for img in tqdm(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_vector = gray.flatten()
        
        flat_features.append(flat_vector)
        
    return np.array(flat_features)

def train_and_evaluate_svm(train_dir, test_dir, use_hog=False, model_save_path=None):
    
    print("--- Loading Training Data ---")
    X_train_raw, y_train = load_images_from_folder(train_dir)
    print(f"Loaded {len(X_train_raw)} training images.")
    
    print("\n--- Loading Test Data ---")
    X_test_raw, y_test = load_images_from_folder(test_dir)
    print(f"Loaded {len(X_test_raw)} test images.")
    
    # --- FEATURE EXTRACTION SWITCH ---
    if use_hog:
        print("\n[INFO] Mode: HOG Features Enabled")
        X_train_feats = extract_hog_features(X_train_raw)
        X_test_feats = extract_hog_features(X_test_raw)
    else:
        print("\n[INFO] Mode: Raw Pixels (No HOG)")
        X_train_feats = flatten_images(X_train_raw)
        X_test_feats = flatten_images(X_test_raw)
    
    print(f"\nFeature Vector Shape: {X_train_feats.shape[1]}")
    
    # --- TRAINING TIMING ---
    print("\n--- Training SVM ---")
    train_start = time.time()
    
    clf = svm.SVC(kernel='rbf', C=10.0, gamma='scale') 
    clf.fit(X_train_feats, y_train)
    
    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training Time: {train_time:.2f} seconds")
    
    # --- INFERENCE TIMING ---
    print("\n--- Evaluating (Inference) ---")
    inference_start = time.time()
    
    y_pred = clf.predict(X_test_feats)
    
    inference_end = time.time()
    inference_time = inference_end - inference_start
    time_per_img = (inference_time / len(X_test_feats)) * 1000 # in ms
    
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    print(f"Inference Time per Image: {time_per_img:.4f} ms")
    
    # --- METRICS ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if model_save_path:
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM for Traffic Sign Classification.")
    
    parser.add_argument('--hog', action='store_true', help="Use HOG features instead of raw pixels.")
    args = parser.parse_args()
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'processed_data')

    train_dir = os.path.join(processed_dir, 'train_cropped')
    test_dir = os.path.join(processed_dir, 'test_cropped')
    
    model_name = 'svm_hog_model.pkl' if args.hog else 'svm_raw_model.pkl'
    model_path = os.path.join(base_dir, 'models', model_name)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    train_and_evaluate_svm(train_dir, test_dir, use_hog=args.hog, model_save_path=model_path)
 
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal Runtime: {minutes}m {seconds}s")