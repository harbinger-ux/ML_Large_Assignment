# ML Large Assignment

## Dataset Setup

1. Download the dataset from the following link:

   [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

2. Extract the downloaded `.zip` file and rename to `data` then place it inside the `ML_Large_Assignment` directory.

   The folder structure should look like this:

   ```
   ML_Large_Assignment
   ├── data
   ├── outputs
   ├── run_data_prep.py
   └── src
   ```

## Data Preparation

3. Execute the data preprocessing script:

   ```bash
   python run_data_prep.py
   ```

   This will create the `processed_data/` folder.

## Classic Machine Learning Models

4. You can enable HOG (Histogram of Oriented Gradients) with classic machine learning algorithms (`knn.py`, `random_forest.py`, `svm.py`) by including the `--hog` flag in the command.

   Example:

   ```bash
   python svm.py --hog
   ```

   The results will be displayed directly in the terminal.

## Deep Learning Model

5. The `deep_learning.py` script is executed without any additional flags.
   ```bash
   python deep_learning.py
   ```
