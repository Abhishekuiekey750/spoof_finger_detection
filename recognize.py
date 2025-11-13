
'''
# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from imutils import paths
import argparse
import cv2
import os
import joblib
import numpy as np
# We use scikit-image for skeletonization (thinning)
# You may need to install it: pip install scikit-image
from skimage.morphology import skeletonize

# -----------------------------------------------------------------
# Step 1: Image Preprocessing (Noise Filtering)
# -----------------------------------------------------------------
def preprocess_image(image):
    """
    Applies standard preprocessing: grayscale conversion and noise filtering.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur for noise filtering
    # A (5, 5) kernel is a common choice.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

# -----------------------------------------------------------------
# Step 2: Ridge Thinning (Skeletonization)
# -----------------------------------------------------------------
def thin_image(image):
    """
    Applies ridge thinning (skeletonization) to a fingerprint.
    NOTE: This is for minutiae-based analysis and is NOT
    recommended for LBP texture-based analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image using Otsu's thresholding
    # We invert the image (THRESH_BINARY_INV) so ridges are white (255)
    # on a black (0) background.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Skeletonize requires the image to be 0 or 1, not 0 or 255
    binary_for_skeleton = binary / 255.0
    
    # Perform skeletonization
    skeleton = skeletonize(binary_for_skeleton)
    
    # Convert back to an 8-bit image (0 or 255) for consistency
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    return skeleton_img

# -----------------------------------------------------------------
# Step 3: Feature Extraction
# -----------------------------------------------------------------
def load_and_extract_features(directory, lbp_descriptor, use_thinning=False):
    """
    Loops over all images in a directory, preprocesses them,
    and extracts LBP features.
    """
    data = []
    labels = []

    print(f"[INFO] Processing images from {directory}...")
    for imagePath in paths.list_images(directory):
        # Load the image
        image = cv2.imread(imagePath)
        
        # Decide which preprocessing to apply
        if use_thinning:
            # Apply thinning (NOT recommended for LBP)
            processed_image = thin_image(image)
        else:
            # Apply standard preprocessing (noise filtering)
            processed_image = preprocess_image(image)
        
        # Describe the processed image using LBP
        hist = lbp_descriptor.describe(processed_image)
        
        # Extract the label (e.g., "Live" or "Fake") from the folder name
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(hist)
        
    return data, labels

# -----------------------------------------------------------------
# Step 4: Model Training
# -----------------------------------------------------------------
def train_and_save_model(train_data, train_labels, model_path="fingerprint_model.pkl"):
    """
    Trains a Linear SVM on the provided data and saves the model.
    """
    print("[INFO] Training Linear SVM model...")
    model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
    model.fit(train_data, train_labels)
    
    # Save the trained model to disk
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully as '{model_path}'")
    
    return model

# -----------------------------------------------------------------
# Step 5: Model Evaluation
# -----------------------------------------------------------------
def evaluate_model(model, lbp_descriptor, test_directory):
    """
    Evaluates the trained model on the testing dataset and prints metrics.
    """
    print(f"[INFO] Evaluating model on {test_directory}...")
    
    true_labels = []
    predicted_labels = []

    # Loop over the testing images one by one
    for imagePath in paths.list_images(test_directory):
        # Load and preprocess the image (using noise filtering)
        image = cv2.imread(imagePath)
        processed_image = preprocess_image(image) # Using our noise filtering
        
        # Extract LBP features
        hist = lbp_descriptor.describe(processed_image)
        
        # Classify the image
        prediction = model.predict(hist.reshape(1, -1))
        
        # Store the true and predicted labels
        true_labels.append(imagePath.split(os.path.sep)[-2])
        predicted_labels.append(prediction[0])

    # --- Calculate and Print Metrics ---
    
    print("\n--- Model Evaluation Results ---")
    
    # Generate confusion matrix
    # We specify the labels to ensure "Live" is Positive and "Fake" is Negative
    labels_order = ["Live", "Fake"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    
    # Extract TP, FN, FP, TN
    # cm[0][0] = True Positive (Predicted Live, Was Live)
    # cm[0][1] = False Negative (Predicted Fake, Was Live)
    # cm[1][0] = False Positive (Predicted Live, Was Fake)
    # cm[1][1] = True Negative (Predicted Fake, Was Fake)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    print(f"True Positives (Live predicted as Live): {TP}")
    print(f"False Negatives (Live predicted as Fake): {FN}")
    print(f"False Positives (Fake predicted as Live): {FP}")
    print(f"True Negatives (Fake predicted as Fake): {TN}")
    print("-" * 30)

    # Calculate metrics using scikit-learn (more reliable)
    # We set pos_label="Live" to define our "Positive" class
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live")
    recall = recall_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (for 'Live'): {precision * 100:.2f}%")
    print(f"Recall (for 'Live'): {recall * 100:.2f}%")
    print("----------------------------------\n")

# -----------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------
def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True,  help="path to the testing images")
    args = vars(ap.parse_args())

    # Initialize the LBP descriptor
    # 24 points, 8 radius (same as your original script)
    desc = LocalBinaryPatterns(24, 8)

    # Step 3: Load training data and extract features
    # We set use_thinning=False to use noise filtering instead
    train_data, train_labels = load_and_extract_features(args["training"], desc, use_thinning=False)
    
    # Step 4: Train and save the model
    model = train_and_save_model(train_data, train_labels)
    
    # Step 5: Evaluate the model
    evaluate_model(model, desc, args["testing"])

if __name__ == "__main__":
    main()
'''

'''
# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------
from localbinarypatterns import LocalBinaryPatterns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from imutils import paths
import argparse
import cv2
import os
import joblib
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# Step 1: Image Preprocessing (*** REVERTED TO 5x5 BLUR ***)
# -----------------------------------------------------------------
def preprocess_image(image):
    """
    Applies standard preprocessing: grayscale conversion and noise filtering.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- CHANGE: Reverted to (5, 5) as it was our best performer ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

# -----------------------------------------------------------------
# Step 2: Ridge Thinning (Skeletonization)
# -----------------------------------------------------------------
# (This function is here for completeness but not used in our main pipeline)
def thin_image(image):
    """
    Applies ridge thinning (skeletonization) to a fingerprint.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_for_skeleton = binary / 255.0
    skeleton = skeletonize(binary_for_skeleton)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    return skeleton_img

# -----------------------------------------------------------------
# Step 3: Feature Extraction
# -----------------------------------------------------------------
def load_and_extract_features(directory, lbp_descriptor, use_thinning=False):
    """
    Loops over all images in a directory, preprocesses them,
    and extracts LBP features.
    """
    data = []
    labels = []

    print(f"[INFO] Processing images from {directory}...")
    for imagePath in paths.list_images(directory):
        image = cv2.imread(imagePath)
        
        if use_thinning:
            processed_image = thin_image(image)
        else:
            processed_image = preprocess_image(image)
        
        hist = lbp_descriptor.describe(processed_image)
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(hist)
        
    return data, labels

# -----------------------------------------------------------------
# Step 4: Model Training (*** UPDATED TO 300 TREES ***)
# -----------------------------------------------------------------
def train_and_save_model(train_data, train_labels, model_path="fingerprint_model.pkl"):
    """
    Trains a tuned RandomForest model on the provided data and saves it.
    """
    print("[INFO] Training tuned RandomForestClassifier (300 estimators)...")
    
    # --- CHANGE: Increased n_estimators from 100 to 300 ---
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', 
                                   random_state=42, n_jobs=-1)
    
    model.fit(train_data, train_labels)
    
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully as '{model_path}'")
    
    return model

# -----------------------------------------------------------------
# Step 5: Model Evaluation
# -----------------------------------------------------------------
def evaluate_model(model, lbp_descriptor, test_directory):
    """
    Evaluates the trained model on the testing dataset, prints metrics,
    and saves a confusion matrix plot.
    """
    print(f"[INFO] Evaluating model on {test_directory}...")
    
    true_labels = []
    predicted_labels = []

    # Loop over the testing images one by one
    for imagePath in paths.list_images(test_directory):
        image = cv2.imread(imagePath)
        processed_image = preprocess_image(image) # Using our noise filtering
        hist = lbp_descriptor.describe(processed_image)
        prediction = model.predict(hist.reshape(1, -1))
        
        true_labels.append(imagePath.split(os.path.sep)[-2])
        predicted_labels.append(prediction[0])

    # --- 1. Calculate and Print Metrics ---
    
    print("\n--- Model Evaluation Results ---")
    
    labels_order = ["Live", "Fake"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    print(f"True Positives (Live predicted as Live): {TP}")
    print(f"False Negatives (Live predicted as Fake): {FN}")
    print(f"False Positives (Fake predicted as Live): {FP}")
    print(f"True Negatives (Fake predicted as Fake): {TN}")
    print("-" * 30)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live", zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (for 'Live'): {precision * 100:.2f}%")
    print(f"Recall (for 'Live'): {recall * 100:.2f}%")
    print("----------------------------------\n")
    
    # --- 2. Generate and Save Plot ---
    
    print(f"[INFO] Saving confusion matrix plot as 'confusion_matrix.png'...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_order, yticklabels=labels_order)
    
    plt.title('Confusion Matrix (RF Model, 5x5 Blur, 300 Trees)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig('confusion_matrix.png')
    print("✅ Plot saved successfully.")

# -----------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------
def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True,  help="path to the testing images")
    args = vars(ap.parse_args())

    # Initialize the LBP descriptor
    desc = LocalBinaryPatterns(24, 8)

    # Step 3: Load training data and extract features
    train_data, train_labels = load_and_extract_features(args["training"], desc, use_thinning=False)
    
    # Step 4: Train and save the model
    model = train_and_save_model(train_data, train_labels)
    
    # Step 5: Evaluate the model
    evaluate_model(model, desc, args["testing"])

if __name__ == "__main__":
    main()
    '''
    
# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------
from localbinarypatterns import LocalBinaryPatterns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# --- NEW IMPORT FOR HOG FEATURES ---
from skimage.feature import hog
from imutils import paths
import argparse
import cv2
import os
import joblib
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# Step 1: Image Preprocessing (5x5 blur)
# -----------------------------------------------------------------
def preprocess_image(image):
    """
    Applies standard preprocessing: grayscale conversion and noise filtering.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Using (5, 5) blur, which was our best performer
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# -----------------------------------------------------------------
# Step 2: Ridge Thinning (Not used)
# -----------------------------------------------------------------
def thin_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_for_skeleton = binary / 255.0
    skeleton = skeletonize(binary_for_skeleton)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    return skeleton_img

# -----------------------------------------------------------------
# Step 3: Feature Extraction (*** UPDATED WITH RESIZE ***)
# -----------------------------------------------------------------
def load_and_extract_features(directory, lbp_descriptor, use_thinning=False):
    """
    Loops over all images, preprocesses them,
    and extracts *combined LBP and HOG* features.
    """
    data = []
    labels = []
    
    # --- FIX: Define a standard size ---
    target_size = (128, 128) 

    print(f"[INFO] Processing images from {directory} (LBP + HOG)...")
    for imagePath in paths.list_images(directory):
        image = cv2.imread(imagePath)
        
        if use_thinning:
            processed_image = thin_image(image)
        else:
            processed_image = preprocess_image(image)
        
        # --- FIX: Resize the image to a standard size ---
        # This ensures all HOG vectors have the same length.
        processed_image = cv2.resize(processed_image, target_size)

        # --- 1. Extract LBP Features ---
        hist_lbp = lbp_descriptor.describe(processed_image)
        
        # --- 2. Extract HOG Features ---
        (hist_hog, hog_image) = hog(processed_image, orientations=8, 
                                   pixels_per_cell=(16, 16),
                                   cells_per_block=(1, 1), 
                                   visualize=True, 
                                   block_norm='L2-Hys')

        # --- 3. Combine Features ---
        combined_features = np.hstack([hist_lbp, hist_hog])
        
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(combined_features)
        
    return data, labels

# -----------------------------------------------------------------
# Step 4: Model Training (Reverted to our best RF)
# -----------------------------------------------------------------
def train_and_save_model(train_data, train_labels, model_path="fingerprint_model.pkl"):
    """
    Trains our best-performing RandomForest model on the new combined features.
    """
    print("[INFO] Training RandomForestClassifier (100 estimators) on LBP+HOG...")
    
    # Using our best manual parameters: n_estimators=100
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                   random_state=42, n_jobs=-1)
    
    model.fit(train_data, train_labels)
    
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully as '{model_path}'")
    
    return model

# -----------------------------------------------------------------
# Step 5: Model Evaluation (*** UPDATED WITH RESIZE ***)
# -----------------------------------------------------------------
def evaluate_model(model, lbp_descriptor, test_directory):
    """
    Evaluates the trained model on the testing dataset.
    MUST extract features in the *exact same way* as training.
    """
    print(f"[INFO] Evaluating model on {test_directory} (LBP + HOG)...")
    
    true_labels = []
    predicted_labels = []
    
    # --- FIX: Define the same standard size ---
    target_size = (128, 128)

    for imagePath in paths.list_images(test_directory):
        image = cv2.imread(imagePath)
        processed_image = preprocess_image(image) # Using our noise filtering
        
        # --- FIX: Resize the image to the same standard size ---
        processed_image = cv2.resize(processed_image, target_size)
        
        # --- 1. Extract LBP Features ---
        hist_lbp = lbp_descriptor.describe(processed_image)
        
        # --- 2. Extract HOG Features ---
        (hist_hog, hog_image) = hog(processed_image, orientations=8, 
                                   pixels_per_cell=(16, 16),
                                   cells_per_block=(1, 1), 
                                   visualize=True, 
                                   block_norm='L2-Hys')

        # --- 3. Combine Features ---
        combined_features = np.hstack([hist_lbp, hist_hog])
        
        # --- 4. Predict on combined features ---
        prediction = model.predict(combined_features.reshape(1, -1))
        
        true_labels.append(imagePath.split(os.path.sep)[-2])
        predicted_labels.append(prediction[0])

    # --- Print Metrics ---
    
    print("\n--- Model Evaluation Results (LBP + HOG) ---")
    
    labels_order = ["Live", "Fake"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    print(f"True Positives (Live predicted as Live): {TP}")
    print(f"False Negatives (Live predicted as Fake): {FN}")
    print(f"False Positives (Fake predicted as Live): {FP}")
    print(f"True Negatives (Fake predicted as Fake): {TN}")
    print("-" * 30)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, labels=labels_order, pos_label="Live", zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (for 'Live'): {precision * 100:.2f}%")
    print(f"Recall (for 'Live'): {recall * 100:.2f}%")
    print("----------------------------------\n")
    
    # --- Generate and Save Plot ---
    
    print(f"[INFO] Saving confusion matrix plot as 'confusion_matrix.png'...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_order, yticklabels=labels_order)
    
    plt.title('Confusion Matrix (RF Model, LBP + HOG)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig('confusion_matrix.png')
    print("✅ Plot saved successfully.")

# -----------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True,  help="path to the testing images")
    args = vars(ap.parse_args())

    desc = LocalBinaryPatterns(24, 8)

    train_data, train_labels = load_and_extract_features(args["training"], desc, use_thinning=False)
    
    model = train_and_save_model(train_data, train_labels)
    
    evaluate_model(model, desc, args["testing"])

if __name__ == "__main__":
    main()
