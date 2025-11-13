# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------
from localbinarypatterns import LocalBinaryPatterns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# --- ADDED HOG IMPORT ---
from skimage.feature import hog
from imutils import paths
import argparse
import cv2
import os
import joblib # Crucial for loading the model
import numpy as np

# Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# Step 1: Image Preprocessing (Noise Filtering)
# (Must be identical to the function used for training)
# -----------------------------------------------------------------
def preprocess_image(image):
    """
    Applies standard preprocessing: grayscale conversion and noise filtering.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Using (5, 5) blur as it was our best performer
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# -----------------------------------------------------------------
# Step 2: Model Evaluation (*** THIS IS THE FIXED FUNCTION ***)
# -----------------------------------------------------------------
def evaluate_model(model, lbp_descriptor, test_directory):
    """
    Evaluates the loaded model on the testing dataset, prints metrics,
    and saves a confusion matrix plot.
    """
    print(f"[INFO] Evaluating model on {test_directory} (LBP + HOG)...")
    
    true_labels = []
    predicted_labels = []

    # --- FIX: Define the same standard size used in training ---
    target_size = (128, 128)

    # Loop over the testing images one by one
    for imagePath in paths.list_images(test_directory):
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Warning: Could not read image {imagePath}. Skipping.")
            continue
            
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

    # --- 1. Calculate and Print Metrics ---
    
    print("\n--- Model Evaluation Results (LBP + HOG) ---")
    
    labels_order = ["Live", "Fake"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)
    
    # Handle cases where a class might not be present in predictions
    if cm.shape != (2, 2):
        print("[ERROR] Confusion matrix is not 2x2. Check your test data.")
        print("Matrix shape:", cm.shape)
        return

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
    
    plot_filename = 'test_dataset_confusion_matrix.png'
    print(f"[INFO] Saving confusion matrix plot as '{plot_filename}'...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_order, yticklabels=labels_order)
    
    plt.title('Test Dataset Confusion Matrix (RF, LBP+HOG)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig(plot_filename)
    print("✅ Plot saved successfully.")

# -----------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------
def main():
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="fingerprint_model.pkl", help="path to the trained .pkl model")
    ap.add_argument("-e", "--testing", required=True,  help="path to the testing images")
    args = vars(ap.parse_args())

    # Check if model file exists
    if not os.path.exists(args["model"]):
        print(f"[ERROR] Model file not found at {args['model']}")
        print("Please run 'recognize.py' first to train and save the model.")
        return

    # Initialize the LBP descriptor (must match training settings)
    desc = LocalBinaryPatterns(24, 8)

    # Load the trained model from disk
    print(f"[INFO] Loading pre-trained model from {args['model']}...")
    model = joblib.load(args["model"])
    
    # Evaluate the loaded model on the test dataset
    evaluate_model(model, desc, args["testing"])

if __name__ == "__main__":
    main()