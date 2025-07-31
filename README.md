import os
print("Files in current directory:", os.listdir("."))

os.makedirs("/mnt/data/", exist_ok=True)
print("Directory created successfully.")

zip_path = "/content/archive.zip"

import os
print("Files in content directory:", os.listdir("/content/"))

import zipfile
import os
import os

for item in os.listdir(extract_path):
    item_path = os.path.join(extract_path, item)
    if os.path.isdir(item_path):
        print(f"Contents of folder {item}:", os.listdir(item_path))
    else:
        print(f"File found: {item}")
zip_path = "/content/archive.zip"  # Update this if your file path is different
extract_path = "/content/dataset"
# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
# List extracted files
print("Extracted files and folders:", os.listdir(extract_path))

import os
import cv2
import numpy as np
import pandas as pd
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths
extract_path = "/content/dataset"
train_csv = os.path.join(extract_path, "train.csv")
train_folder = os.path.join(extract_path, "train")

# Load train.csv
df = pd.read_csv(train_csv)

# Load images & labels
def load_images_labels(folder, dataframe):
    images, labels = [], []
    for index, row in dataframe.iterrows():
        img_path = os.path.join(folder, row["Image ID"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
        img = cv2.resize(img, (128, 128))  # Resize
        img = img / 255.0  # Normalize
        edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)  # Canny Edge Detection
        images.append(edges)
        labels.append(row["Flooded"])  # Target variable
    return np.array(images).reshape(-1, 128, 128, 1), np.array(labels)

X, y = load_images_labels(train_folder, df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Improved CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),  # Prevent Overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile Model with Lower Learning Rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
# Train model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
# Evaluate & Print Accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# Save model
model.save("/content/flood_detection_model.h5")
print("Model training complete. Saved as flood_detection_model.h5")

import os
import cv2
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import hog

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load trained model without compiling to suppress warnings
model = load_model("/content/flood_detection_model.h5", compile=False)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize
    img = img / 255.0  # Normalize
    return img

# Apply segmentation using adaptive thresholding
def segment_image(img):
    segmented = cv2.adaptiveThreshold((img * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return segmented

# Extract features using edge detection
def extract_features(img):
    edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    return edges

# Feature selection using Histogram of Oriented Gradients (HOG)
def select_features(img):
    features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_image

# Perform result analysis & visualization
def analyze_result(img_path):
    original_img = cv2.imread(img_path)  # Load original image
    img = preprocess_image(img_path)
    segmented = segment_image(img)
    features = extract_features(img)
    selected_features = select_features(features)

    # Model prediction
    img_input = features.reshape(1, 128, 128, 1)
    pred = model.predict(img_input)[0][0]
    prediction = "Flooded" if pred > 0.5 else "Normal"

    # Decision Making
    if pred > 0.75:
        decision = "High risk of flooding"
        risk_level = "high"
    elif pred > 0.5:
        decision = "Moderate risk of flooding"
        risk_level = "moderate"
    else:
        decision = "Low risk of flooding"
        risk_level = "low"

    return (original_img, img, segmented, features, selected_features, decision, risk_level, img_path, pred)

# Process all test images and store their results
test_folder = "/content/dataset/test"
test_images = os.listdir(test_folder)
results = []

total_images = len(test_images)
flooded_count = 0

for img_name in test_images:
    image_path = os.path.join(test_folder, img_name)
    result = analyze_result(image_path)
    results.append(result)
    if result[6] != "low":
        flooded_count += 1

# Sort images by prediction score to ensure variety
results.sort(key=lambda x: x[8])  # Sort by prediction score

# Select images with required categories
low_risk = [res for res in results if res[6] == "low"][:10]  # Pick first 3 low risk
moderate_risk = [res for res in results if res[6] == "moderate"][:20]  # Pick first 2 moderate risk

# Ensure at least 2 moderate-risk images
if len(moderate_risk) < 2:
    moderate_risk.extend(results[len(results) // 2 : len(results) // 2 + (2 - len(moderate_risk))])

# Combine selected images
selected_images = low_risk + moderate_risk

# Display selected images
for res in selected_images:
    original_img, img, segmented, features, selected_features, decision, risk_level, img_path, pred = res
    print(f"Processing: {img_path}, Decision: {decision}")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.subplot(1, 6, 2)
    plt.imshow(img, cmap='gray')
    plt.title("Preprocessed Image")
    plt.subplot(1, 6, 3)
    plt.imshow(segmented, cmap='gray')
    plt.title("Segmented Image")
    plt.subplot(1, 6, 4)
    plt.imshow(features, cmap='gray')
    plt.title("Feature Extraction")
    plt.subplot(1, 6, 5)
    plt.imshow(selected_features, cmap='gray')
    plt.title("Feature Selection")
    plt.subplot(1, 6, 6)
    plt.text(0.5, 0.5, decision, fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))
    plt.axis("off")
    plt.title("Decision")
    plt.show()
