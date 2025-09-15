import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Preprocessing Function
# -----------------------------
def preprocess_image(img_path, img_size=(128,128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    edges = cv2.Canny(img, 100, 200)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    features, _ = hog(img, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(2,2), visualize=True)
    return img, edges, thresh, features

# -----------------------------
# 2. CNN Model
# -----------------------------
def build_cnn(input_shape=(128,128,1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# 3. Load Dataset
# -----------------------------
def load_dataset(dataset_path="data/", img_size=(128,128)):
    X, y = [], []
    for label, folder in enumerate(["flood","no_flood"]):
        folder_path = os.path.join(dataset_path, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img, _, _, _ = preprocess_image(img_path, img_size)
                X.append(img)
                y.append(label)
            except:
                continue
    X = np.array(X).reshape(-1,img_size[0],img_size[1],1)
    y = np.array(y)
    return X, y

# -----------------------------
# 4. Train Model
# -----------------------------
def train_model():
    X, y = load_dataset("data/")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, horizontal_flip=True)
    model = build_cnn(input_shape=(128,128,1))
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_test/255.0, y_test),
              epochs=10)
    model.save("flood_detection_model.h5")
    print("✅ Model saved as flood_detection_model.h5")

# -----------------------------
# 5. Evaluate Model
# -----------------------------
def evaluate_model():
    model = load_model("flood_detection_model.h5")
    X, y = load_dataset("data/")
    _, acc = model.evaluate(X/255.0, y)
    print(f"📊 Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# Run Everything
# -----------------------------
if __name__ == "__main__":
    train_model()
    evaluate_model()
