# --- Import library ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler


# --- Load dataset MNIST (sudah termasuk train dan test split) ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Visualisasi beberapa contoh gambar
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()


# --- Preprocessing ---

# Reshape untuk CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalisasi pixel values ke range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding untuk labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"\nSetelah pre-processing:")
print(f"x_train shape: {x_train.shape}")
print(f"y_train_cat shape: {y_train_cat.shape}")

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=10,        # Rotasi random hingga 10 derajat
    width_shift_range=0.1,    # Shift horizontal hingga 10%
    height_shift_range=0.1,   # Shift vertikal hingga 10%
    zoom_range=0.1,           # Zoom in/out hingga 10%
    shear_range=0.1,          # Shear transformation
    fill_mode='nearest'       # Cara mengisi pixel yang kosong
) 

# Fit generator pada training data
datagen.fit(x_train)


# --- Build CNN Model ---
model = keras.Sequential([
    # Convolutional Layer 1
    layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)),

    # Convolutional Layer 2
    layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    
    # Convolutional Layer 3
    layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
    layers.Dropout(0.3),
    
    # Convolutional Layer 4
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),

    # Convolutional Layer 5
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),  # Hanya 1 pooling lagi
    layers.Dropout(0.4),
    
    # Layer 6
    layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
    
    # Fully-Connected Layer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary model
model.summary()


# --- Training ---
# Decrease Learning Rate Each Epoch
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# Callback untuk early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Training
history = model.fit(
    x_train, y_train_cat,
    batch_size=64,
    epochs=45,
    validation_split=0.1,
    callbacks=[early_stop, annealer],
    verbose=1
)


# --- Evaluasi ---
# Evaluasi pada test set
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Plot history training
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# --- Prediksi dan Visualisasi ---
# Prediksi
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Visualisasi prediksi
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.ravel()

# Pilih 20 sample random
indices = np.random.choice(len(x_test), 20, replace=False)

for i, idx in enumerate(indices):
    axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
    pred_label = y_pred_classes[idx]
    true_label = y_test[idx]
    
    # Warna hijau jika benar, merah jika salah
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))


# --- Save Model ---
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")

# Save as Keras format
model.save('mnist_cnn_model.keras')
print("Model saved as 'mnist_cnn_model.keras'")

print("\n" + "-"*50)
print("MNIST Digit Recognition - COMPLETED!")
print("="*50)
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")