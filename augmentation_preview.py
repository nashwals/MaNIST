# ============================================================
# DATA AUGMENTATION PREVIEW CODE
# Copy kode ini ke dalam cell di notebook mnist-cnn.ipynb
# ============================================================

# Import yang diperlukan (pastikan sudah ada di cell sebelumnya)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Setup Data Augmentation ---
print("="*60)
print("SETUP DATA AUGMENTATION")
print("="*60)

# Konfigurasi ImageDataGenerator
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

print("✓ Data augmentation generator telah dikonfigurasi")
print("\nParameter augmentasi:")
print(f"  • Rotation range: ±10°")
print(f"  • Width shift: ±10%")
print(f"  • Height shift: ±10%")
print(f"  • Zoom range: ±10%")
print(f"  • Shear range: 10%")

# --- Preview Augmentation: Single Image ---
print("\n" + "="*60)
print("PREVIEW 1: Single Image Augmentation")
print("="*60)

# Pilih satu gambar untuk demo
sample_idx = np.random.randint(0, len(x_train))
sample_image = x_train[sample_idx].reshape(1, 28, 28, 1)
sample_label = y_train[sample_idx]

# Buat visualisasi
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Gambar pertama adalah original
axes[0].imshow(x_train[sample_idx].reshape(28, 28), cmap='gray')
axes[0].set_title(f'ORIGINAL\nLabel: {sample_label}', fontweight='bold', fontsize=12, color='blue')
axes[0].axis('off')
axes[0].spines['top'].set_visible(True)
axes[0].spines['right'].set_visible(True)
axes[0].spines['bottom'].set_visible(True)
axes[0].spines['left'].set_visible(True)
for spine in axes[0].spines.values():
    spine.set_edgecolor('blue')
    spine.set_linewidth(3)

# Generate 9 augmented versions
aug_iter = datagen.flow(sample_image, batch_size=1)

for i in range(1, 10):
    augmented_image = next(aug_iter)[0]
    axes[i].imshow(augmented_image.reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Augmented {i}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Data Augmentation: 1 Original Image → 9 Augmented Versions', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# --- Preview Augmentation: Multiple Images ---
print("\n" + "="*60)
print("PREVIEW 2: Multiple Images Comparison")
print("="*60)

num_samples = 5
num_augmentations = 4

fig, axes = plt.subplots(num_samples, num_augmentations + 1, 
                         figsize=(16, 3 * num_samples))

# Pilih random samples dari berbagai digit
sample_indices = np.random.choice(len(x_train), num_samples, replace=False)

for row, idx in enumerate(sample_indices):
    # Kolom 1: Gambar Original
    axes[row, 0].imshow(x_train[idx].reshape(28, 28), cmap='gray')
    axes[row, 0].set_title(f'Original\nDigit: {y_train[idx]}', 
                           fontweight='bold', fontsize=11, color='darkblue')
    axes[row, 0].axis('off')
    
    # Tambahkan border pada gambar original
    for spine in axes[row, 0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('darkblue')
        spine.set_linewidth(2)
    
    # Kolom 2-5: Augmented versions
    sample_image = x_train[idx].reshape(1, 28, 28, 1)
    aug_iter = datagen.flow(sample_image, batch_size=1)
    
    for col in range(1, num_augmentations + 1):
        augmented_image = next(aug_iter)[0]
        axes[row, col].imshow(augmented_image.reshape(28, 28), cmap='gray')
        axes[row, col].set_title(f'Aug {col}', fontsize=10)
        axes[row, col].axis('off')

plt.suptitle('Data Augmentation Comparison: Original vs Augmented Images', 
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# --- Preview Augmentation: Batch Visualization ---
print("\n" + "="*60)
print("PREVIEW 3: Augmentation Effects Showcase")
print("="*60)

# Pilih satu digit untuk menunjukkan variasi augmentasi
digit_to_show = 7
digit_indices = np.where(y_train == digit_to_show)[0]
sample_idx = np.random.choice(digit_indices)
sample_image = x_train[sample_idx].reshape(1, 28, 28, 1)

fig, axes = plt.subplots(3, 6, figsize=(15, 8))
axes = axes.ravel()

# Original di tengah
axes[0].imshow(x_train[sample_idx].reshape(28, 28), cmap='gray')
axes[0].set_title(f'ORIGINAL\nDigit: {digit_to_show}', 
                  fontweight='bold', fontsize=12, color='red')
axes[0].axis('off')
for spine in axes[0].spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('red')
    spine.set_linewidth(3)

# Generate 17 augmented versions
aug_iter = datagen.flow(sample_image, batch_size=1)

for i in range(1, 18):
    augmented_image = next(aug_iter)[0]
    axes[i].imshow(augmented_image.reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Aug {i}', fontsize=9)
    axes[i].axis('off')

plt.suptitle(f'Augmentation Showcase: Digit "{digit_to_show}" with 17 Variations', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# --- Summary ---
print("\n" + "="*60)
print("AUGMENTATION SUMMARY")
print("="*60)
print(f"✓ Original training samples: {len(x_train):,}")
print(f"✓ Dengan augmentation, setiap epoch akan melihat variasi berbeda")
print(f"✓ Ini membantu model generalisasi lebih baik dan mengurangi overfitting")
print("="*60)
