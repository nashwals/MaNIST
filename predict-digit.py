"""
Prediksi Digit dari Gambar Custom
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import os

# ===========================
# FUNGSI PREPROCESSING
# ===========================

def preprocess_image(image_path):
    """
    Preprocessing gambar untuk prediksi
        
    Returns:
        Gambar yang sudah dipreprocess dalam format yang siap untuk model
    """
    # Load gambar
    img = Image.open(image_path)
    
    # Konversi ke grayscale jika gambar berwarna
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize ke 28x28 (ukuran MNIST)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert ke numpy array
    img_array = np.array(img)
    
    # MNIST menggunakan background hitam, digit putih
    # Jika gambar Anda sebaliknya (background putih, digit hitam), invert:
    # Cek apakah perlu di-invert (jika rata-rata pixel > 127, kemungkinan background putih)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # Invert
    
    # Normalisasi ke [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape untuk CNN: (1, 28, 28, 1)
    # 1 = batch size 1 gambar
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, img


def predict_digit(model, image_path, show_plot=True):
    """
    Prediksi digit dari gambar
    
    Args:
        model: Model Keras yang sudah di-load
        image_path: Path ke file gambar
        show_plot: Apakah menampilkan plot (True/False)
        
    Returns:
        predicted_digit: Digit yang diprediksi (0-9)
        confidence: Confidence score (0-1)
    """
    # Preprocessing
    img_processed, img_original = preprocess_image(image_path)
    
    # Prediksi
    prediction = model.predict(img_processed, verbose=0)
    
    # Ambil class dengan probability tertinggi
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit]
    
    # Visualisasi
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Gambar original
        axes[0].imshow(img_original, cmap='gray')
        axes[0].set_title('Gambar Original')
        axes[0].axis('off')
        
        # Plot 2: Gambar setelah preprocessing (28x28)
        axes[1].imshow(img_processed.reshape(28, 28), cmap='gray')
        axes[1].set_title('Setelah Preprocessing (28x28)')
        axes[1].axis('off')
        
        # Plot 3: Bar chart probability semua digit
        axes[2].bar(range(10), prediction[0], color='skyblue')
        axes[2].axhline(y=confidence, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Digit')
        axes[2].set_ylabel('Probability')
        axes[2].set_title(f'Prediksi: {predicted_digit} (Confidence: {confidence:.2%})')
        axes[2].set_xticks(range(10))
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = 'prediction_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Hasil visualisasi disimpan di: {output_path}")
        plt.show()
    
    return predicted_digit, confidence, prediction[0]


# ===========================
# MAIN PROGRAM
# ===========================

def main():
    print("="*60)
    print("MNIST Digit Recognition - Prediksi Gambar Custom")
    print("="*60)
    
    # 1. Load model yang sudah di-training
    print("\n1. Loading model...")
    model_path = 'mnist_cnn_model.h5'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model tidak ditemukan di {model_path}")
        print("Silakan train model dulu dengan menjalankan 'mnist_digit_recognition.py'")
        return
    
    model = keras.models.load_model(model_path)
    print("✓ Model berhasil di-load!")
    
    # 2. Path ke gambar yang ingin diprediksi
    print("\n2. Memproses gambar...")
    
    # GANTI PATH INI dengan path gambar Anda
    image_path = 'preprocessed_gambar.png'
    
    # Cek apakah file ada
    if not os.path.exists(image_path):
        print(f"ERROR: Gambar tidak ditemukan di {image_path}")
        print("\nCara menggunakan:")
        print("1. Upload gambar digit Anda")
        print("2. Ganti 'image_path' di script ini dengan path gambar Anda")
        print("3. Jalankan script lagi")
        return
    
    print(f"✓ Gambar ditemukan: {image_path}")
    
    # 3. Prediksi
    print("\n3. Melakukan prediksi...")
    predicted_digit, confidence, all_probs = predict_digit(model, image_path, show_plot=True)
    
    # 4. Hasil
    print("\n" + "="*60)
    print("HASIL PREDIKSI")
    print("="*60)
    print(f"Digit yang diprediksi: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nProbability semua digit:")
    for digit in range(10):
        bar = "█" * int(all_probs[digit] * 50)
        print(f"  Digit {digit}: {all_probs[digit]:.4f} {bar}")
    
    print("\n✓ Prediksi selesai!")


# ===========================
# FUNGSI BATCH PREDICTION
# ===========================

def predict_multiple_images(model, image_folder):
    """
    Prediksi multiple gambar sekaligus dari satu folder
    
    Args:
        model: Model Keras
        image_folder: Path ke folder berisi gambar-gambar
    """
    print(f"\nMemproses gambar dari folder: {image_folder}")
    
    # Daftar file gambar
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("Tidak ada gambar ditemukan di folder tersebut!")
        return
    
    print(f"Ditemukan {len(image_files)} gambar\n")
    
    results = []
    
    # Plot grid
    n_images = len(image_files)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Prediksi tanpa plot individual
            predicted_digit, confidence, _ = predict_digit(model, image_path, show_plot=False)
            
            # Simpan hasil
            results.append({
                'filename': filename,
                'predicted_digit': predicted_digit,
                'confidence': confidence
            })
            
            # Load gambar untuk visualisasi
            img = Image.open(image_path).convert('L').resize((28, 28))
            
            # Plot
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'{filename}\nPred: {predicted_digit} ({confidence:.1%})')
            axes[idx].axis('off')
            
            print(f"✓ {filename}: Predicted = {predicted_digit}, Confidence = {confidence:.2%}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Batch prediction results saved!")
    plt.show()
    
    return results


if __name__ == "__main__":
    main()
    
    # Uncomment untuk batch prediction:
    # model = keras.models.load_model('/mnt/user-data/outputs/mnist_cnn_model.keras')
    # results = predict_multiple_images(model, '/mnt/user-data/uploads/')