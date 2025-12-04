from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import io
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load model baru (pastikan file .h5 atau .keras hasil training baru sudah ada di folder model)
try:
    model = tf.keras.models.load_model('model/mnist_cnn_model.h5')
    print("Model loaded successfully.")
except:
    # Fallback jika nama file berbeda
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.keras')
        print("Model (.keras) loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def get_model_summary_str(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

if model:
    MODEL_SUMMARY_TEXT = get_model_summary_str(model)
else:
    MODEL_SUMMARY_TEXT = "Model not loaded."

def encode_image_array(img_array, resize_factor=4):
    # Normalisasi untuk visualisasi yang enak dilihat
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    # Rescale ke 0-255 untuk jadi image
    if img_array.max() != img_array.min():
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    else:
        img_array = img_array * 255 
        
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    
    w, h = img.size
    # Resize agar tidak terlalu kecil di layar
    img = img.resize((w * resize_factor, h * resize_factor), Image.NEAREST)
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # 1. Preprocessing Input Gambar
        img = Image.open(BytesIO(image_bytes)).convert('L')
        img = ImageOps.invert(img) # Invert karena canvas putih, training data hitam
        img = img.resize((28, 28))
        
        img_array = np.array(img).astype('float32') / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)
        
        # 2. Prediksi Akhir
        prediction = model.predict(img_input)
        predicted_digit = int(np.argmax(prediction))
        probabilities = prediction[0].tolist()

        # 3. Ekstraksi Visualisasi Layer per Layer
        viz_data = {}
        logs = [] 
        
        # Simpan visualisasi input
        viz_data['input'] = encode_image_array(img_array, resize_factor=5)
        logs.append({
            "step": "Input Layer",
            "info": f"Shape: {img_input.shape}",
            "type": "input"
        })

        # Kita gunakan tensor untuk jalan manual melewati setiap layer
        current_tensor = tf.convert_to_tensor(img_input)

        for i, layer in enumerate(model.layers):
            input_shape_str = str(current_tensor.shape)
            
            # Jalankan layer
            current_tensor = layer(current_tensor)
            output_shape_str = str(current_tensor.shape)
            
            layer_info = {
                "id": i,
                "name": layer.name,
                "type": layer.__class__.__name__,
                "input_shape": input_shape_str,
                "output_shape": output_shape_str,
                "params": layer.count_params()
            }

            # Ambil nilai output sebagai numpy array
            layer_output = current_tensor.numpy()
            
            # LOGIKA BARU: Deteksi tipe visualisasi berdasarkan BENTUK DATA (SHAPE)
            # Bukan berdasarkan nama layer. Ini lebih aman untuk arsitektur kompleks.
            
            # Cek dimensi: (Batch, Height, Width, Channels) -> Rank 4
            if len(layer_output.shape) == 4:
                # Ini adalah data GAMBAR (Conv2D, MaxPooling, atau Dropout pada image)
                n_features = layer_output.shape[-1]
                images_in_layer = []
                
                # Ambil maksimal 16 filter pertama saja agar tidak berat
                max_filters_to_show = 16 
                for idx in range(min(n_features, max_filters_to_show)): 
                    img_feature = layer_output[0, :, :, idx]
                    images_in_layer.append(encode_image_array(img_feature))
                
                viz_data[layer.name] = {
                    'type': 'image',
                    'data': images_in_layer
                }
                layer_info['has_viz'] = True

            # Cek dimensi: (Batch, Features) -> Rank 2
            elif len(layer_output.shape) == 2:
                # Ini adalah data VEKTOR (Flatten, Dense, Dropout pada vector)
                flat_array = layer_output.flatten()
                
                # Ambil sampel data (maks 100) untuk barcode
                sample_data = flat_array[:100].tolist() 
                
                viz_data[layer.name] = {
                    'type': 'vector',
                    'data': sample_data,
                    'min': float(flat_array.min()),
                    'max': float(flat_array.max()),
                    'total_elements': len(flat_array)
                }
                layer_info['has_viz'] = True
            
            else:
                # Layer lain yang mungkin bentuknya aneh (jarang terjadi di CNN standar)
                layer_info['has_viz'] = False

            logs.append(layer_info)

        return jsonify({
            'digit': predicted_digit,
            'probabilities': probabilities,
            'visualization': viz_data,
            'logs': logs,
            'model_summary': MODEL_SUMMARY_TEXT,
            'status': 'success'
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)