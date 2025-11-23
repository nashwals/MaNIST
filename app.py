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

model = tf.keras.models.load_model('model/mnist_cnn_model.h5')

def get_model_summary_str(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

MODEL_SUMMARY_TEXT = get_model_summary_str(model)

def encode_image_array(img_array, resize_factor=4):
    if img_array.max() != img_array.min():
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    else:
        img_array = img_array * 255 
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    w, h = img.size
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
        
        img = Image.open(BytesIO(image_bytes)).convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img_input)
        predicted_digit = int(np.argmax(prediction))
        probabilities = prediction[0].tolist()

        viz_data = {}
        logs = [] 
        
        viz_data['input'] = encode_image_array(img_array, resize_factor=5)
        logs.append({
            "step": "Input Layer",
            "info": f"Menerima Gambar: Shape {img_input.shape}",
            "type": "input"
        })

        current_tensor = tf.convert_to_tensor(img_input)

        for i, layer in enumerate(model.layers):
            input_shape_str = str(current_tensor.shape)
            
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

            layer_name = layer.name.lower()
            if 'conv' in layer_name or 'pool' in layer_name:
                layer_output = current_tensor.numpy()
                n_features = layer_output.shape[-1]
                images_in_layer = []
                for idx in range(min(n_features, 8)): 
                    img_feature = layer_output[0, :, :, idx]
                    images_in_layer.append(encode_image_array(img_feature))
                
                viz_data[layer.name] = images_in_layer
                layer_info['has_viz'] = True
            else:
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