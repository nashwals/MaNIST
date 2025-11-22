from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import random
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

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
        
        img = Image.open(BytesIO(image_bytes))
        
        predicted_digit = random.randint(0, 9)
        probs = np.random.random(10)
        probs[predicted_digit] += 2.0
        probs = probs / probs.sum()
        probabilities = probs.tolist()

        return jsonify({
            'digit': predicted_digit,
            'probabilities': probabilities,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)