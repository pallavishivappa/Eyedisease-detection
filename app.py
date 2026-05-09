from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global model
model = None

# History
prediction_history = []


# ---------------- LOAD MODEL ----------------
def load_model():
    global model

    try:
        import gdown

        model_path = 'best_cnn_model.keras'

        # Download only if not exists
        if not os.path.exists(model_path):

            file_id = "1RrAAdhHZlnGipzHuVBIL38zPBtdosGG6"
            url = f"https://drive.google.com/uc?id={file_id}"

            logger.info("Downloading model...")
            gdown.download(url, model_path, quiet=False)

        # ✅ FIX FOR TF 2.20 ERROR
        model = tf.keras.models.load_model(model_path, compile=False)

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e


# ---------------- PREPROCESS ----------------
def preprocess_image(image_file):

    img = Image.open(BytesIO(image_file.read()))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((128, 128))

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------- VALIDATE ----------------
def validate_image_file(file):

    if not file:
        return False, "No file provided"

    if file.filename == '':
        return False, "No file selected"

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in allowed_extensions:
        return False, "Invalid file format"

    return True, "Valid file"


# ---------------- HOME ----------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():

    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        valid, msg = validate_image_file(file)
        if not valid:
            return jsonify({'error': msg}), 400

        img_array = preprocess_image(file)

        prediction = model.predict(img_array)

        class_names = [
            'Cataract',
            'Diabetic Retinopathy',
            'Glaucoma',
            'Normal'
        ]

        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        predicted_class = class_names[predicted_index]

        probabilities = {
            class_names[i]: float(prediction[0][i])
            for i in range(len(class_names))
        }

        prediction_history.append({
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": confidence
        })

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- HISTORY ----------------
@app.route('/history')
def history():
    return jsonify(prediction_history)


# ---------------- HEALTH ----------------
@app.route('/health')
def health_check():

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


# ---------------- ERROR HANDLING ----------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ---------------- CREATE APP ----------------
def create_app():
    load_model()
    return app


# ---------------- RUN ----------------
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)