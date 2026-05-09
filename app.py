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

# Flask application initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variable for model
model = None

# HISTORY STORAGE
prediction_history = []

# LOAD MODEL
# LOAD MODEL
def load_model():
    """Load the trained model with error handling"""

    global model

    try:

        import gdown

        model_path = 'best_cnn_model.keras'

        # Download model if not present
        if not os.path.exists(model_path):

            file_id = "1RrAAdhHZlnGipzHuVBIL38zPBtdosGG6"

            url = f"https://drive.google.com/uc?id={file_id}"

            logger.info("Downloading model from Google Drive...")

            gdown.download(url, model_path, quiet=False)

        # Load model
        model = tf.keras.models.load_model(model_path)

        logger.info("Model loaded successfully")

    except Exception as e:

        logger.error(f"Error loading model: {str(e)}")

        raise e


# PREPROCESS IMAGE
def preprocess_image(image_file):
    """Preprocess the uploaded image for model prediction"""

    try:
        img = Image.open(BytesIO(image_file.read()))

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image
        img = img.resize((128, 128))

        # Convert to array
        img_array = img_to_array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e


# VALIDATE IMAGE
def validate_image_file(file):

    if not file:
        return False, "No file provided"

    if file.filename == '':
        return False, "No file selected"

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in allowed_extensions:
        return False, "Invalid file format. Upload PNG, JPG, JPEG, BMP, or TIFF"

    return True, "Valid file"


# HOME ROUTE
@app.route('/')
def index():
    return render_template('index.html')


# PREDICTION ROUTE
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Check model
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500

        # Check file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']

        # Validate file
        is_valid, message = validate_image_file(file)

        if not is_valid:
            return jsonify({
                'success': False,
                'error': message
            }), 400

        # Preprocess image
        img_array = preprocess_image(file)

        # Prediction
        prediction = model.predict(img_array)

        predicted_class_index = np.argmax(prediction, axis=1)[0]

        confidence = float(np.max(prediction))

        # CLASS NAMES
        class_names = [
            'Cataract',
            'Diabetic Retinopathy',
            'Glaucoma',
            'Normal'
        ]

        predicted_class = class_names[predicted_class_index]

        # ALL PROBABILITIES
        class_probabilities = {}

        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(prediction[0][i])

        logger.info(
            f"Prediction successful: {predicted_class} "
            f"(confidence: {confidence:.2f})"
        )

        # SAVE HISTORY
        history_item = {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": confidence
        }

        prediction_history.append(history_item)

        # RETURN RESPONSE
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        })

    except Exception as e:

        logger.error(f"Prediction error: {str(e)}")

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# HISTORY ROUTE
@app.route('/history')
def history():
    return jsonify(prediction_history)


# HEALTH CHECK
@app.route('/health')
def health_check():

    try:

        model_status = "loaded" if model is not None else "not loaded"

        return jsonify({
            'status': 'healthy',
            'model_status': model_status
        })

    except Exception as e:

        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# FILE TOO LARGE
@app.errorhandler(413)
def too_large(e):

    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


# 404 ERROR
@app.errorhandler(404)
def not_found(e):

    return render_template('index.html'), 404


# 500 ERROR
@app.errorhandler(500)
def internal_error(e):

    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# CREATE APP
def create_app():

    try:

        load_model()

        logger.info("Application initialized successfully")

        return app

    except Exception as e:

        logger.error(f"Failed to initialize application: {str(e)}")

        raise e


# MAIN
if __name__ == "__main__":

    try:

        app = create_app()

        app.run(
            debug=False,
            host='0.0.0.0',
            port=7860,
            threaded=True
        )

    except Exception as e:

        logger.error(f"Failed to start application: {str(e)}")

        print(f"Error: {str(e)}")

        print("Please ensure:")
        print("1. Model file exists")
        print("2. Templates folder exists")
        print("3. Required packages are installed")
