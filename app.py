from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
import gdown
import pickle

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- APP INIT ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ---------------- GLOBALS ----------------
model = None
label_encoder = None
prediction_history = []

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure these folders exist or gdown has permission to create them
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "models", "final_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "scripts", "models", "label_encoder.pkl")

# ---------------- DOWNLOAD & LOAD LOGIC ----------------
FILE_ID = "1kfNWW10MOqLnm5awhJvIJKB3E_V3bt7a"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        logger.info("Downloading model from Google Drive...")
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")

def load_model_fn():
    global model, label_encoder
    try:
        download_model()
        
        if os.path.exists(MODEL_PATH):
            logger.info("Loading Keras model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully")
        else:
            logger.error("Model file not found after download attempt.")

        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully")
            
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")

# ---------------- PRE-START TRIGGER ----------------
# This runs when Gunicorn imports the script on Render
load_model_fn()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_file):
    try:
        img = Image.open(BytesIO(image_file.read()))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise e

# ---------------- VALIDATION ----------------
def validate_image_file(file):
    if not file or file.filename == "":
        return False, "No file selected"
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed:
        return False, "Invalid file format"
    return True, "Valid file"

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded on server"}), 500

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        is_valid, msg = validate_image_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": msg}), 400

        img_array = preprocess_image(file)
        prediction = model.predict(img_array)

        class_names = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

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
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ---------------- START LOCAL ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)