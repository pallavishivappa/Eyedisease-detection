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
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "models", "final_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "scripts", "models", "label_encoder.pkl")

# ---------------- LOAD MODEL ----------------
FILE_ID = "1kfNWW10MOqLnm5awhJvIJKB3E_V3bt7a"

# ---------------- DOWNLOAD MODEL ----------------
def download_model():

    # Download only if model is missing
    if not os.path.exists(MODEL_PATH):

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        print("Downloading model...")

        gdown.download(
            url,
            MODEL_PATH,
            quiet=False
        )

        print("Model downloaded successfully")

# ---------------- LOAD MODEL ----------------
def load_model_fn():
    global model, label_encoder

    try:

        # Download model if missing
        download_model()

        # Load model
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # Load label encoder
        if os.path.exists(ENCODER_PATH):

            with open(ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)

        logger.info(
            "Model and encoder loaded successfully"
        )

    except Exception as e:

        logger.error(
            f"Model loading error: {str(e)}"
        )

        raise e

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
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]

        is_valid, msg = validate_image_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": msg}), 400

        img_array = preprocess_image(file)

        prediction = model.predict(img_array)

        class_names = [
            "Cataract",
            "Diabetic Retinopathy",
            "Glaucoma",
            "Normal"
        ]

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        probabilities = {
            class_names[i]: float(prediction[0][i])
            for i in range(len(class_names))
        }

        # history
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
        logger.error(str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/history")
def history():
    return jsonify(prediction_history)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ---------------- ERROR HANDLERS ----------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 16MB)"}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

# ---------------- START APP ----------------
def create_app():
    load_model_fn()
    return app

if __name__ == "__main__":
    app = create_app()

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )