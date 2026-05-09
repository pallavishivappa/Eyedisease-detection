from tensorflow.keras.models import load_model

model = load_model("scripts/models/best_cnn_model.keras")

print("Model loaded successfully")