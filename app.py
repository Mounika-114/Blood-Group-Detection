from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import json


app = Flask(__name__)

# Load Trained Model
MODEL_PATH = "blood_group_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> label
class_labels = {v: k for k, v in class_indices.items()}
# Define Class Labels (Adjust based on your dataset)
class_labels = ["A-", "A+", "B+", "B-", "O+", "O-", "AB+", "AB-"]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # UI Page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)
        pred_index = np.argmax(pred)
        predicted_class = class_labels[pred_index]
        confidence = float(np.max(pred)) * 100

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
