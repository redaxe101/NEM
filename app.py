from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = None  # defer loading


@app.route("/")
def index():
    return "TF model is live!"


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "transformer_model.keras")
        model = tf.keras.models.load_model(model_path)

    data = request.get_json()
    inputs = np.array(data.get("inputs"))
    preds = model.predict(inputs).tolist()
    return jsonify({"predictions": preds})
