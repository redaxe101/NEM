from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import joblib
import threading
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

latest_prediction = None
latest_timestamp = None

feature_scaler = joblib.load(
    os.path.join(os.path.dirname(__file__), "feature_scaler.pkl")
)
all_feature_cols = feature_scaler.feature_names_in_.tolist()

app = Flask(__name__)
# Load model at startup instead of using before_first_request
model_path = os.path.join(os.path.dirname(__file__), "transformer_model.keras")
model = tf.keras.models.load_model(model_path)


def build_encoder_input_from_aemo():
    # Fetch latest NEM/AEMO data
    # Preprocess into shape (1, 48, 14)
    # Return as np.float32

    # === Dummy encoder input: (1, 48, 14) ===
    encoder_input_raw = np.random.rand(1, 48, 14)

    # Scale each 14-feature row by embedding into a 49-feature dummy row
    encoder_scaled = []
    num_features = feature_scaler.n_features_in_
    for row in encoder_input_raw[0]:
        full_row = np.zeros((num_features,))
        full_row[:14] = row  # Assuming encoder features are first 14
        full_row_scaled = feature_scaler.transform(full_row.reshape(1, -1))[0]
        encoder_scaled.append(full_row_scaled[:14])  # take back only 14 features
    encoder_input = np.expand_dims(np.array(encoder_scaled), axis=0).astype(np.float32)
    return encoder_input


def build_decoder_input_from_aemo():
    # Fetch forecast/predisp data
    # Preprocess into shape (1, 32, 35)
    # Return as np.float32
    num_features = feature_scaler.n_features_in_
    # === Dummy decoder input: (1, 32, 35) ===
    decoder_input_raw = np.random.rand(1, 32, 35)

    decoder_scaled = []
    for row in decoder_input_raw[0]:
        full_row = np.zeros((num_features,))
        full_row[:35] = row  # Assuming decoder features are first 35
        full_row_scaled = feature_scaler.transform(full_row.reshape(1, -1))[0]
        decoder_scaled.append(full_row_scaled[:35])  # take back only 35 features
    decoder_input = np.expand_dims(np.array(decoder_scaled), axis=0).astype(np.float32)
    return decoder_input


def fetch_and_predict_loop():
    global latest_prediction, latest_timestamp

    while True:
        try:
            print("🔁 Fetching AEMO data and running prediction...")

            # === Replace this with real AEMO fetching and input generation ===
            encoder_input = build_encoder_input_from_aemo()  # shape: (1, 48, 14)
            decoder_input = build_decoder_input_from_aemo()  # shape: (1, 32, 35)

            preds_scaled = model.predict(
                [encoder_input, decoder_input]
            )  # shape: (1, 32, 1)
            preds_scaled = preds_scaled.reshape(-1)

            # === Unscale predictions ===
            num_features = feature_scaler.n_features_in_
            rrp_index = all_feature_cols.index("RRP")
            X_dummy = np.zeros((32, num_features))
            X_dummy[:, rrp_index] = preds_scaled
            preds_unscaled = feature_scaler.inverse_transform(X_dummy)[
                :, rrp_index
            ].tolist()

            # Save result
            latest_prediction = preds_unscaled
            latest_timestamp = datetime.now()

            print("✅ Prediction updated at", latest_timestamp)

        except Exception as e:
            print("❌ Error in prediction loop:", e)

        time.sleep(1800)


@app.route("/")
def index():
    return "TF model is live!"


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/predict", methods=["GET"])
def predict():
    if latest_prediction is None:
        return jsonify({"error": "Prediction not ready yet"}), 503

    return jsonify(
        {
            "timestamp": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": latest_prediction,
        }
    )


@app.route("/predictold", methods=["GET"])
def predictold():
    global model
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        num_features = feature_scaler.n_features_in_  # 49
        rrp_index = all_feature_cols.index("RRP")
        output_steps = 32

        # === Dummy encoder input: (1, 48, 14) ===
        encoder_input_raw = np.random.rand(1, 48, 14)

        # Scale each 14-feature row by embedding into a 49-feature dummy row
        encoder_scaled = []
        for row in encoder_input_raw[0]:
            full_row = np.zeros((num_features,))
            full_row[:14] = row  # Assuming encoder features are first 14
            full_row_scaled = feature_scaler.transform(full_row.reshape(1, -1))[0]
            encoder_scaled.append(full_row_scaled[:14])  # take back only 14 features
        encoder_input = np.expand_dims(np.array(encoder_scaled), axis=0).astype(
            np.float32
        )

        # === Dummy decoder input: (1, 32, 35) ===
        decoder_input_raw = np.random.rand(1, 32, 35)

        decoder_scaled = []
        for row in decoder_input_raw[0]:
            full_row = np.zeros((num_features,))
            full_row[:35] = row  # Assuming decoder features are first 35
            full_row_scaled = feature_scaler.transform(full_row.reshape(1, -1))[0]
            decoder_scaled.append(full_row_scaled[:35])  # take back only 35 features
        decoder_input = np.expand_dims(np.array(decoder_scaled), axis=0).astype(
            np.float32
        )

        # === Predict ===
        preds_scaled = model.predict(
            [encoder_input, decoder_input]
        )  # shape: (1, 32, 1)
        preds_scaled = preds_scaled.reshape(-1)  # (32,)

        # === Unscale predictions using dummy full-feature rows ===
        X_dummy = np.zeros((output_steps, num_features), dtype=np.float32)
        X_dummy[:, rrp_index] = preds_scaled
        preds_unscaled = feature_scaler.inverse_transform(X_dummy)[
            :, rrp_index
        ].tolist()

        return jsonify({"predictions": preds_unscaled})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


threading.Thread(target=fetch_and_predict_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
