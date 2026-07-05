import os
import traceback

import gradio as gr
import librosa
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers


# =========================================================
# PATH SETTINGS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "final_stress_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "stress_scaler.pkl")

TARGET_SR = 22050
MAX_TIME_STEPS = 174
N_MFCC = 40
FEATURE_DIM = 122
THRESHOLD = 0.5


# =========================================================
# CUSTOM ATTENTION LAYER
# =========================================================
class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, hidden_states):
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, tf.squeeze(weights, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Model and scaler loaded successfully.")


# =========================================================
# AUDIO LOADING
# =========================================================
def load_audio(audio_input):
    """
    Works with Gradio filepath audio.
    """

    if audio_input is None:
        raise ValueError("No audio received. Please record 3 to 5 seconds.")

    print("Received audio input:", audio_input)
    print("Input type:", type(audio_input))

    # Gradio 3 filepath mode usually gives string path
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")

        y, sr = librosa.load(audio_input, sr=TARGET_SR, mono=True)
        return y, sr

    # Sometimes Gradio may return tuple: (sample_rate, array)
    if isinstance(audio_input, tuple):
        sr, y = audio_input
        y = np.asarray(y)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        y = y.astype(np.float32)

        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        return y, sr

    raise TypeError(f"Unsupported audio input type: {type(audio_input)}")


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(audio_input):
    y, sr = load_audio(audio_input)

    if y is None or len(y) == 0:
        raise ValueError("Audio is empty. Please record again.")

    print("Audio loaded.")
    print("Sample rate:", sr)
    print("Audio length:", len(y))
    print("Duration:", len(y) / sr, "seconds")

    # Normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    else:
        raise ValueError("Silent audio detected. Please speak clearly.")

    # Do NOT aggressively trim, because short recordings may become empty
    # y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < int(0.5 * sr):
        raise ValueError("Audio is too short. Please record at least 3 seconds.")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Pad if too few frames for delta
    if mfcc.shape[1] < 9:
        pad_amount = 9 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_amount)), mode="constant")

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    min_frames = min(
        mfcc.shape[1],
        delta_mfcc.shape[1],
        delta2_mfcc.shape[1],
        zcr.shape[1],
        rms.shape[1]
    )

    mfcc = mfcc[:, :min_frames]
    delta_mfcc = delta_mfcc[:, :min_frames]
    delta2_mfcc = delta2_mfcc[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]

    features = np.vstack([
        mfcc,
        delta_mfcc,
        delta2_mfcc,
        zcr,
        rms
    ])

    features = features.T

    print("Feature shape before pad/truncate:", features.shape)

    if features.shape[0] < MAX_TIME_STEPS:
        pad_len = MAX_TIME_STEPS - features.shape[0]
        features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
    else:
        features = features[:MAX_TIME_STEPS, :]

    print("Final feature shape:", features.shape)

    if features.shape != (MAX_TIME_STEPS, FEATURE_DIM):
        raise ValueError(
            f"Feature shape mismatch. Got {features.shape}, expected {(MAX_TIME_STEPS, FEATURE_DIM)}"
        )

    return features


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_stress(audio_input):
    try:
        print("\n==============================")
        print("New prediction started")
        print("==============================")

        features = extract_features(audio_input)

        # Scale features
        features_scaled = scaler.transform(features)

        # Model input: (1, 174, 122)
        model_input = np.expand_dims(features_scaled, axis=0)

        print("Model input shape:", model_input.shape)

        prediction = model.predict(model_input, verbose=0)

        print("Raw prediction:", prediction)

        # If model has multiple outputs
        if isinstance(prediction, list):
            prediction = prediction[0]

        stress_prob = float(prediction[0][0])
        calm_prob = 1.0 - stress_prob

        if stress_prob >= THRESHOLD:
            label = "🚨 STRESS DETECTED"
            confidence = stress_prob * 100
        else:
            label = "✅ CALM / NO STRESS DETECTED"
            confidence = calm_prob * 100

        return (
            f"{label}\n\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Stress Probability: {stress_prob:.4f}\n"
            f"Calm Probability: {calm_prob:.4f}\n\n"
            f"Input Shape: {model_input.shape}"
        )

    except Exception as e:
        error_text = traceback.format_exc()
        print(error_text)

        return (
            "❌ ERROR DURING PREDICTION\n\n"
            f"{str(e)}\n\n"
            "Check terminal for full traceback."
        )


# =========================================================
# GRADIO UI - GRADIO 3.50.2 VERSION
# =========================================================
audio_input = gr.Audio(
    source="microphone",
    type="filepath",
    label="Record Your Voice"
)

output_box = gr.Textbox(
    label="Prediction Result",
    lines=10
)

demo = gr.Interface(
    fn=predict_stress,
    inputs=audio_input,
    outputs=output_box,
    title="Speech Stress Detection using Attention-BiLSTM",
    description=(
        "Record 3 to 5 seconds of speech. "
        "The system extracts MFCC, Delta MFCC, Delta-Delta MFCC, ZCR, and RMS features, "
        "then predicts whether the voice indicates stress."
    ),
    allow_flagging="never"
)


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True
    )