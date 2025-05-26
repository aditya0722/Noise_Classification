import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pickle
import logging
from io import BytesIO
from sklearn.preprocessing import LabelEncoder # <--- IMPORTANT: Need LabelEncoder here
import matplotlib.pyplot as plt
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Noise Type Predictor (YAMNet)",
    description="Upload an audio file to get its noise type prediction using YAMNet embeddings."
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Model and LabelEncoder Paths ---
CLASSIFIER_MODEL_PATH = os.path.join('models', '93.h5') # Use os.path.join for robustness
# Use the correct path for your .npy file, likely not in 'models' subfolder
LABEL_ENCODER_CLASSES_PATH = 'label_encoder_classes.npy'

# --- 4. Global Resources (Loaded Once) ---
yamnet_model = None
classifier_model = None
label_encoder = None # This will be a LabelEncoder instance
TARGET_SR = 16000

# --- 5. Resource Loading on Startup ---
@app.on_event("startup")
async def load_resources():
    global yamnet_model, classifier_model, label_encoder

    try:
        logging.info("Loading YAMNet model from TensorFlow Hub...")
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        logging.info("YAMNet model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load YAMNet model: {e}")
        raise RuntimeError(f"Failed to load YAMNet model: {e}")

    try:
        logging.info(f"Loading custom classifier model from {CLASSIFIER_MODEL_PATH}...")
        classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
        logging.info("Custom classifier model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load custom classifier model from {CLASSIFIER_MODEL_PATH}: {e}")
        raise RuntimeError(f"Failed to load custom classifier model: {e}")

    try:
        logging.info(f"Loading LabelEncoder classes from {LABEL_ENCODER_CLASSES_PATH}...")
        # Load the array of classes
        loaded_classes = np.load(LABEL_ENCODER_CLASSES_PATH, allow_pickle=True)

        # Recreate a LabelEncoder instance and fit it with the loaded classes
        label_encoder = LabelEncoder()
        label_encoder.fit(loaded_classes) # Fit the encoder with the classes
        
        logging.info(f"LabelEncoder loaded and fitted successfully. Classes: {list(label_encoder.classes_)}")
    except FileNotFoundError:
        logging.error(f"LabelEncoder classes file not found at {LABEL_ENCODER_CLASSES_PATH}.")
        raise RuntimeError(f"Failed to load LabelEncoder classes: File not found. Make sure '{LABEL_ENCODER_CLASSES_PATH}' exists.")
    except Exception as e:
        logging.error(f"Failed to load or fit LabelEncoder from {LABEL_ENCODER_CLASSES_PATH}: {e}")
        raise RuntimeError(f"Failed to load LabelEncoder: {e}")
    
def generate_melspectrogram_image(audio_bytes: bytes, target_sr: int = TARGET_SR):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=target_sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set(title='Mel Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
# --- 6. YAMNet Preprocessing Functions (from your original Kaggle script) ---
def preprocess_audio_for_yamnet(audio_file_bytes: bytes, target_sr: int = TARGET_SR):
    try:
        waveform, sr_orig = librosa.load(BytesIO(audio_file_bytes), sr=None, mono=True)
        if sr_orig != target_sr:
            waveform = librosa.resample(y=waveform, orig_sr=sr_orig, target_sr=target_sr)
        if np.max(np.abs(waveform)) > 1e-8:
            waveform = waveform / np.max(np.abs(waveform))
        else:
            waveform = np.zeros_like(waveform)
        return waveform.astype(np.float32)
    except Exception as e:
        logging.error(f"Error during audio preprocessing for YAMNet: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")

def extract_yamnet_embedding(audio_file_bytes: bytes):
    waveform = preprocess_audio_for_yamnet(audio_file_bytes)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)

# --- 7. Prediction Endpoint ---
@app.post("/upload-audio")
async def predict_noise_type(audio_file: UploadFile = File(...)):
    logging.info(f"Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")
    
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio file.")
    if classifier_model is None or yamnet_model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Models or LabelEncoder not loaded. Server might be initializing.")

    try:
        audio_bytes = await audio_file.read()

        # --- Generate Mel Spectrogram ---
        spectrogram_url = generate_melspectrogram_image(audio_bytes)

        # --- Extract YAMNet Embedding and Predict ---
        embedding = extract_yamnet_embedding(audio_bytes)
        processed_input = np.expand_dims(embedding, axis=0)
        logging.info(f"Input shape to classifier model: {processed_input.shape}")

        predictions = classifier_model.predict(processed_input)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])
        predicted_noise_type = label_encoder.inverse_transform([predicted_class_index])[0]

        # --- Top 3 Predictions ---
        top_3_indices = predictions.argsort()[-3:][::-1]
        top_3_predictions = [
            {
                "label": label_encoder.inverse_transform([i])[0],
                "confidence": float(predictions[i])
            }
            for i in top_3_indices
        ]

        return {
            "predicted_noise_type": predicted_noise_type,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "spectrogram_url": spectrogram_url
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("An error occurred during prediction:")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- 8. Health Check Endpoint ---
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "yamnet_loaded": yamnet_model is not None,
        "classifier_loaded": classifier_model is not None,
        "label_encoder_loaded": label_encoder is not None
    }

# --- 9. Run the FastAPI Application (for local development) ---
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT env var, or 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
