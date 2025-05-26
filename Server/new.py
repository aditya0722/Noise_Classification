from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import librosa
import numpy as np
from io import BytesIO
import soundfile as sf

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained .h5 model
model_path = 'C:\\Users\\sharm\\OneDrive\\Desktop\\AdityaReact\\Noise_Classification\\Server\\models\\urban_sound_classifier.h5'
model = load_model(model_path)

# Labels (update if different)
LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music"
]
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file into bytes
        audio_bytes = await file.read()

        # Load audio with librosa from bytes
        audio_np, sr = sf.read(BytesIO(audio_bytes))
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)  # convert to mono if stereo

        # Resample if needed
        if sr != 22050:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=22050)
            sr = 22050

        # Generate mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=audio_np,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        log_spectrogram = librosa.power_to_db(spectrogram)

        # Resize or pad to ensure (128, 173)
        desired_shape = (128, 173)
        if log_spectrogram.shape[1] < desired_shape[1]:
            pad_width = desired_shape[1] - log_spectrogram.shape[1]
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_spectrogram = log_spectrogram[:, :desired_shape[1]]

        # Normalize
        log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))

        # Prepare input for model
        input_data = log_spectrogram[..., np.newaxis]  # (128, 173, 1)
        input_data = np.expand_dims(input_data, axis=0)  # (1, 128, 173, 1)

        # Predict
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = LABELS[predicted_class_index]
        confidence = float(np.max(predictions))

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))