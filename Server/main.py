import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import logging
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import gc
import psutil
import base64
import threading
import time
from functools import lru_cache

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Ultra Memory Optimization Settings ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['TFHUB_CACHE_COMPRESSED'] = '1'  # Compress hub cache
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA

# Pre-import cleanup
gc.collect()

# Aggressive TensorFlow settings
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.optimizer.set_jit(False)  # Disable XLA JIT compilation

# --- Ultra Memory Manager ---
class UltraMemoryManager:
    def __init__(self, threshold_mb=150):  # Even lower threshold
        self.threshold_mb = threshold_mb
        self.monitoring = False
        self._lock = threading.Lock()

    def get_memory_usage(self):
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0

    def aggressive_cleanup(self):
        with self._lock:
            # Multiple garbage collection passes
            for _ in range(5):  # More aggressive
                gc.collect()
            
            # Clear TensorFlow session and graph
            try:
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass
            
            # Clear TensorFlow Hub cache
            try:
                import tempfile
                import shutil
                hub_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
                if os.path.exists(hub_cache):
                    for item in os.listdir(hub_cache):
                        item_path = os.path.join(hub_cache, item)
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path, ignore_errors=True)
                        except Exception:
                            pass
            except Exception:
                pass
            
            # System level cleanup (Windows/Linux)
            try:
                if os.name == 'nt':  # Windows
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                else:  # Linux
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
            except Exception:
                pass

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            threading.Thread(target=self._monitor_loop, daemon=True).start()

    def _monitor_loop(self):
        while self.monitoring:
            try:
                mem = self.get_memory_usage()
                if mem > self.threshold_mb:
                    self.aggressive_cleanup()
                    logging.warning(f"Aggressive cleanup at {mem:.1f} MB")
                time.sleep(15)  # More frequent monitoring
            except Exception:
                pass

memory_manager = UltraMemoryManager(threshold_mb=120)  # Very aggressive threshold

# --- FastAPI App ---
app = FastAPI(
    title="Ultra Optimized Noise Predictor",
    description="Minimal memory footprint audio classification."
)

# Minimal logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# CORS setup
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "http://59.97.138.60:3000",
    "https://noise-classification-jvmg.vercel.app",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Only needed methods
    allow_headers=["*"],
)

# --- Configurations ---
CLASSIFIER_MODEL_PATH = os.path.join('models', '93.h5')
LABEL_ENCODER_CLASSES_PATH = 'label_encoder_classes.npy'
TARGET_SR = 8000  # Reduced from 16000 for memory savings

# Global Models and Locks
_yamnet_model = None
_classifier_model = None
_label_encoder = None
_yamnet_lock = threading.Lock()
_classifier_lock = threading.Lock()

# --- Ultra-cached Label Encoder ---
@lru_cache(maxsize=1)
def _load_label_encoder_cached():
    classes = np.load(LABEL_ENCODER_CLASSES_PATH, allow_pickle=True)
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder

def get_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        _label_encoder = _load_label_encoder_cached()
    return _label_encoder

# --- Load YAMNet with Memory Constraints ---
def load_yamnet_minimal():
    global _yamnet_model
    with _yamnet_lock:
        if _yamnet_model is None:
            memory_manager.aggressive_cleanup()
            
            # Clear TensorFlow Hub cache first
            try:
                import shutil
                import tempfile
                hub_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
                if os.path.exists(hub_cache):
                    shutil.rmtree(hub_cache, ignore_errors=True)
            except Exception:
                pass
                
            # Load with minimal options and error handling
            try:
                # Force fresh download and clear any cached corrupted files
                os.environ['TFHUB_CACHE_DIR'] = tempfile.mkdtemp()
                _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
                logging.warning("YAMNet loaded successfully")
            except Exception as e:
                logging.error(f"YAMNet load failed: {e}")
                # Try alternative loading method
                try:
                    tf.keras.backend.clear_session()
                    memory_manager.aggressive_cleanup()
                    _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
                    logging.warning("YAMNet loaded on retry")
                except Exception as e2:
                    logging.error(f"YAMNet retry failed: {e2}")
                    raise HTTPException(status_code=500, detail="YAMNet model loading failed - please restart server")
    return _yamnet_model

# --- Aggressive YAMNet Unload ---
def unload_yamnet():
    global _yamnet_model
    with _yamnet_lock:
        if _yamnet_model is not None:
            try:
                del _yamnet_model
                _yamnet_model = None
                memory_manager.aggressive_cleanup()
            except Exception:
                pass

# --- Load Classifier with Memory Constraints ---
def load_classifier_minimal():
    global _classifier_model
    with _classifier_lock:
        if _classifier_model is None:
            try:
                memory_manager.aggressive_cleanup()
                _classifier_model = tf.keras.models.load_model(
                    CLASSIFIER_MODEL_PATH, 
                    compile=False
                )
                # Optimize model for inference
                _classifier_model.trainable = False
                logging.warning("Classifier loaded")
            except Exception as e:
                logging.error(f"Classifier load failed: {e}")
                raise HTTPException(status_code=500, detail="Classifier loading failed")
    return _classifier_model

# --- Ultra-minimal Audio Preprocessing ---
def preprocess_audio_ultra_minimal(audio_bytes: bytes, max_duration=15.0):  # Reduced duration
    try:
        # Use lower sample rate and shorter duration
        waveform, sr = librosa.load(
            BytesIO(audio_bytes), 
            sr=TARGET_SR,  # 8kHz instead of 16kHz
            mono=True, 
            duration=max_duration,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        if len(waveform) == 0:
            raise ValueError("Empty audio")
            
        # Normalize more efficiently
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
            
        # Convert to float32 and cleanup
        result = waveform.astype(np.float32)
        del waveform
        
        return result
        
    except Exception as e:
        logging.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file")

# --- Ultra-optimized YAMNet Embedding ---
def extract_yamnet_embedding_ultra_optimized(audio_bytes: bytes):
    try:
        # Preprocess with lower sample rate
        waveform = preprocess_audio_ultra_minimal(audio_bytes)
        
        # Upsample only for YAMNet (it expects 16kHz)
        waveform_16k = librosa.resample(waveform, orig_sr=TARGET_SR, target_sr=16000, res_type='kaiser_fast')
        
        yamnet = load_yamnet_minimal()
        
        # Get embeddings
        with tf.device('/CPU:0'):  # Force CPU
            _, embeddings, _ = yamnet(waveform_16k)
            
        # Get mean embedding and convert immediately
        embedding_mean = np.mean(embeddings.numpy(), axis=0, dtype=np.float32)
        
        # Immediate cleanup
        del waveform, waveform_16k, embeddings
        unload_yamnet()
        memory_manager.aggressive_cleanup()
        
        return embedding_mean
        
    except Exception as e:
        unload_yamnet()
        memory_manager.aggressive_cleanup()
        logging.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Feature extraction failed")

# --- Minimal Spectrogram Generation ---
def generate_ultra_minimal_spectrogram(audio_bytes: bytes):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        # Ultra-minimal spectrogram settings
        y, sr = librosa.load(
            BytesIO(audio_bytes), 
            sr=4000,  # Very low sample rate for spectrogram
            mono=True, 
            duration=8  # Shorter duration
        )
        
        # Minimal mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=16,  # Reduced from 32
            n_fft=256,  # Reduced from 512
            hop_length=256  # Reduced from 512
        )
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Ultra-minimal plot
        fig, ax = plt.subplots(figsize=(2.5, 1.5))  # Smaller figure
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', 
                                cmap='viridis', ax=ax)
        ax.set_title('Spectrogram', fontsize=6)
        plt.tight_layout()

        # Save with lower quality
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=30, bbox_inches='tight')  # Lower DPI
        plt.close(fig)
        plt.clf()
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Cleanup
        del y, S, S_DB, buf
        memory_manager.aggressive_cleanup()

        return img_base64
        
    except Exception as e:
        logging.warning(f"Spectrogram generation failed: {e}")
        return None

# --- Ultra-optimized Main Prediction ---
@app.post("/upload-audio")
async def predict_noise_type(audio_file: UploadFile = File(...)):
    # Validate file
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid audio file")

    try:
        audio_bytes = await audio_file.read()
        
        # Smaller file size limit
        if len(audio_bytes) > 5 * 1024 * 1024:  # 5MB instead of 10MB
            raise HTTPException(status_code=400, detail="File too large (max 5MB)")

        # Extract embedding with aggressive cleanup
        embedding = extract_yamnet_embedding_ultra_optimized(audio_bytes)
        input_array = np.expand_dims(embedding, axis=0)

        # Load classifier and predict
        classifier = load_classifier_minimal()
        
        with tf.device('/CPU:0'):
            preds = classifier.predict(input_array, verbose=0, batch_size=1)[0]

        # Get top predictions efficiently
        encoder = get_label_encoder()
        top_3_idx = np.argpartition(preds, -3)[-3:]
        top_3_idx = top_3_idx[np.argsort(preds[top_3_idx])[::-1]]
        
        top_3 = []
        for i in top_3_idx:
            label = encoder.inverse_transform([i])[0]
            confidence = float(preds[i])
            top_3.append({"label": label, "confidence": confidence})

        best_label = top_3[0]["label"]
        best_confidence = top_3[0]["confidence"]

        # Generate minimal spectrogram
        spectrogram_base64 = None
        try:
            spectrogram_base64 = generate_ultra_minimal_spectrogram(audio_bytes)
        except Exception:
            pass  # Continue without spectrogram if it fails

        # Final cleanup
        del audio_bytes, embedding, input_array, preds
        memory_manager.aggressive_cleanup()

        return {
            "predicted_noise_type": best_label,
            "confidence": best_confidence,
            "top_3_predictions": top_3,
            "spectrogram_url": spectrogram_base64,
        }

    except HTTPException as e:
        memory_manager.aggressive_cleanup()
        raise e
    except Exception as e:
        memory_manager.aggressive_cleanup()
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Health Check ---
@app.get("/health")
async def health_check():
    mem_usage = memory_manager.get_memory_usage()
    return {
        "status": "healthy",
        "memory_mb": round(mem_usage, 1)
    }

# Start ultra memory monitoring
memory_manager.start_monitoring()

# Startup cleanup
@app.on_event("startup")
async def startup_event():
    # Clear any existing TF Hub cache at startup
    try:
        import tempfile
        import shutil
        hub_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
        if os.path.exists(hub_cache):
            shutil.rmtree(hub_cache, ignore_errors=True)
    except Exception:
        pass
    
    memory_manager.aggressive_cleanup()
    logging.warning("Ultra-optimized server started")

@app.on_event("shutdown") 
async def shutdown_event():
    global _yamnet_model, _classifier_model
    try:
        if _yamnet_model is not None:
            del _yamnet_model
            _yamnet_model = None
        if _classifier_model is not None:
            del _classifier_model
            _classifier_model = None
    except Exception:
        pass
    memory_manager.aggressive_cleanup()