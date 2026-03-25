"""
╔══════════════════════════════════════════════════════════════════════╗
║  packages.py — Single source of truth for ALL package versions       ║
║                                                                      ║
║  HOW VERSIONS ARE CHOSEN:                                            ║
║  • torch 2.3.1 + tensorflow 2.16.1 → last tested coexisting pair    ║
║  • numpy MUST stay <2.0 — both frameworks require it                 ║
║  • torchaudio/torchvision/torchtext versions MUST match torch        ║
║  • numba/llvmlite pair MUST be pinned together                       ║
║                                                                      ║
║  To upgrade: change versions here only.  Everything else auto-       ║
║  adapts.                                                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════
#  CORE — must coexist without conflicts
# ════════════════════════════════════════════════════════════════════

PYTHON_MIN       = (3, 9)

# PyTorch family  — ALL must share the same minor version
TORCH_VER        = "2.3.1"
TORCHVISION_VER  = "0.18.1"
TORCHAUDIO_VER   = "2.3.1"
TORCHTEXT_VER    = "0.18.0"
TORCHCODEC_VER   = "0.1.0"      # Linux/CUDA only

# TensorFlow family
TENSORFLOW_VER   = "2.16.1"
KERAS_VER        = "3.3.3"
TF_HUB_VER       = "0.16.1"
TF_DATASETS_VER  = "4.9.4"
TF_ADDONS_VER    = "0.23.0"

# numpy — CRITICAL: <2.0 required by both torch and tensorflow
NUMPY_VER        = "1.26.4"

# numba + llvmlite — must be pinned together
NUMBA_VER        = "0.60.0"
LLVMLITE_VER     = "0.43.0"

# CUDA wheel index URLs (matched to torch 2.3.1 release)
CUDA_INDEX = {
    "12" : "https://download.pytorch.org/whl/cu121",
    "11" : "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# ════════════════════════════════════════════════════════════════════
#  PRE-INSTALL
#  Must be installed BEFORE torch/tf to avoid build conflicts.
# ════════════════════════════════════════════════════════════════════

PRE_PACKAGES = [
    f"numpy=={NUMPY_VER}",       # pin first — everything depends on this
    "scipy==1.13.1",
    "packaging>=23.1",
    "typing-extensions>=4.8",
    "wheel",
    "setuptools>=68",
    "Cython",
    "pybind11",
    "cffi",
]

# ════════════════════════════════════════════════════════════════════
#  AUDIO PACKAGES  ← core of your project
# ════════════════════════════════════════════════════════════════════

AUDIO_PACKAGES = [
    # ── Signal Processing ─────────────────────────────────────────
    "librosa==0.10.2",              # gold-standard: mel, STFT, MFCCs, CQT
    "soundfile==0.12.1",            # .wav / .flac / .ogg read-write (libsndfile backend)
    "audioread==3.0.1",             # ffmpeg backend fallback for .mp3/.ogg
    "pydub==0.25.1",                # audio manipulation: trim, concat, convert, mix
    "noisereduce==3.0.2",           # noise reduction
    "resampy==0.4.3",               # high-quality audio resampling
    "soxr==0.3.7",                  # fast resampling (used by librosa)

    # ── Feature Extraction ────────────────────────────────────────
    "python-speech-features==0.6",  # MFCCs, delta, filterbanks
    "opensmile==2.5.0",             # 6k+ audio features (eGeMAPS, ComParE)
    "pyworld==0.3.4",               # vocoder, F0/pitch extraction
    "praat-parselmouth==0.4.3",     # Praat bindings: formants, prosody, pitch

    # ── Speech / Audio Models ─────────────────────────────────────
    "speechbrain==1.0.0",           # ASR, speaker ID, emotion, enhancement

    # ── Audio I/O ─────────────────────────────────────────────────
    "sounddevice==0.4.7",           # real-time audio I/O (PortAudio)
    "mutagen==1.47.0",              # read/write .ogg/.mp3/.wav metadata/tags
    "tinytag==1.10.1",              # lightweight tag reader

    # ── Codec support ─────────────────────────────────────────────
    "ffmpeg-python==0.2.0",         # Python bindings for ffmpeg CLI
]

# ════════════════════════════════════════════════════════════════════
#  DEEP LEARNING UTILITIES
# ════════════════════════════════════════════════════════════════════

DL_PACKAGES = [
    # TensorFlow ecosystem (torch itself installed hardware-aware)
    f"tensorflow=={TENSORFLOW_VER}",
    f"keras=={KERAS_VER}",
    f"tensorflow-hub=={TF_HUB_VER}",
    f"tensorflow-datasets=={TF_DATASETS_VER}",
    f"tensorflow-addons=={TF_ADDONS_VER}",

    # PyTorch utilities
    "torchmetrics==1.4.0",          # metrics: accuracy, F1, AUC, etc.
    "lightning==2.3.3",             # PyTorch Lightning (training loop abstraction)
    "timm==1.0.3",                  # 700+ pretrained vision/audio models
    "einops==0.8.0",                # tensor reshaping for transformers / CNNs
    "opt-einsum==3.3.0",
]

# ════════════════════════════════════════════════════════════════════
#  NLP / TEXT MODELS
# ════════════════════════════════════════════════════════════════════

NLP_PACKAGES = [
    "transformers==4.42.4",         # Wav2Vec2, Whisper, BERT, HuBERT, etc.
    "tokenizers==0.19.1",
    "datasets==2.20.0",
    "accelerate==0.31.0",           # multi-GPU / mixed precision training
    "huggingface-hub==0.23.4",
    "sentence-transformers==3.0.1",
    "nltk==3.8.1",
    "spacy==3.7.5",
    "gensim==4.3.2",
    "sentencepiece==0.2.0",         # tokenizer for Whisper, T5, mBERT
    "sacremoses==0.1.1",
]

# ════════════════════════════════════════════════════════════════════
#  COMPUTER VISION  (spectrogram → image pipelines)
# ════════════════════════════════════════════════════════════════════

CV_PACKAGES = [
    "opencv-python-headless==4.10.0.84",
    "albumentations==1.4.10",
    "Pillow==10.3.0",
    "imageio==2.34.2",
    "scikit-image==0.23.2",
]

# ════════════════════════════════════════════════════════════════════
#  CLASSICAL ML
# ════════════════════════════════════════════════════════════════════

ML_PACKAGES = [
    "scikit-learn==1.5.0",
    "xgboost==2.0.3",
    "lightgbm==4.3.0",
    "imbalanced-learn==0.12.3",
    "joblib==1.4.2",
    "cloudpickle==3.0.0",
]

# ════════════════════════════════════════════════════════════════════
#  EXPERIMENT TRACKING & VISUALISATION
# ════════════════════════════════════════════════════════════════════

EXPERIMENT_PACKAGES = [
    "mlflow==2.14.1",
    "wandb==0.17.3",
    "tensorboard==2.16.2",
    "plotly==5.22.0",
    "seaborn==0.13.2",
    "matplotlib==3.9.0",
    "tqdm==4.66.4",
    "rich==13.7.1",
]

# ════════════════════════════════════════════════════════════════════
#  JUPYTER / DEV
# ════════════════════════════════════════════════════════════════════

JUPYTER_PACKAGES = [
    "jupyterlab==4.2.3",
    "notebook==7.2.1",
    "ipywidgets==8.1.3",
    "ipykernel==6.29.4",
    "nbformat==5.10.4",
    "ipython==8.25.0",
    "jupyter-contrib-nbextensions==0.7.0",
]

# ════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════

UTIL_PACKAGES = [
    "pandas==2.2.2",
    "h5py==3.11.0",
    "pyarrow==16.1.0",
    "requests==2.32.3",
    "pyyaml==6.0.1",
    "python-dotenv==1.0.1",
    "psutil==6.0.0",
    f"numba=={NUMBA_VER}",          # JIT compiler — speeds up librosa significantly
    f"llvmlite=={LLVMLITE_VER}",    # numba backend — MUST match numba version
]
