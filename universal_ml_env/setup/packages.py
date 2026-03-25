"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  packages.py  —  Single source of truth for EVERY package + version         ║
║                                                                              ║
║  To upgrade any version → change it HERE only.                              ║
║  Everything else (install, verify, requirements.txt) auto-adapts.           ║
║                                                                              ║
║  Compatibility rules enforced here:                                         ║
║    • numpy < 2.0         — required by both torch AND tensorflow             ║
║    • torchaudio/vision/text minor must match torch minor                    ║
║    • numba + llvmlite must be pinned as a matched pair                      ║
║    • tensorflow + torch coexistence tested at these exact versions          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════
#  CORE VERSION PINS
# ════════════════════════════════════════════════════════════════════════

PYTHON_MIN      = (3, 9)

# PyTorch family  ← ALL must share the same minor (2.3.x)
TORCH_VER       = "2.3.1"
TORCHVISION_VER = "0.18.1"
TORCHAUDIO_VER  = "2.3.1"
TORCHTEXT_VER   = "0.18.0"
TORCHCODEC_VER  = "0.1.0"        # Linux only

# TensorFlow family
TENSORFLOW_VER  = "2.16.1"
KERAS_VER       = "3.3.3"

# numpy — MUST stay < 2.0 (both torch + tf hard-require this)
NUMPY_VER       = "1.26.4"

# numba + llvmlite are a matched pair — never change one without the other
NUMBA_VER       = "0.60.0"
LLVMLITE_VER    = "0.43.0"

# CUDA PyTorch wheel indices (matched to torch 2.3.1 release)
CUDA_INDEX = {
    "12" : "https://download.pytorch.org/whl/cu121",
    "11" : "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# ════════════════════════════════════════════════════════════════════════
#  GROUP 0 — PRE-INSTALL
#  Must be installed BEFORE torch/tf to avoid build conflicts.
# ════════════════════════════════════════════════════════════════════════

PRE = [
    f"numpy=={NUMPY_VER}",        # pin first — everything depends on this
    "scipy==1.13.1",
    "packaging>=23.1",
    "typing-extensions>=4.8",
    "wheel",
    "setuptools>=68",
    "Cython",
    "pybind11",
    "cffi",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 1 — CORE (every profile gets this)
# ════════════════════════════════════════════════════════════════════════

CORE = [
    # Scientific base
    f"numpy=={NUMPY_VER}",
    "scipy==1.13.1",
    "pandas==2.2.2",
    "matplotlib==3.9.0",
    "seaborn==0.13.2",
    "Pillow==10.3.0",

    # Utilities
    "tqdm==4.66.4",
    "rich==13.7.1",
    "pyyaml==6.0.1",
    "python-dotenv==1.0.1",
    "requests==2.32.3",
    "joblib==1.4.2",
    "cloudpickle==3.0.0",
    "psutil==6.0.0",
    "h5py==3.11.0",
    "pyarrow==16.1.0",
    "openpyxl==3.1.4",
    "packaging>=23.1",

    # Visualization
    "plotly==5.22.0",
    "bokeh==3.4.1",
    "altair==5.3.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 2 — CLASSICAL ML  (tabular + general ML)
# ════════════════════════════════════════════════════════════════════════

ML = [
    "scikit-learn==1.5.0",
    "xgboost==2.0.3",
    "lightgbm==4.3.0",
    "catboost==1.2.5",
    "imbalanced-learn==0.12.3",

    # Feature engineering
    "feature-engine==1.8.1",
    "category-encoders==2.6.3",

    # Model interpretation
    "shap==0.45.1",
    "lime==0.2.0.1",

    # Hyperparameter search
    "optuna==3.6.1",
    "hyperopt==0.2.7",

    # AutoML
    "auto-sklearn==0.15.0",     # Linux only — skipped gracefully on others
    "pycaret==3.3.2",

    # Stats
    "statsmodels==0.14.2",
    "pingouin==0.5.4",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 3 — DEEP LEARNING CORE
# ════════════════════════════════════════════════════════════════════════

DL_CORE = [
    # TensorFlow ecosystem
    f"tensorflow=={TENSORFLOW_VER}",
    f"keras=={KERAS_VER}",
    "tensorflow-hub==0.16.1",
    "tensorflow-datasets==4.9.4",
    "tensorflow-addons==0.23.0",

    # PyTorch utilities (torch itself installed hardware-aware)
    "torchmetrics==1.4.0",
    "lightning==2.3.3",           # PyTorch Lightning
    "einops==0.8.0",              # tensor reshaping
    "opt-einsum==3.3.0",

    # Experiment tracking
    "mlflow==2.14.1",
    "wandb==0.17.3",
    "tensorboard==2.16.2",

    # numba for JIT (used by librosa, UMAP, etc.)
    f"numba=={NUMBA_VER}",
    f"llvmlite=={LLVMLITE_VER}",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 4 — COMPUTER VISION
# ════════════════════════════════════════════════════════════════════════

CV = [
    "opencv-python-headless==4.10.0.84",
    "albumentations==1.4.10",
    "scikit-image==0.23.2",
    "imageio==2.34.2",
    "timm==1.0.3",                # 700+ pretrained vision models
    "torchvision==0.18.1",        # also needed by torch — matched version

    # Object detection / segmentation
    "ultralytics==8.2.0",         # YOLOv8 — state-of-the-art detection
    "supervision==0.21.0",        # detection utilities

    # Image generation / GANs
    "diffusers==0.29.0",
    "accelerate==0.31.0",

    # OCR
    "pytesseract==0.3.10",
    "easyocr==1.7.1",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 5 — NLP / TEXT
# ════════════════════════════════════════════════════════════════════════

NLP = [
    "transformers==4.42.4",
    "tokenizers==0.19.1",
    "datasets==2.20.0",
    "accelerate==0.31.0",
    "huggingface-hub==0.23.4",
    "sentence-transformers==3.0.1",

    # Classic NLP
    "nltk==3.8.1",
    "spacy==3.7.5",
    "gensim==4.3.2",
    "textblob==0.18.0",
    "sentencepiece==0.2.0",
    "sacremoses==0.1.1",

    # Text utilities
    "beautifulsoup4==4.12.3",
    "lxml==5.2.2",
    "regex==2024.5.15",
    "ftfy==6.2.0",               # fix unicode text issues
    "langdetect==1.0.9",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 6 — AUDIO / SPEECH
# ════════════════════════════════════════════════════════════════════════

AUDIO = [
    # Signal processing
    "librosa==0.10.2",
    "soundfile==0.12.1",
    "audioread==3.0.1",
    "pydub==0.25.1",
    "noisereduce==3.0.2",
    "resampy==0.4.3",
    "soxr==0.3.7",

    # Feature extraction
    "python-speech-features==0.6",
    "opensmile==2.5.0",
    "pyworld==0.3.4",
    "praat-parselmouth==0.4.3",

    # Speech models
    "speechbrain==1.0.0",

    # Audio I/O
    "sounddevice==0.4.7",
    "mutagen==1.47.0",
    "ffmpeg-python==0.2.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 7 — REINFORCEMENT LEARNING
# ════════════════════════════════════════════════════════════════════════

RL = [
    "gymnasium==0.29.1",          # updated OpenAI Gym
    "stable-baselines3==2.3.2",
    "sb3-contrib==2.3.0",
    "ale-py==0.9.0",              # Atari environments
    "box2d-py==2.3.5",
    "pygame==2.6.0",
    "shimmy==1.3.0",              # Gym compatibility shim
    "tensorboard==2.16.2",        # RL training dashboards
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 8 — LLMs / GENERATIVE AI
# ════════════════════════════════════════════════════════════════════════

LLM = [
    "transformers==4.42.4",
    "accelerate==0.31.0",
    "peft==0.11.1",               # LoRA, QLoRA, prefix tuning
    "bitsandbytes==0.43.1",       # 4-bit / 8-bit quantisation
    "trl==0.9.4",                 # RLHF, SFT, DPO training
    "datasets==2.20.0",
    "huggingface-hub==0.23.4",
    "sentencepiece==0.2.0",

    # LangChain ecosystem
    "langchain==0.2.5",
    "langchain-community==0.2.5",
    "langchain-core==0.2.9",
    "openai==1.35.3",             # OpenAI API client
    "anthropic==0.29.0",          # Anthropic API client
    "tiktoken==0.7.0",            # tokenizer for OpenAI models

    # Vector databases
    "faiss-cpu==1.8.0",           # Facebook vector search
    "chromadb==0.5.3",            # local vector DB
    "qdrant-client==1.9.1",

    # Embeddings / RAG
    "sentence-transformers==3.0.1",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 9 — TIME SERIES
# ════════════════════════════════════════════════════════════════════════

TIMESERIES = [
    "statsmodels==0.14.2",
    "prophet==1.1.5",
    "pmdarima==2.0.4",            # auto-ARIMA
    "sktime==0.30.0",
    "darts==0.30.0",
    "tsfresh==0.20.2",
    "tslearn==0.6.3",
    "pyod==1.1.3",                # anomaly detection
    "kats==0.2.0",                # Meta Kats
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 10 — DATA ENGINEERING / PIPELINES
# ════════════════════════════════════════════════════════════════════════

DATA_ENG = [
    "sqlalchemy==2.0.30",
    "psycopg2-binary==2.9.9",     # PostgreSQL
    "pymongo==4.7.3",
    "redis==5.0.6",
    "apache-airflow==2.9.2",      # pipeline orchestration
    "prefect==2.19.8",
    "great-expectations==0.18.15",# data quality
    "pandas-profiling==3.6.6",    # EDA reports
    "ydata-profiling==4.8.3",     # successor to pandas-profiling
    "dask==2024.6.0",             # parallel pandas
    "polars==0.20.31",            # fast dataframes
    "pyarrow==16.1.0",
    "fastparquet==2024.5.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 11 — JUPYTER / DEV
# ════════════════════════════════════════════════════════════════════════

JUPYTER = [
    "jupyterlab==4.2.3",
    "notebook==7.2.1",
    "ipywidgets==8.1.3",
    "ipykernel==6.29.4",
    "nbformat==5.10.4",
    "nbconvert==7.16.4",
    "ipython==8.25.0",
    "jupyter-contrib-nbextensions==0.7.0",
    "black==24.4.2",              # code formatter
    "isort==5.13.2",
]

# ════════════════════════════════════════════════════════════════════════
#  PROFILE → PACKAGE GROUP MAPPING
#  Defines exactly which groups each profile installs.
# ════════════════════════════════════════════════════════════════════════

PROFILE_GROUPS = {
    "full"      : ["core", "ml", "dl_core", "cv", "nlp", "audio",
                   "rl", "llm", "timeseries", "jupyter"],
    "nlp"       : ["core", "ml", "dl_core", "nlp", "jupyter"],
    "cv"        : ["core", "ml", "dl_core", "cv", "jupyter"],
    "tabular"   : ["core", "ml", "jupyter"],
    "rl"        : ["core", "dl_core", "rl", "jupyter"],
    "audio"     : ["core", "ml", "dl_core", "nlp", "audio", "jupyter"],
    "llm"       : ["core", "dl_core", "nlp", "llm", "jupyter"],
    "timeseries": ["core", "ml", "timeseries", "jupyter"],
}

GROUP_MAP = {
    "core"      : CORE,
    "ml"        : ML,
    "dl_core"   : DL_CORE,
    "cv"        : CV,
    "nlp"       : NLP,
    "audio"     : AUDIO,
    "rl"        : RL,
    "llm"       : LLM,
    "timeseries": TIMESERIES,
    "jupyter"   : JUPYTER,
}


def get_profile_packages(profile: str) -> dict:
    """
    Return an ordered dict of {group_name: [packages]}
    for the given profile.
    """
    groups = PROFILE_GROUPS.get(profile, PROFILE_GROUPS["full"])
    return {g: GROUP_MAP[g] for g in groups if g in GROUP_MAP}


def get_flat_list(profile: str) -> list:
    """Flat list of all packages for this profile (for requirements.txt)."""
    result = []
    for pkgs in get_profile_packages(profile).values():
        result.extend(pkgs)
    # deduplicate preserving order
    seen = set()
    return [p for p in result if not (p in seen or seen.add(p))]
