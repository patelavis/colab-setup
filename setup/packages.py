"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  packages.py — EXACT Google Colab Runtime Package Versions                  ║
║                                                                              ║
║  Source  : github.com/googlecolab/backend-info  pip-freeze.txt (live repo)  ║
║  Docker  : us-docker.pkg.dev/colab-images/public/runtime  (GPU)             ║
║            us-docker.pkg.dev/colab-images/public/cpu-runtime (CPU)          ║
║                                                                              ║
║  Runtime snapshot (2025, latest Colab GPU image):                           ║
║    Python  3.12                                                              ║
║    CUDA    12.5    cuDNN 9.x    NVIDIA driver 550.54.15                     ║
║    torch   2.5.1+cu121   (Colab ships cu121 wheel on CUDA 12.5 host)        ║
║    tensorflow  2.18.0                                                        ║
║    keras   3.8.0                                                             ║
║    numpy   1.26.4  (still <2.0 — TF 2.18 requires this)                    ║
║                                                                              ║
║  To upgrade: edit ONLY this file. Everything else auto-adapts.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════
#  RUNTIME IDENTITY — mirrors actual Colab Docker image
# ════════════════════════════════════════════════════════════════════════

COLAB_PYTHON          = "3.12"
COLAB_CUDA            = "12.5"
COLAB_CUDNN           = "9.x"
COLAB_NVIDIA_DRIVER   = "550.54.15"

# ════════════════════════════════════════════════════════════════════════
#  CORE FRAMEWORK PINS  (from real pip-freeze.txt + colabtools issues)
# ════════════════════════════════════════════════════════════════════════

# ── PyTorch family ── (Colab ships cu121 wheels even on CUDA 12.5 host)
TORCH_VER         = "2.5.1"
TORCHVISION_VER   = "0.20.1"
TORCHAUDIO_VER    = "2.5.1"
TORCHTEXT_VER     = "0.18.0"      # last stable matching minor
TORCHCODEC_VER    = "0.1.0"       # Linux only

# ── TensorFlow family ──
TENSORFLOW_VER    = "2.18.0"      # confirmed: colabtools issue #5061
KERAS_VER         = "3.8.0"       # confirmed: colabtools issue #5061

# ── numpy — CRITICAL: TF 2.18 still requires <2.0 ──
NUMPY_VER         = "1.26.4"      # from actual pip-freeze.txt

# ── numba + llvmlite exact pair ──
NUMBA_VER         = "0.60.0"
LLVMLITE_VER      = "0.43.0"

# ── CUDA PyTorch wheel indexes ──
# Colab uses cu121 wheels (stable, tested). Use cu124 if your CUDA is 12.4+
CUDA_INDEX = {
    "12" : "https://download.pytorch.org/whl/cu121",   # Colab's actual choice
    "11" : "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# ════════════════════════════════════════════════════════════════════════
#  GROUP 0 — PRE-INSTALL
#  Must run BEFORE torch/tf. Sets up build tools + pins numpy.
# ════════════════════════════════════════════════════════════════════════

PRE = [
    f"numpy=={NUMPY_VER}",        # MUST be first — both torch and tf require <2.0
    "scipy==1.15.3",               # from pip-freeze.txt
    "packaging>=23.1",
    "typing-extensions>=4.8",
    "wheel",
    "setuptools>=75.1.0",          # from pip-freeze.txt
    "Cython",
    "pybind11",
    "cffi",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 1 — CORE   (every profile gets this — mirrors Colab base)
# ════════════════════════════════════════════════════════════════════════

CORE = [
    # ── Exact from Colab pip-freeze.txt ────────────────────────────
    f"numpy=={NUMPY_VER}",
    "scipy==1.15.3",
    "pandas==2.2.2",
    "matplotlib==3.10.0",
    "seaborn==0.13.2",
    "Pillow==11.2.1",
    "sympy==1.13.1",
    "networkx==3.4.2",
    "pyarrow==19.0.1",
    "requests==2.32.3",
    "urllib3==2.4.0",
    "PyYAML==6.0.2",
    "rich==13.9.4",
    "tqdm==4.67.1",
    "regex==2024.11.6",
    "safetensors==0.5.3",
    "packaging>=23.1",
    "six==1.17.0",
    "python-dateutil==2.9.0.post0",
    "pytz==2025.2",
    "joblib==1.4.2",
    "cloudpickle==3.1.1",
    "psutil==6.1.1",
    "h5py==3.13.0",
    "openpyxl==3.1.5",
    "xlrd==2.0.1",
    "SQLAlchemy==2.0.40",

    # ── Visualization ────────────────────────────────────────────
    "plotly==6.0.1",
    "bokeh==3.7.2",
    "altair==5.5.0",
    "PyWavelets==1.8.0",

    # ── Polars (Colab ships this now) ────────────────────────────
    "polars==1.21.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 2 — CLASSICAL ML   (from Colab pip-freeze.txt, exact versions)
# ════════════════════════════════════════════════════════════════════════

ML = [
    # ── Exact Colab versions ─────────────────────────────────────
    "scikit-learn==1.6.1",         # from pip-freeze.txt
    "scikit-image==0.25.2",        # from pip-freeze.txt
    "xgboost==2.1.4",              # from pip-freeze.txt (3.0 in newer)
    "lightgbm==4.6.0",
    "catboost==1.2.8",
    "imbalanced-learn==0.13.0",
    "statsmodels==0.14.4",
    "shap==0.47.2",                # from pip-freeze.txt
    "optuna==4.3.0",
    "hyperopt==0.2.7",
    "feature-engine==1.8.2",
    "category-encoders==2.6.4",
    "pycaret==3.3.2",
    "pingouin==0.5.5",

    # ── Dimensionality reduction (Colab has these) ───────────────
    "umap-learn==0.5.7",
    "hdbscan==0.8.40",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 3 — DEEP LEARNING CORE
#  torch/tf installed separately (hardware-aware). This group = utilities.
# ════════════════════════════════════════════════════════════════════════

DL_CORE = [
    # ── TF ecosystem (from Colab pip-freeze.txt) ─────────────────
    f"tensorflow=={TENSORFLOW_VER}",
    f"keras=={KERAS_VER}",
    "tensorflow-hub==0.16.1",
    "tensorflow-datasets==4.9.7",

    # ── JAX (Colab ships JAX!) ───────────────────────────────────
    "jax[cuda12]",                 # GPU JAX — Colab includes this
    # On CPU: "jax[cpu]"

    # ── PyTorch utilities ────────────────────────────────────────
    "torchmetrics==1.7.1",
    "lightning==2.5.0",
    "einops==0.8.1",
    "opt-einsum==3.4.0",
    "timm==1.0.15",                # from pip-freeze.txt

    # ── Experiment tracking ──────────────────────────────────────
    "mlflow==2.22.0",
    "wandb==0.20.1",               # from pip-freeze.txt
    "tensorboard==2.18.0",

    # ── numba (Colab ships this) ─────────────────────────────────
    f"numba=={NUMBA_VER}",
    f"llvmlite=={LLVMLITE_VER}",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 4 — COMPUTER VISION
# ════════════════════════════════════════════════════════════════════════

CV = [
    "opencv-python-headless==4.11.0.86",  # from pip-freeze.txt (4.11.0.86)
    "albumentations==2.0.7",
    "imageio==2.37.0",
    "scikit-image==0.25.2",        # from pip-freeze.txt
    "timm==1.0.15",
    "torchvision==0.20.1",

    # Detection / segmentation
    "ultralytics==8.3.100",        # YOLOv8/v11
    "supervision==0.25.1",

    # Generative / diffusion
    "diffusers==0.33.1",
    "accelerate==1.6.0",           # from pip-freeze.txt

    # OCR
    "pytesseract==0.3.13",
    "easyocr==1.7.2",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 5 — NLP / TEXT
# ════════════════════════════════════════════════════════════════════════

NLP = [
    # ── From Colab pip-freeze.txt ────────────────────────────────
    "transformers==4.52.4",        # from pip-freeze.txt
    "tokenizers==0.21.1",          # from pip-freeze.txt
    "datasets==3.6.0",
    "accelerate==1.6.0",
    "huggingface-hub==0.31.1",
    "sentence-transformers==3.4.1",
    "sentencepiece==0.2.0",        # from pip-freeze.txt
    "safetensors==0.5.3",          # from pip-freeze.txt

    # ── Classic NLP ──────────────────────────────────────────────
    "nltk==3.9.1",
    "spacy==3.7.5",                # from pip-freeze.txt
    "gensim==4.3.3",
    "textblob==0.18.0",
    "sacremoses==0.1.1",
    "beautifulsoup4==4.13.4",
    "lxml==5.3.1",
    "ftfy==6.3.1",
    "langdetect==1.0.9",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 6 — AUDIO / SPEECH
# ════════════════════════════════════════════════════════════════════════

AUDIO = [
    # ── From Colab pip-freeze.txt ────────────────────────────────
    "soundfile==0.12.1",           # from pip-freeze.txt
    "soxr==0.5.0.post1",           # from pip-freeze.txt
    "librosa==0.10.2",

    # ── Audio processing ─────────────────────────────────────────
    "audioread==3.0.1",
    "pydub==0.25.1",
    "noisereduce==3.0.3",
    "resampy==0.4.3",

    # ── Feature extraction ────────────────────────────────────────
    "python-speech-features==0.6",
    "opensmile==2.5.0",
    "pyworld==0.3.4",
    "praat-parselmouth==0.4.3",

    # ── Speech models ─────────────────────────────────────────────
    "speechbrain==1.0.2",

    # ── I/O / metadata ────────────────────────────────────────────
    "sounddevice==0.5.1",
    "mutagen==1.47.0",
    "ffmpeg-python==0.2.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 7 — REINFORCEMENT LEARNING
# ════════════════════════════════════════════════════════════════════════

RL = [
    "gymnasium==1.1.1",
    "stable-baselines3==2.6.0",
    "sb3-contrib==2.6.0",
    "ale-py==0.10.2",
    "box2d-py==2.3.5",
    "pygame==2.6.1",
    "shimmy==2.0.0",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 8 — LLMs / GENERATIVE AI
# ════════════════════════════════════════════════════════════════════════

LLM = [
    # ── From Colab pip-freeze.txt ────────────────────────────────
    "transformers==4.52.4",
    "accelerate==1.6.0",
    "huggingface-hub==0.31.1",
    "safetensors==0.5.3",
    "sentencepiece==0.2.0",

    # ── Fine-tuning / quantisation ───────────────────────────────
    "peft==0.15.2",
    "bitsandbytes==0.45.5",
    "trl==0.17.0",
    "datasets==3.6.0",

    # ── LangChain ecosystem ──────────────────────────────────────
    "langchain==0.3.25",
    "langchain-community==0.3.25",
    "langchain-core==0.3.60",
    "openai==1.86.0",              # from pip-freeze.txt
    "anthropic==0.52.0",
    "tiktoken==0.9.0",
    "google-genai==1.20.0",        # from pip-freeze.txt — Colab ships this

    # ── Vector stores ─────────────────────────────────────────────
    "faiss-cpu==1.10.0",
    "chromadb==1.0.7",
    "qdrant-client==1.14.2",

    # ── Embeddings ────────────────────────────────────────────────
    "sentence-transformers==3.4.1",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 9 — TIME SERIES
# ════════════════════════════════════════════════════════════════════════

TIMESERIES = [
    "statsmodels==0.14.4",
    "prophet==1.1.6",
    "pmdarima==2.0.4",
    "sktime==0.35.0",
    "darts==0.32.0",
    "tsfresh==0.21.0",
    "tslearn==0.6.3",
    "pyod==2.0.5",
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 10 — JUPYTER / DEV
#  Colab ships all of these — exact versions from pip-freeze.txt
# ════════════════════════════════════════════════════════════════════════

JUPYTER = [
    "jupyterlab==4.4.2",
    "notebook==7.4.2",
    "ipywidgets==8.1.5",
    "ipykernel==6.29.5",
    "nbformat==5.10.4",
    "nbconvert==7.16.6",
    "ipython==8.34.0",
    "black==25.1.0",
    "isort==5.13.2",
    "ruff==0.11.12",               # from pip-freeze.txt — Colab ships ruff
]

# ════════════════════════════════════════════════════════════════════════
#  GROUP 11 — GENAI / DATA SCIENCE EXTRAS  (new Colab additions 2024-25)
# ════════════════════════════════════════════════════════════════════════

EXTRAS = [
    # ── Colab now ships these by default ─────────────────────────
    "gradio==5.31.0",              # from pip-freeze.txt
    "google-genai==1.20.0",        # from pip-freeze.txt
    "duckdb==1.2.2",               # from pip-freeze.txt
    "polars==1.21.0",              # from pip-freeze.txt
    "narwhals==1.42.0",            # from pip-freeze.txt
    "geopandas==1.0.1",            # from pip-freeze.txt
    "pymc==5.23.0",                # from pip-freeze.txt
    "aesara==2.9.4",
    "xarray==2025.3.1",            # from pip-freeze.txt
    "zarr==3.0.8",
]

# ════════════════════════════════════════════════════════════════════════
#  PROFILE → GROUP MAPPING
# ════════════════════════════════════════════════════════════════════════

PROFILE_GROUPS = {
    "full"      : ["core", "ml", "dl_core", "cv", "nlp", "audio",
                   "rl", "llm", "timeseries", "extras", "jupyter"],
    "nlp"       : ["core", "ml", "dl_core", "nlp", "jupyter"],
    "cv"        : ["core", "ml", "dl_core", "cv", "jupyter"],
    "tabular"   : ["core", "ml", "jupyter"],
    "rl"        : ["core", "dl_core", "rl", "jupyter"],
    "audio"     : ["core", "ml", "dl_core", "nlp", "audio", "jupyter"],
    "llm"       : ["core", "dl_core", "nlp", "llm", "jupyter"],
    "timeseries": ["core", "ml", "timeseries", "jupyter"],
    "genai"     : ["core", "dl_core", "nlp", "llm", "extras", "jupyter"],
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
    "extras"    : EXTRAS,
    "jupyter"   : JUPYTER,
}


def get_profile_packages(profile: str) -> dict:
    groups = PROFILE_GROUPS.get(profile, PROFILE_GROUPS["full"])
    return {g: GROUP_MAP[g] for g in groups if g in GROUP_MAP}


def get_flat_list(profile: str) -> list:
    result = []
    for pkgs in get_profile_packages(profile).values():
        result.extend(pkgs)
    seen = set()
    return [p for p in result if not (p in seen or seen.add(p))]
