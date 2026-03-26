# 🧠 Universal DS / ML / AI Environment — Exact Google Colab Mirror

> Versions sourced directly from **`github.com/googlecolab/backend-info`** `pip-freeze.txt`  
> Docker image: **`us-docker.pkg.dev/colab-images/public/runtime`** (GPU) / **`cpu-runtime`** (CPU)

Push this folder to GitHub. Clone anywhere. Run one command. Identical environment to Colab.

---

## 📌 Exact Colab Runtime Snapshot

| Component | Version |
|---|---|
| **Python** | 3.12 |
| **CUDA** | 12.5 |
| **cuDNN** | 9.x |
| **NVIDIA Driver** | 550.54.15 |
| **PyTorch** | 2.5.1+cu121 *(Colab uses cu121 wheel on CUDA 12.5 host)* |
| **TensorFlow** | 2.18.0 |
| **Keras** | 3.8.0 |
| **JAX** | cuda12 *(Colab now ships JAX by default)* |
| **numpy** | 1.26.4 *(pinned < 2.0 — both TF and torch require this)* |
| **scikit-learn** | 1.6.1 |
| **scipy** | 1.15.3 |
| **transformers** | 4.52.4 |
| **opencv** | 4.11.0.86 |
| **wandb** | 0.20.1 |
| **gradio** | 5.31.0 *(Colab now ships this!)* |
| **polars** | 1.21.0 *(Colab now ships this!)* |
| **google-genai** | 1.20.0 *(Colab now ships this!)* |
| **duckdb** | 1.2.2 *(Colab now ships this!)* |
| **ruff** | 0.11.12 *(Colab now ships this!)* |

Source: [`googlecolab/backend-info`](https://github.com/googlecolab/backend-info) · [`colabtools` upgrade announcements](https://github.com/googlecolab/colabtools/issues)

---

## ⚡ Quick Start — One Command

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/colab_mirror

# Run once — auto-detects OS + GPU
python colab_setup.py

# Activate
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Open Jupyter  →  select kernel: Python (colab_mirror)
jupyter lab
```

---

## 🎯 Profiles — Install Only What You Need

```bash
python colab_setup.py                      # full Colab mirror (default)
python colab_setup.py -p nlp               # NLP / text models
python colab_setup.py -p cv                # Computer Vision
python colab_setup.py -p tabular           # DS / Classical ML
python colab_setup.py -p audio             # Audio / Speech
python colab_setup.py -p rl                # Reinforcement Learning
python colab_setup.py -p llm               # LLMs / GenAI / RAG
python colab_setup.py -p timeseries        # Time Series / Forecasting
python colab_setup.py -p genai             # LLMs + Gradio + google-genai + DuckDB
```

---

## 🔧 All Flags

```bash
python colab_setup.py --profile nlp        # choose a profile
python colab_setup.py --cpu-only           # force CPU PyTorch (no GPU)
python colab_setup.py --skip-system        # skip apt/brew system installs
python colab_setup.py --skip-nlp-dl        # skip spaCy/NLTK model downloads
python colab_setup.py --verify-only        # check what's installed
python colab_setup.py --dry-run            # print packages without installing
```

---

## 📁 Folder Structure

```
colab_mirror/
│
├── colab_setup.py           ← 🚀 THE ONE FILE YOU RUN
├── requirements_full.txt    ← pip reference / CI (CPU fallback)
├── environment.yml          ← conda alternative
├── Dockerfile.gpu           ← GPU Docker image (mirrors runtime)
├── Dockerfile.cpu           ← CPU Docker image (mirrors cpu-runtime)
├── docker-compose.yml       ← run GPU + CPU services
├── .gitignore
├── README.md
│
└── setup/                   ← internal modules
    ├── __init__.py
    ├── packages.py          ← ALL versions (edit here to upgrade)
    ├── detect.py            ← OS / GPU / platform detection
    ├── install.py           ← all install steps
    ├── verify.py            ← post-install checks + smoke tests
    └── utils.py             ← logging + pip helpers
```

> **To upgrade a package** → edit `setup/packages.py` only. Everything else auto-adapts.

---

## 🐳 Docker — Run the Exact Colab Image

### GPU (mirrors `us-docker.pkg.dev/colab-images/public/runtime`)

```bash
# Build
docker build -f Dockerfile.gpu -t colab_mirror:gpu .

# Run  (requires NVIDIA Container Toolkit)
docker run --gpus all -p 8888:8888 \
  -v "$PWD/notebooks":/workspace/notebooks \
  colab_mirror:gpu
```

### CPU (mirrors `us-docker.pkg.dev/colab-images/public/cpu-runtime`)

```bash
docker build -f Dockerfile.cpu -t colab_mirror:cpu .

docker run -p 8888:8888 \
  -v "$PWD/notebooks":/workspace/notebooks \
  colab_mirror:cpu
```

### Docker Compose (both at once)

```bash
docker compose up colab-gpu    # GPU runtime → http://localhost:8888
docker compose up colab-cpu    # CPU runtime → http://localhost:8889
```

---

## 🖥️ Platform → Command

| Platform | Command |
|---|---|
| Local Linux / macOS | `python colab_setup.py` |
| Local Windows | `python colab_setup.py` |
| AWS EC2 (GPU) | `python colab_setup.py` |
| AWS SageMaker | `python colab_setup.py --skip-system` |
| Lightning AI | `python colab_setup.py` |
| RunPod / Vast.ai | `python colab_setup.py` |
| Databricks | `python colab_setup.py --skip-system` |
| Kaggle | `python colab_setup.py --skip-system --skip-nlp-dl` |
| Paperspace | `python colab_setup.py` |
| GCP Vertex AI | `python colab_setup.py --skip-system` |
| Azure ML | `python colab_setup.py --skip-system` |
| Conda | `conda env create -f environment.yml` then `python colab_setup.py --skip-system` |
| Docker GPU | `docker compose up colab-gpu` |
| Docker CPU | `docker compose up colab-cpu` |

---

## 🔋 Hardware Auto-Detection

| Hardware | What gets installed |
|---|---|
| NVIDIA CUDA 12.x | `torch==2.5.1+cu121` ← same wheel Colab uses |
| NVIDIA CUDA 11.x | `torch==2.5.1+cu118` |
| Apple Silicon M1/M2/M3 | Standard PyTorch + MPS |
| CPU only | `torch==2.5.1+cpu` |
| GPU: JAX | `jax[cuda12]` |
| CPU: JAX | `jax[cpu]` |

---

## 🔒 Why These Exact Versions?

### The numpy problem (most common cross-framework crash)
```
TensorFlow 2.18 installs  → tries to upgrade numpy to 2.x
PyTorch 2.5.1              → breaks if numpy ≥ 2.0
```
**Fix applied:** numpy `1.26.4` is pinned first, then **re-pinned again** immediately after TF installs. This is the same constraint Google enforces in the Colab Docker image.

### PyTorch uses cu121 wheels on a CUDA 12.5 host
Colab runs CUDA 12.5 but deliberately installs **PyTorch cu121 wheels** — they are more stable and widely tested. This setup replicates that exact choice.

### numba + llvmlite must be a matched pair
```
numba==0.60.0   ←→   llvmlite==0.43.0
```
Never change one without the other. Mismatched versions cause silent JIT crashes in librosa, UMAP, etc.

---

## 📦 What Each Profile Installs

| Profile | Key packages |
|---|---|
| **core** *(all)* | numpy · pandas · scipy · matplotlib · polars · plotly · tqdm |
| **tabular** | scikit-learn 1.6.1 · XGBoost · LightGBM · CatBoost · SHAP · Optuna · UMAP |
| **nlp** | transformers 4.52.4 · tokenizers · datasets · sentence-transformers · spaCy · NLTK |
| **cv** | opencv 4.11 · albumentations · YOLOv8 · diffusers · EasyOCR · timm |
| **audio** | torchaudio · librosa · soundfile · pydub · speechbrain · opensmile |
| **rl** | gymnasium 1.1.1 · stable-baselines3 2.6 · sb3-contrib · ALE · Box2D |
| **llm** | PEFT · LoRA · bitsandbytes · TRL · LangChain · OpenAI · google-genai · faiss · chromadb |
| **timeseries** | Prophet · pmdarima · sktime · darts · tsfresh · PyOD |
| **genai** | llm + gradio · google-genai · duckdb · geopandas · pymc |
| **dl_core** *(all DL)* | torch 2.5.1 · TF 2.18 · JAX · Keras 3.8 · Lightning · timm · wandb · MLflow |

---

## ❓ Troubleshooting

### Verify the full environment
```bash
python colab_setup.py --verify-only
```

### Preview what would install (no changes made)
```bash
python colab_setup.py --dry-run --profile llm
```

### numpy conflict after TF install
```bash
pip install numpy==1.26.4 --force-reinstall
```

### `torch.cuda.is_available()` → False
```bash
nvidia-smi                           # is GPU visible to OS?
python colab_setup.py --verify-only  # check torch version + build
python colab_setup.py                # reinstall with correct CUDA wheel
```

### Audio: `.ogg` / `.mp3` fails outside Colab
```bash
# Linux: reinstall codec libs
sudo apt-get install -y ffmpeg libsox-fmt-all libsndfile1
python -c "import torchaudio; print(torchaudio.get_audio_backend())"
```

### Clean reinstall
```bash
rm -rf venv              # Linux / macOS
rmdir /s /q venv         # Windows
python colab_setup.py
```

### Restricted machines (Kaggle, SageMaker, Databricks)
```bash
python colab_setup.py --skip-system --skip-nlp-dl -p nlp
```
