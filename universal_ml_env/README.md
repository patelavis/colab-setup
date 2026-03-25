# 🧠 Universal DS / ML / AI Environment — Google Colab Mirror

> Push once to GitHub. Clone anywhere. Run one command. Every package works.

**OS:** Linux · macOS · Windows · EC2 · RunPod · Lightning AI · Databricks · Kaggle · Paperspace · GCP · Azure  
**Hardware:** NVIDIA GPU (CUDA 11/12) · Apple Silicon MPS · CPU-only  
**Profiles:** NLP · Computer Vision · Tabular · Reinforcement Learning · Audio · LLMs · Time Series · Full

---

## ⚡ Quick Start

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/universal_ml_env

# 2. Run once — auto-detects OS + GPU
python colab_setup.py

# 3. Activate + use
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
jupyter lab                   # kernel: Python (ml_env — Colab)
```

---

## 🎯 Profiles — Install Only What You Need

```bash
python colab_setup.py --profile full        # everything (default)
python colab_setup.py --profile nlp         # NLP / Text models
python colab_setup.py --profile cv          # Computer Vision
python colab_setup.py --profile tabular     # DS / Classical ML / EDA
python colab_setup.py --profile rl          # Reinforcement Learning
python colab_setup.py --profile audio       # Audio / Speech
python colab_setup.py --profile llm         # LLMs / GenAI / RAG
python colab_setup.py --profile timeseries  # Time Series / Forecasting
```

Each profile installs only its packages. Use `full` to mirror Colab exactly.

---

## 🔧 All Flags

```bash
python colab_setup.py --profile nlp         # choose a profile
python colab_setup.py --cpu-only            # force CPU PyTorch (no GPU)
python colab_setup.py --skip-system         # skip apt/brew installs
python colab_setup.py --skip-nlp-dl         # skip spaCy/NLTK downloads
python colab_setup.py --verify-only         # check what's installed
python colab_setup.py --dry-run             # print packages without installing
```

---

## 📁 Folder Structure

```
universal_ml_env/
│
├── colab_setup.py            ← 🚀 THE ONE FILE YOU RUN
├── requirements_full.txt     ← pip reference / CI
├── environment.yml           ← conda alternative
├── .gitignore
├── README.md
│
└── setup/                    ← internal modules
    ├── __init__.py
    ├── packages.py           ← ALL versions + profile→group mapping
    ├── detect.py             ← OS / GPU / platform detection
    ├── install.py            ← all install steps
    ├── verify.py             ← post-install checks + smoke tests
    └── utils.py              ← logging + pip helpers
```

To upgrade a package → **edit `setup/packages.py` only**. Everything else auto-adapts.

---

## 📦 What Each Profile Installs

| Profile | Packages |
|---|---|
| **core** (all profiles) | numpy · pandas · scipy · matplotlib · seaborn · plotly · tqdm · rich |
| **tabular** | scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · Optuna · statsmodels · feature-engine |
| **nlp** | transformers · tokenizers · datasets · sentence-transformers · spaCy · NLTK · gensim · sentencepiece |
| **cv** | OpenCV · albumentations · timm · YOLOv8 (ultralytics) · diffusers · scikit-image · EasyOCR |
| **audio** | torchaudio · librosa · soundfile · pydub · speechbrain · opensmile · pyworld |
| **rl** | gymnasium · stable-baselines3 · sb3-contrib · ALE (Atari) · Box2D |
| **llm** | PEFT · LoRA · bitsandbytes · TRL · LangChain · OpenAI API · faiss · chromadb |
| **timeseries** | Prophet · pmdarima · sktime · darts · tsfresh · tslearn · PyOD |
| **dl_core** (all DL profiles) | PyTorch + TF + Keras + TF-Hub · Lightning · timm · einops · MLflow · wandb |

---

## 🔋 Hardware Detection

| Hardware | What gets installed |
|---|---|
| NVIDIA CUDA 12.x | `torch==2.3.1+cu121` |
| NVIDIA CUDA 11.x | `torch==2.3.1+cu118` |
| Apple Silicon M1/M2/M3 | Standard PyTorch with MPS |
| CPU only | `torch==2.3.1+cpu` |

Your notebook code — `device = "cuda" if torch.cuda.is_available() else "cpu"` — works unchanged everywhere.

---

## 🖥️ Platform → Command

| Platform | Command |
|---|---|
| Local Linux / macOS | `python colab_setup.py` |
| Local Windows | `python colab_setup.py` |
| AWS EC2 | `python colab_setup.py` |
| AWS SageMaker | `python colab_setup.py --skip-system` |
| Lightning AI | `python colab_setup.py` |
| RunPod / Vast.ai | `python colab_setup.py` |
| Databricks | `python colab_setup.py --skip-system` |
| Kaggle | `python colab_setup.py --skip-system --skip-nlp-dl` |
| Paperspace | `python colab_setup.py` |
| Conda env | `conda env create -f environment.yml` then `python colab_setup.py --skip-system` |
| Docker | `RUN python colab_setup.py --skip-system` |

---

## 🔒 Version Compatibility — Why Pins Matter

The most common failure when running PyTorch + TensorFlow together:

| Problem | Cause | Fix applied here |
|---|---|---|
| `numpy` conflict | TF upgrades numpy to 2.x, breaking torch | numpy pinned to `1.26.4`, re-pinned after TF install |
| torch/audio/vision version mismatch | Minor versions must match | All pinned to `2.3.x` |
| `numba` crash | numba + llvmlite must be exact pair | Both pinned together |
| CUDA torch not using GPU | Wrong wheel index | Auto-detected by CUDA version |

To upgrade: change versions in **`setup/packages.py`** only.

---

## 🔁 Daily Use

```bash
# Activate
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Jupyter (select kernel: Python (ml_env — Colab))
jupyter lab

# Or run a script
python train.py
```

---

## ❓ Troubleshooting

### Verify everything
```bash
python colab_setup.py --verify-only
```

### See what would install (no changes)
```bash
python colab_setup.py --dry-run --profile nlp
```

### numpy conflict after TF install
```bash
pip install numpy==1.26.4 --force-reinstall
```

### torch.cuda.is_available() → False
```bash
nvidia-smi                          # check GPU is visible
python colab_setup.py --verify-only # see installed torch version
python colab_setup.py               # reinstall with correct CUDA wheel
```

### Clean reinstall
```bash
rm -rf venv          # Linux/macOS
rmdir /s /q venv     # Windows
python colab_setup.py
```

### Skip slow downloads on restricted machines
```bash
python colab_setup.py --skip-system --skip-nlp-dl
```
