# 🎵 Audio ML Environment — Google Colab Mirror

> Push this folder to your GitHub. Clone it anywhere. Run **one command**. Your notebook works.

Works on: **Linux · macOS · Windows · AWS EC2 · RunPod · Lightning AI · Databricks · Kaggle · Paperspace**

---

## ⚡ Quick Start

```bash
# Clone your repo, enter the env folder
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/audio_ml_env

# Run once — detects your OS + GPU automatically
python colab_setup.py
```

That's it. Then open Jupyter and pick kernel **`Python (audio_ml — Colab)`**.

---

## 📁 Folder Structure

```
audio_ml_env/
│
├── colab_setup.py          ← 🚀 THE ONE FILE YOU RUN  (entry point)
├── requirements.txt        ← pip reference / CI installs
├── environment.yml         ← conda alternative
├── .gitignore
├── README.md
│
└── setup/                  ← internal modules (don't edit unless needed)
    ├── __init__.py
    ├── packages.py         ← ALL version pins in one place
    ├── detect.py           ← OS, GPU, runtime env detection
    ├── install.py          ← installation steps
    ├── verify.py           ← post-install checks + audio smoke test
    └── utils.py            ← shared logging / pip helpers
```

---

## 🖥️ Platform → Command

| Platform | Command |
|---|---|
| Local Linux / macOS | `python colab_setup.py` |
| Local Windows (CMD) | `python colab_setup.py` |
| AWS EC2 | `python colab_setup.py` |
| Lightning AI | `python colab_setup.py` |
| RunPod / Vast.ai | `python colab_setup.py` |
| Databricks | `python colab_setup.py` |
| Kaggle | `python colab_setup.py --skip-system` |
| Conda env | `conda env create -f environment.yml` then `python colab_setup.py --skip-system` |

---

## 🔧 Command-Line Flags

```bash
python colab_setup.py                  # full setup (default)
python colab_setup.py --cpu-only       # force CPU PyTorch even if GPU found
python colab_setup.py --skip-system    # skip apt/brew system package install
python colab_setup.py --skip-nlp       # skip spaCy + NLTK model downloads
python colab_setup.py --verify-only    # just check what's installed, no installs
```

---

## 🔋 Hardware Auto-Detection

| Hardware | What gets installed |
|---|---|
| NVIDIA GPU (CUDA 12.x) | `torch==2.3.1+cu121` wheels |
| NVIDIA GPU (CUDA 11.x) | `torch==2.3.1+cu118` wheels |
| Apple Silicon M1/M2/M3 | Standard PyTorch with MPS enabled |
| CPU only | `torch==2.3.1+cpu` (lighter build) |

Your notebook code — `device = "cuda" if torch.cuda.is_available() else "cpu"` — works unchanged everywhere.

---

## 🎵 Audio Stack — Supported Formats

| Format | How it loads |
|---|---|
| `.wav` | `torchaudio.load()` → soundfile backend |
| `.ogg` | `torchaudio.load()` → ffmpeg backend |
| `.mp3` | `torchaudio.load()` → ffmpeg / audioread backend |
| `.flac` | `torchaudio.load()` → soundfile backend |
| `.aiff` | `torchaudio.load()` → soundfile/ffmpeg |

```python
# This works on ALL platforms after setup
import torchaudio
waveform, sr = torchaudio.load("audio.ogg")   # .wav .ogg .mp3 .flac
```

---

## 📦 What's Installed

### PyTorch Ecosystem
`torch 2.3.1` · `torchaudio` · `torchvision` · `torchtext` · `torchcodec` (Linux) · `torchmetrics` · `pytorch-lightning` · `timm` · `einops`

### TensorFlow Ecosystem
`tensorflow 2.16.1` · `keras` · `tensorflow-hub` · `tensorflow-datasets` · `tensorflow-addons`

### Audio Signal Processing
`librosa` · `soundfile` · `pydub` · `audioread` · `noisereduce` · `resampy` · `soxr` · `opensmile` · `python-speech-features` · `pyworld` · `praat-parselmouth` · `speechbrain` · `ffmpeg-python`

### NLP / Text Models
`transformers` (Wav2Vec2, Whisper, BERT, HuBERT) · `tokenizers` · `datasets` · `accelerate` · `sentence-transformers` · `spaCy` · `NLTK` · `sentencepiece` · `gensim`

### Computer Vision (Spectrogram → Image)
`opencv` · `albumentations` · `Pillow` · `scikit-image`

### Classical ML
`scikit-learn` · `xgboost` · `lightgbm` · `imbalanced-learn`

### Experiment Tracking
`mlflow` · `wandb` · `tensorboard` · `plotly` · `seaborn`

---

## 🔒 Version Compatibility Matrix

The most common failure across PyTorch + TensorFlow environments is **numpy**. TF upgrades it to 2.x, which breaks PyTorch. This setup handles it:

```
numpy  == 1.26.4   ← pinned before install, re-pinned after TF
torch  == 2.3.1    ← tested to coexist with TF 2.16.1
tensorflow == 2.16.1
numba  == 0.60.0   ← paired exactly with llvmlite 0.43.0
llvmlite == 0.43.0
```

To change versions, edit **`setup/packages.py`** only — all other files auto-adapt.

---

## 🔁 Daily Use

```bash
# After setup is done, activate env each session
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# Launch Jupyter
jupyter lab
# Select kernel: Python (audio_ml — Colab)

# Or run a script directly
python your_notebook_converted.py
```

---

## ❓ Troubleshooting

### `.ogg` / `.mp3` fails with "no audio backend"
```bash
# Linux — reinstall ffmpeg
sudo apt-get install -y ffmpeg libsox-fmt-all
python -c "import torchaudio; print(torchaudio.get_audio_backend())"
```

### `torch.cuda.is_available()` returns False
```bash
# Check drivers
nvidia-smi
# Reinstall with correct CUDA version
python colab_setup.py --verify-only   # see what's installed
python colab_setup.py                 # reinstall
```

### numpy conflict after install
```bash
pip install numpy==1.26.4 --force-reinstall
```

### Clean reinstall
```bash
rm -rf venv                     # Linux / macOS
rmdir /s /q venv                # Windows
python colab_setup.py
```

### Verify everything is working
```bash
python colab_setup.py --verify-only
```
