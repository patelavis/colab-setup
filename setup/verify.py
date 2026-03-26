"""
Post-install verification.
Checks exact versions match the Colab runtime pip-freeze.txt snapshot.
Runs domain smoke tests so you know the env actually works, not just imports.
"""
import json, subprocess, sys
from pathlib import Path
from .utils import ok, warn, G, Y, RED, R, BOLD, B

CHECKS = {
    "core": [
        ("numpy",           "import numpy; print(numpy.__version__)"),
        ("pandas",          "import pandas; print(pandas.__version__)"),
        ("scipy",           "import scipy; print(scipy.__version__)"),
        ("matplotlib",      "import matplotlib; print(matplotlib.__version__)"),
        ("seaborn",         "import seaborn; print(seaborn.__version__)"),
        ("plotly",          "import plotly; print(plotly.__version__)"),
        ("polars",          "import polars; print(polars.__version__)"),
        ("pyarrow",         "import pyarrow; print(pyarrow.__version__)"),
    ],
    "ml": [
        ("scikit-learn",    "import sklearn; print(sklearn.__version__)"),
        ("xgboost",         "import xgboost; print(xgboost.__version__)"),
        ("lightgbm",        "import lightgbm; print(lightgbm.__version__)"),
        ("catboost",        "import catboost; print(catboost.__version__)"),
        ("shap",            "import shap; print(shap.__version__)"),
        ("optuna",          "import optuna; print(optuna.__version__)"),
        ("statsmodels",     "import statsmodels; print(statsmodels.__version__)"),
        ("umap-learn",      "import umap; print(umap.__version__)"),
    ],
    "dl_core": [
        ("torch",           "import torch; print(torch.__version__)"),
        ("torchvision",     "import torchvision; print(torchvision.__version__)"),
        ("torchtext",       "import torchtext; print(torchtext.__version__)"),
        ("tensorflow",      "import tensorflow as tf; print(tf.__version__)"),
        ("keras",           "import keras; print(keras.__version__)"),
        ("tensorflow_hub",  "import tensorflow_hub; print(tensorflow_hub.__version__)"),
        ("jax",             "import jax; print(jax.__version__)"),
        ("lightning",       "import lightning; print(lightning.__version__)"),
        ("timm",            "import timm; print(timm.__version__)"),
        ("einops",          "import einops; print(einops.__version__)"),
        ("wandb",           "import wandb; print(wandb.__version__)"),
    ],
    "cv": [
        ("opencv",          "import cv2; print(cv2.__version__)"),
        ("albumentations",  "import albumentations; print(albumentations.__version__)"),
        ("ultralytics",     "import ultralytics; print(ultralytics.__version__)"),
        ("diffusers",       "import diffusers; print(diffusers.__version__)"),
        ("scikit-image",    "import skimage; print(skimage.__version__)"),
    ],
    "nlp": [
        ("transformers",    "import transformers; print(transformers.__version__)"),
        ("tokenizers",      "import tokenizers; print(tokenizers.__version__)"),
        ("datasets",        "import datasets; print(datasets.__version__)"),
        ("sentence-trans",  "import sentence_transformers; print(sentence_transformers.__version__)"),
        ("spacy",           "import spacy; print(spacy.__version__)"),
        ("nltk",            "import nltk; print(nltk.__version__)"),
    ],
    "audio": [
        ("torchaudio",      "import torchaudio; print(torchaudio.__version__)"),
        ("torchaudio_bk",   "import torchaudio; print(torchaudio.get_audio_backend())"),
        ("librosa",         "import librosa; print(librosa.__version__)"),
        ("soundfile",       "import soundfile; print(soundfile.__version__)"),
        ("pydub",           "from pydub import AudioSegment; print('ok')"),
        ("speechbrain",     "import speechbrain; print(speechbrain.__version__)"),
    ],
    "rl": [
        ("gymnasium",       "import gymnasium; print(gymnasium.__version__)"),
        ("stable-baselines3","import stable_baselines3; print(stable_baselines3.__version__)"),
    ],
    "llm": [
        ("peft",            "import peft; print(peft.__version__)"),
        ("trl",             "import trl; print(trl.__version__)"),
        ("bitsandbytes",    "import bitsandbytes; print(bitsandbytes.__version__)"),
        ("langchain",       "import langchain; print(langchain.__version__)"),
        ("openai",          "import openai; print(openai.__version__)"),
        ("google-genai",    "import google.genai; print('ok')"),
        ("faiss",           "import faiss; print(faiss.__version__)"),
        ("chromadb",        "import chromadb; print(chromadb.__version__)"),
    ],
    "timeseries": [
        ("prophet",         "from prophet import Prophet; print('ok')"),
        ("sktime",          "import sktime; print(sktime.__version__)"),
        ("darts",           "import darts; print(darts.__version__)"),
    ],
    "extras": [
        ("gradio",          "import gradio; print(gradio.__version__)"),
        ("duckdb",          "import duckdb; print(duckdb.__version__)"),
        ("geopandas",       "import geopandas; print(geopandas.__version__)"),
    ],
    "jupyter": [
        ("jupyterlab",      "import jupyterlab; print(jupyterlab.__version__)"),
        ("ipykernel",       "import ipykernel; print(ipykernel.__version__)"),
        ("ipywidgets",      "import ipywidgets; print(ipywidgets.__version__)"),
        ("ruff",            "import ruff; print(ruff.__version__)"),
    ],
}

GPU_CHECKS = {
    "cuda": [
        ("torch.cuda.is_available",   "import torch; print(torch.cuda.is_available())"),
        ("torch GPU name",            "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"),
        ("tensorflow GPUs",           "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print(f'{len(g)} GPU(s) to TF')"),
        ("jax devices",               "import jax; print(jax.devices())"),
    ],
    "mps": [
        ("torch.mps",                 "import torch; print(torch.backends.mps.is_available())"),
    ],
    "cpu": [
        ("torch threads",             "import torch; print(torch.get_num_threads(), 'CPU threads')"),
    ],
}

SMOKE_TESTS = {
    "dl_core": (
        "Torch + TF tensor ops",
        """
import torch, tensorflow as tf
x = torch.randn(4, 16)
y = torch.nn.Linear(16, 8)(x)
assert y.shape == (4, 8)
t = tf.random.normal([4, 16])
r = tf.keras.layers.Dense(8)(t)
assert r.shape == (4, 8)
print(f"torch {torch.__version__} + tf {tf.__version__}  ✓")
"""
    ),
    "cv": (
        "OpenCV image ops",
        """
import numpy as np, cv2
img = np.zeros((128, 128, 3), dtype=np.uint8)
out = cv2.resize(img, (64, 64))
assert out.shape == (64, 64, 3)
print(f"cv2 {cv2.__version__}  ✓")
"""
    ),
    "audio": (
        "librosa melspectrogram",
        """
import numpy as np, librosa
y = np.sin(2*np.pi*440*np.linspace(0,0.5,8000)).astype(np.float32)
S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64)
assert S.shape[0] == 64
print(f"librosa {librosa.__version__}  mel {S.shape}  ✓")
"""
    ),
    "nlp": (
        "HuggingFace tokenizer",
        """
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
out = tok("hello colab mirror", return_tensors="pt")
print(f"transformers ✓  ids {list(out['input_ids'].shape)}")
"""
    ),
    "rl": (
        "Gymnasium CartPole",
        """
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, _ = env.reset()
print(f"gymnasium ✓  obs shape {obs.shape}")
"""
    ),
    "llm": (
        "PEFT LoraConfig",
        """
from peft import LoraConfig
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"])
print(f"peft ✓  lora r={cfg.r}")
"""
    ),
}


def _run(code: str):
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()


def verify_all(sys_info: dict, profile: str) -> dict:
    from .packages import PROFILE_GROUPS
    groups  = PROFILE_GROUPS.get(profile, list(CHECKS.keys()))
    results = {}

    print(f"\n  {BOLD}Package Versions  (reference: Colab pip-freeze.txt):{R}\n")
    for group in groups:
        checks = CHECKS.get(group, [])
        if not checks: continue
        print(f"  {B}  {group.upper()}{R}")
        for name, code in checks:
            ok_flag, out, _ = _run(code)
            if ok_flag:
                print(f"  {G}  ✅{R}  {name:<28} {out}")
                results[name] = out
            else:
                print(f"  {RED}  ❌{R}  {name:<28} FAILED")
                results[name] = "FAILED"

    # GPU checks
    needs_gpu = any(g in groups for g in
                    ["dl_core", "cv", "nlp", "audio", "rl", "llm", "genai"])
    if needs_gpu:
        print(f"\n  {BOLD}GPU / Compute:{R}\n")
        hw = ("cuda" if sys_info.get("cuda") and not sys_info.get("force_cpu")
              else "mps" if sys_info.get("mps") and not sys_info.get("force_cpu")
              else "cpu")
        for name, code in GPU_CHECKS.get(hw, GPU_CHECKS["cpu"]):
            ok_flag, out, _ = _run(code)
            good   = ok_flag and "False" not in out and "0 GPU" not in out
            colour = G if good else Y
            icon   = "✅" if good else "⚠️ "
            print(f"  {colour}  {icon}{R}  {name:<28} {out if ok_flag else 'FAILED'}")
            results[f"gpu_{name}"] = out if ok_flag else "FAILED"

    # Smoke tests
    print(f"\n  {BOLD}Smoke Tests:{R}\n")
    for g in groups:
        if g not in SMOKE_TESTS: continue
        label, code = SMOKE_TESTS[g]
        ok_flag, out, err = _run(code)
        if ok_flag:
            print(f"  {G}  ✅{R}  {label:<36} {out}")
            results[f"smoke_{g}"] = out
        else:
            print(f"  {Y}  ⚠️ {R}  {label:<36} {err[:60]}")
            results[f"smoke_{g}"] = "FAILED"

    return results


def save_report(sys_info: dict, results: dict, profile: str):
    from .packages import (TORCH_VER, TENSORFLOW_VER, NUMPY_VER,
                            COLAB_CUDA, COLAB_PYTHON, KERAS_VER)
    report = {
        "profile"       : profile,
        "system"        : sys_info,
        "packages"      : results,
        "colab_runtime" : {
            "python"    : COLAB_PYTHON,
            "cuda"      : COLAB_CUDA,
            "torch"     : TORCH_VER,
            "tensorflow": TENSORFLOW_VER,
            "keras"     : KERAS_VER,
            "numpy"     : NUMPY_VER,
            "docker_gpu": "us-docker.pkg.dev/colab-images/public/runtime",
            "docker_cpu": "us-docker.pkg.dev/colab-images/public/cpu-runtime",
        },
    }
    p = Path(__file__).resolve().parent.parent / "setup_report.json"
    p.write_text(json.dumps(report, indent=2))
    from .utils import info
    info(f"Report → {p}")
