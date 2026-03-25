"""
Post-install verification. Checks imports + versions for all installed
profile groups. Runs a quick smoke test for each domain.
"""
import json, subprocess, sys
from pathlib import Path
from .utils import ok, warn, info, G, Y, RED, R, BOLD, B, DIM

# ── Checks per group: (display_name, python_snippet) ────────────────

CHECKS = {
    "core": [
        ("numpy",           "import numpy; print(numpy.__version__)"),
        ("pandas",          "import pandas; print(pandas.__version__)"),
        ("scipy",           "import scipy; print(scipy.__version__)"),
        ("matplotlib",      "import matplotlib; print(matplotlib.__version__)"),
        ("seaborn",         "import seaborn; print(seaborn.__version__)"),
        ("plotly",          "import plotly; print(plotly.__version__)"),
        ("tqdm",            "import tqdm; print(tqdm.__version__)"),
    ],
    "ml": [
        ("scikit-learn",    "import sklearn; print(sklearn.__version__)"),
        ("xgboost",         "import xgboost; print(xgboost.__version__)"),
        ("lightgbm",        "import lightgbm; print(lightgbm.__version__)"),
        ("catboost",        "import catboost; print(catboost.__version__)"),
        ("shap",            "import shap; print(shap.__version__)"),
        ("optuna",          "import optuna; print(optuna.__version__)"),
        ("statsmodels",     "import statsmodels; print(statsmodels.__version__)"),
    ],
    "dl_core": [
        ("torch",           "import torch; print(torch.__version__)"),
        ("torchvision",     "import torchvision; print(torchvision.__version__)"),
        ("torchtext",       "import torchtext; print(torchtext.__version__)"),
        ("tensorflow",      "import tensorflow as tf; print(tf.__version__)"),
        ("keras",           "import keras; print(keras.__version__)"),
        ("tensorflow_hub",  "import tensorflow_hub; print(tensorflow_hub.__version__)"),
        ("lightning",       "import lightning; print(lightning.__version__)"),
        ("timm",            "import timm; print(timm.__version__)"),
        ("einops",          "import einops; print(einops.__version__)"),
        ("torchmetrics",    "import torchmetrics; print(torchmetrics.__version__)"),
        ("mlflow",          "import mlflow; print(mlflow.__version__)"),
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
        ("gensim",          "import gensim; print(gensim.__version__)"),
    ],
    "audio": [
        ("torchaudio",      "import torchaudio; print(torchaudio.__version__)"),
        ("torchaudio bk",   "import torchaudio; print(torchaudio.get_audio_backend())"),
        ("librosa",         "import librosa; print(librosa.__version__)"),
        ("soundfile",       "import soundfile; print(soundfile.__version__)"),
        ("pydub",           "from pydub import AudioSegment; print('ok')"),
        ("speechbrain",     "import speechbrain; print(speechbrain.__version__)"),
        ("opensmile",       "import opensmile; print(opensmile.__version__)"),
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
        ("faiss-cpu",       "import faiss; print(faiss.__version__)"),
        ("chromadb",        "import chromadb; print(chromadb.__version__)"),
    ],
    "timeseries": [
        ("prophet",         "from prophet import Prophet; print('ok')"),
        ("pmdarima",        "import pmdarima; print(pmdarima.__version__)"),
        ("sktime",          "import sktime; print(sktime.__version__)"),
        ("darts",           "import darts; print(darts.__version__)"),
        ("tsfresh",         "import tsfresh; print(tsfresh.__version__)"),
        ("pyod",            "import pyod; print(pyod.__version__)"),
    ],
    "jupyter": [
        ("jupyterlab",      "import jupyterlab; print(jupyterlab.__version__)"),
        ("ipykernel",       "import ipykernel; print(ipykernel.__version__)"),
        ("ipywidgets",      "import ipywidgets; print(ipywidgets.__version__)"),
    ],
}

GPU_CHECKS = {
    "cuda": [
        ("torch.cuda",          "import torch; print(torch.cuda.is_available())"),
        ("torch GPU name",      "import torch; g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(g)"),
        ("tf GPUs",             "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print(f'{len(g)} GPU(s)')"),
    ],
    "mps": [
        ("torch.mps",           "import torch; print(torch.backends.mps.is_available())"),
    ],
    "cpu": [
        ("torch threads",       "import torch; print(torch.get_num_threads(), 'threads')"),
    ],
}

SMOKE_TESTS = {
    "dl_core": (
        "DL smoke (tensor ops)",
        """
import torch, numpy as np
x = torch.randn(4, 8)
y = torch.nn.Linear(8, 4)(x)
assert y.shape == (4, 4)
print(f"torch ok — shape {list(y.shape)}")
"""
    ),
    "cv": (
        "CV smoke (image resize)",
        """
import numpy as np, cv2
img = np.zeros((100, 100, 3), dtype=np.uint8)
out = cv2.resize(img, (50, 50))
assert out.shape == (50, 50, 3)
print("cv2 ok — resize 100→50")
"""
    ),
    "audio": (
        "Audio smoke (mel spectrogram)",
        """
import numpy as np, librosa
y = np.sin(2*np.pi*440*np.linspace(0, 0.5, 8000)).astype(np.float32)
S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64)
assert S.shape[0] == 64
print(f"librosa ok — mel shape {list(S.shape)}")
"""
    ),
    "nlp": (
        "NLP smoke (tokenizer)",
        """
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
out = tok("hello world", return_tensors="pt")
print(f"transformers ok — ids shape {list(out['input_ids'].shape)}")
"""
    ),
    "rl": (
        "RL smoke (gymnasium env)",
        """
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, _ = env.reset()
print(f"gymnasium ok — obs shape {obs.shape}")
"""
    ),
}


def _run(code: str):
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()


def verify_all(sys_info: dict, profile: str) -> dict:
    from .packages import PROFILE_GROUPS
    groups   = PROFILE_GROUPS.get(profile, list(CHECKS.keys()))
    results  = {}

    # ── Package checks ───────────────────────────────────────────────
    print(f"\n  {BOLD}Package Versions:{R}\n")
    for group in groups:
        checks = CHECKS.get(group, [])
        if not checks:
            continue
        print(f"  {B}  {group.upper()}{R}")
        for name, code in checks:
            success, out, _ = _run(code)
            if success:
                print(f"  {G}  ✅{R}  {name:<26} {out}")
                results[name] = out
            else:
                print(f"  {RED}  ❌{R}  {name:<26} FAILED")
                results[name] = "FAILED"

    # ── GPU checks ───────────────────────────────────────────────────
    needs_gpu = any(g in groups for g in ["dl_core", "cv", "nlp", "audio", "rl", "llm"])
    if needs_gpu:
        print(f"\n  {BOLD}GPU / Compute:{R}\n")
        hw = ("cuda" if sys_info.get("cuda") and not sys_info.get("force_cpu")
              else "mps" if sys_info.get("mps") and not sys_info.get("force_cpu")
              else "cpu")
        for name, code in GPU_CHECKS.get(hw, GPU_CHECKS["cpu"]):
            success, out, _ = _run(code)
            good   = success and "False" not in out and "0 GPU" not in out
            colour = G if good else Y
            icon   = "✅" if good else "⚠️ "
            print(f"  {colour}  {icon}{R}  {name:<26} {out if success else 'FAILED'}")
            results[f"gpu_{name}"] = out if success else "FAILED"

    # ── Smoke tests ──────────────────────────────────────────────────
    smoke_groups = [g for g in groups if g in SMOKE_TESTS]
    if smoke_groups:
        print(f"\n  {BOLD}Smoke Tests:{R}\n")
        for g in smoke_groups:
            label, code = SMOKE_TESTS[g]
            success, out, err = _run(code)
            if success:
                print(f"  {G}  ✅{R}  {label:<34} {out}")
                results[f"smoke_{g}"] = out
            else:
                print(f"  {Y}  ⚠️ {R}  {label:<34} {err[:70]}")
                results[f"smoke_{g}"] = "FAILED"

    return results


def save_report(sys_info: dict, results: dict, profile: str):
    from .packages import TORCH_VER, TENSORFLOW_VER, NUMPY_VER
    report = {
        "profile"   : profile,
        "system"    : sys_info,
        "packages"  : results,
        "pinned"    : {
            "torch"      : TORCH_VER,
            "tensorflow" : TENSORFLOW_VER,
            "numpy"      : NUMPY_VER,
        },
    }
    p = Path(__file__).resolve().parent.parent / "setup_report.json"
    p.write_text(json.dumps(report, indent=2))
    info(f"Report saved → {p}")
