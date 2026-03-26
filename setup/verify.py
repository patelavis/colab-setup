"""
Post-install verification: imports every key package and prints versions.
Also checks GPU availability for both torch and tensorflow.
"""
import json, subprocess, sys
from pathlib import Path
from .utils import ok, warn, info, G, Y, RED, R, BOLD, B

# (display_name, python_snippet_that_prints_version_or_status)
CHECKS = [
    # ── PyTorch family ──────────────────────────────────────────
    ("torch",               "import torch; print(torch.__version__)"),
    ("torchaudio",          "import torchaudio; print(torchaudio.__version__)"),
    ("torchvision",         "import torchvision; print(torchvision.__version__)"),
    ("torchtext",           "import torchtext; print(torchtext.__version__)"),
    ("torchaudio backend",  "import torchaudio; print(torchaudio.get_audio_backend())"),

    # ── TensorFlow family ───────────────────────────────────────
    ("tensorflow",          "import tensorflow as tf; print(tf.__version__)"),
    ("keras",               "import keras; print(keras.__version__)"),
    ("tensorflow_hub",      "import tensorflow_hub as hub; print(hub.__version__)"),

    # ── Audio ───────────────────────────────────────────────────
    ("librosa",             "import librosa; print(librosa.__version__)"),
    ("soundfile",           "import soundfile; print(soundfile.__version__)"),
    ("pydub",               "from pydub import AudioSegment; print('ok')"),
    ("audioread",           "import audioread; print(audioread.__version__)"),
    ("speechbrain",         "import speechbrain; print(speechbrain.__version__)"),
    ("opensmile",           "import opensmile; print(opensmile.__version__)"),

    # ── NLP ─────────────────────────────────────────────────────
    ("transformers",        "import transformers; print(transformers.__version__)"),
    ("tokenizers",          "import tokenizers; print(tokenizers.__version__)"),
    ("datasets",            "import datasets; print(datasets.__version__)"),
    ("sentence_transformers","import sentence_transformers; print(sentence_transformers.__version__)"),

    # ── CV / ML / Utils ─────────────────────────────────────────
    ("numpy",               "import numpy; print(numpy.__version__)"),
    ("pandas",              "import pandas; print(pandas.__version__)"),
    ("sklearn",             "import sklearn; print(sklearn.__version__)"),
    ("opencv",              "import cv2; print(cv2.__version__)"),
    ("numba",               "import numba; print(numba.__version__)"),
    ("scipy",               "import scipy; print(scipy.__version__)"),

    # ── Jupyter ─────────────────────────────────────────────────
    ("ipykernel",           "import ipykernel; print(ipykernel.__version__)"),
]

GPU_CHECKS = {
    "cuda": [
        ("torch.cuda.is_available",
         "import torch; print(torch.cuda.is_available())"),
        ("torch GPU name",
         "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"),
        ("tensorflow GPU list",
         "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'{len(gpus)} GPU(s) visible')"),
    ],
    "mps": [
        ("torch.backends.mps",
         "import torch; print(torch.backends.mps.is_available())"),
    ],
    "cpu": [
        ("torch CPU threads",
         "import torch; print(torch.get_num_threads(), 'threads')"),
    ],
}


def _run(code: str):
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()


def verify_all(sys_info: dict) -> dict:
    results = {}

    # ── Package checks ───────────────────────────────────────────
    print(f"\n  {BOLD}Package Versions:{R}\n")
    for name, code in CHECKS:
        success, out, _ = _run(code)
        if success:
            print(f"  {G}✅{R}  {name:<28} {out}")
            results[name] = out
        else:
            print(f"  {RED}❌{R}  {name:<28} FAILED")
            results[name] = "FAILED"

    # ── GPU checks ───────────────────────────────────────────────
    print(f"\n  {BOLD}GPU / Compute:{R}\n")

    hw = ("cuda" if sys_info.get("cuda") and not sys_info.get("force_cpu")
          else "mps"  if sys_info.get("mps")  and not sys_info.get("force_cpu")
          else "cpu")

    for name, code in GPU_CHECKS.get(hw, GPU_CHECKS["cpu"]):
        success, out, _ = _run(code)
        colour = G if success and "False" not in out and "0 GPU" not in out else Y
        icon   = "✅" if colour == G else "⚠️ "
        print(f"  {colour}{icon}{R}  {name:<28} {out if success else 'FAILED'}")
        results[f"gpu_{name}"] = out if success else "FAILED"

    # ── Audio codec smoke test ────────────────────────────────────
    print(f"\n  {BOLD}Audio Codec Smoke Test:{R}\n")
    _audio_smoke_test(results)

    return results


def _audio_smoke_test(results: dict):
    """Try to load a tiny synthetic WAV via torchaudio and librosa."""
    smoke = """
import numpy as np, io, torch
import torchaudio, soundfile as sf

# generate 0.1s 440 Hz sine at 16kHz
sr = 16000
t  = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
wave = np.sin(2 * np.pi * 440 * t)

buf = io.BytesIO()
sf.write(buf, wave, sr, format='WAV', subtype='PCM_16')
buf.seek(0)
waveform, sample_rate = torchaudio.load(buf)
assert waveform.shape[1] > 0, "empty waveform"
print(f"shape={list(waveform.shape)} sr={sample_rate}")
"""
    success, out, err = _run(smoke)
    if success:
        print(f"  {G}✅{R}  {'torchaudio.load (WAV)':<28} {out}")
        results["audio_smoke_wav"] = out
    else:
        print(f"  {Y}⚠️ {R}  {'torchaudio.load (WAV)':<28} {err[:80]}")
        results["audio_smoke_wav"] = "FAILED"

    # librosa load
    smoke2 = """
import numpy as np, librosa
sr = 16000
y  = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr*0.1))).astype(np.float32)
# just check STFT and mel
S  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print(f"mel shape={list(S.shape)}")
"""
    success, out, err = _run(smoke2)
    if success:
        print(f"  {G}✅{R}  {'librosa melspectrogram':<28} {out}")
        results["audio_smoke_mel"] = out
    else:
        print(f"  {Y}⚠️ {R}  {'librosa melspectrogram':<28} {err[:80]}")
        results["audio_smoke_mel"] = "FAILED"


def save_report(sys_info: dict, results: dict):
    from .packages import TORCH_VER, TENSORFLOW_VER, NUMPY_VER
    report = {
        "system"    : sys_info,
        "packages"  : results,
        "versions"  : {
            "torch"      : TORCH_VER,
            "tensorflow" : TENSORFLOW_VER,
            "numpy"      : NUMPY_VER,
        },
    }
    p = Path(__file__).resolve().parent.parent / "setup_report.json"
    p.write_text(json.dumps(report, indent=2))
    info(f"Setup report saved → {p}")
