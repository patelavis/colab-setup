"""
All installation steps, each as a standalone function.
Called in order from colab_setup.py.
"""
import platform, shutil, subprocess, sys
from .utils    import ok, warn, info, run_cmd, pip_install, pip_batch
from .packages import (
    PRE_PACKAGES, AUDIO_PACKAGES, DL_PACKAGES, NLP_PACKAGES,
    CV_PACKAGES, ML_PACKAGES, EXPERIMENT_PACKAGES,
    JUPYTER_PACKAGES, UTIL_PACKAGES,
    TORCH_VER, TORCHVISION_VER, TORCHAUDIO_VER,
    TORCHTEXT_VER, TORCHCODEC_VER,
    TENSORFLOW_VER, KERAS_VER, TF_HUB_VER,
    TF_DATASETS_VER, TF_ADDONS_VER,
    NUMPY_VER, CUDA_INDEX,
)


# ════════════════════════════════════════════════════════════════════
#  1. PIP UPGRADE
# ════════════════════════════════════════════════════════════════════

def upgrade_pip():
    run_cmd(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        "pip upgrade")


# ════════════════════════════════════════════════════════════════════
#  2. SYSTEM AUDIO LIBRARIES
#     These are required so torchaudio.load / librosa / pydub can
#     decode .ogg / .mp3 / .wav on the OS level.
# ════════════════════════════════════════════════════════════════════

# Packages needed per package manager
_APT_PKGS = [
    "ffmpeg",             # mp3/ogg/wav codec — used by torchaudio, audioread, pydub
    "libsndfile1",        # soundfile backend (.wav .flac .ogg)
    "libsndfile1-dev",
    "sox",                # Swiss Army knife for audio
    "libsox-fmt-all",     # all sox format plugins
    "libportaudio2",      # pyaudio / sounddevice backend
    "portaudio19-dev",
    "libavcodec-extra",   # extra ffmpeg codecs (mp3, aac, ogg)
    "libffi-dev",
    "libssl-dev",
    "libhdf5-dev",        # h5py
    "pkg-config",
    "cmake",
    "ninja-build",
    "build-essential",
    "git",
    "curl",
    "wget",
    "unzip",
]

_YUM_PKGS = [
    "ffmpeg", "libsndfile", "sox", "portaudio-devel",
    "libffi-devel", "openssl-devel", "hdf5-devel",
    "cmake", "ninja-build", "gcc", "gcc-c++", "git", "curl",
]

_BREW_PKGS = [
    "ffmpeg", "libsndfile", "sox", "portaudio",
    "cmake", "ninja", "libomp", "openblas",
]


def install_system_audio_deps():
    os_name = platform.system()

    if os_name == "Linux":
        if shutil.which("apt-get"):
            info("Detected apt (Debian/Ubuntu/EC2 Amazon Linux 2022+)")
            subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
            subprocess.run(["apt-get", "install", "-y", "-q"] + _APT_PKGS)
            ok("System packages installed via apt")

        elif shutil.which("yum"):
            info("Detected yum (RHEL/CentOS/Amazon Linux)")
            # Enable EPEL for ffmpeg
            subprocess.run(["yum", "install", "-y", "-q",
                            "epel-release"], capture_output=True)
            subprocess.run(["yum", "install", "-y", "-q"] + _YUM_PKGS)
            ok("System packages installed via yum")

        elif shutil.which("dnf"):
            info("Detected dnf (Fedora)")
            subprocess.run(["dnf", "install", "-y", "-q"] + _YUM_PKGS)
            ok("System packages installed via dnf")

        elif shutil.which("pacman"):
            info("Detected pacman (Arch)")
            subprocess.run(["pacman", "-Sy", "--noconfirm",
                            "ffmpeg", "libsndfile", "sox",
                            "portaudio", "cmake", "ninja"])
            ok("System packages installed via pacman")

        else:
            warn("Unknown Linux distro — install ffmpeg + libsndfile manually "
                 "if audio fails")

    elif os_name == "Darwin":
        if not shutil.which("brew"):
            info("Homebrew not found — installing…")
            subprocess.run(
                ["/bin/bash", "-c",
                 "$(curl -fsSL https://raw.githubusercontent.com/"
                 "Homebrew/install/HEAD/install.sh)"])
        subprocess.run(["brew", "update", "-q"])
        subprocess.run(["brew", "install", "-q"] + _BREW_PKGS)
        ok("System packages installed via Homebrew")

    elif os_name == "Windows":
        info("Windows detected — checking for ffmpeg in PATH…")
        if not shutil.which("ffmpeg"):
            warn("ffmpeg NOT found in PATH.\n"
                 "     Install it from https://ffmpeg.org/download.html\n"
                 "     or via: winget install -e --id Gyan.FFmpeg\n"
                 "     Then add to PATH and re-run this script.")
        else:
            ok("ffmpeg found in PATH")


# ════════════════════════════════════════════════════════════════════
#  3. PRE-PACKAGES  (numpy pin must happen before torch/tf)
# ════════════════════════════════════════════════════════════════════

def install_pre_packages():
    pip_batch(PRE_PACKAGES, "pre-install")


# ════════════════════════════════════════════════════════════════════
#  4. PYTORCH — hardware-aware, all sub-packages pinned to same minor
# ════════════════════════════════════════════════════════════════════

def install_pytorch(sys_info: dict):
    torch_pkgs = [
        f"torch=={TORCH_VER}",
        f"torchvision=={TORCHVISION_VER}",
        f"torchaudio=={TORCHAUDIO_VER}",
    ]

    force_cpu = sys_info.get("force_cpu", False)

    if sys_info["cuda"] and not force_cpu:
        cv = sys_info.get("cuda_version") or ""
        if cv.startswith("12"):
            idx = CUDA_INDEX["12"]
        elif cv.startswith("11"):
            idx = CUDA_INDEX["11"]
        else:
            idx = CUDA_INDEX["12"]   # safe default
        info(f"CUDA {cv} → PyTorch wheel index: {idx}")
        pip_batch(
            torch_pkgs + ["--extra-index-url", idx],
            f"PyTorch {TORCH_VER} + CUDA")

    elif sys_info["mps"] and not force_cpu:
        info("Apple Silicon → standard PyTorch (MPS enabled since 2.0)")
        pip_batch(torch_pkgs, f"PyTorch {TORCH_VER} + MPS")

    else:
        idx = CUDA_INDEX["cpu"]
        info(f"CPU-only → PyTorch wheel index: {idx}")
        pip_batch(
            torch_pkgs + ["--extra-index-url", idx],
            f"PyTorch {TORCH_VER} CPU")

    # torchtext — must follow torch install
    info("Installing torchtext…")
    pip_install(f"torchtext=={TORCHTEXT_VER}", label="torchtext")

    # torchcodec — Linux only (requires ffmpeg libs)
    if sys_info["os"] == "Linux":
        info("Installing torchcodec (Linux only)…")
        extra_idx = (CUDA_INDEX["12"]
                     if (sys_info["cuda"] and not force_cpu)
                     else CUDA_INDEX["cpu"])
        pip_install(f"torchcodec=={TORCHCODEC_VER}",
                    extra_args=["--extra-index-url", extra_idx],
                    label="torchcodec")
    else:
        warn(f"torchcodec skipped — Linux only. "
             f"Use torchaudio.load() on {sys_info['os']}.")


# ════════════════════════════════════════════════════════════════════
#  5. TENSORFLOW — installed AFTER torch; numpy re-pinned after
# ════════════════════════════════════════════════════════════════════

def install_tensorflow(sys_info: dict):
    tf_pkgs = [
        f"tensorflow=={TENSORFLOW_VER}",
        f"keras=={KERAS_VER}",
        f"tensorflow-hub=={TF_HUB_VER}",
        f"tensorflow-datasets=={TF_DATASETS_VER}",
        f"tensorflow-addons=={TF_ADDONS_VER}",
    ]
    # On Linux TF needs gcs filesystem driver
    if sys_info["os"] == "Linux":
        tf_pkgs.append("tensorflow-io-gcs-filesystem>=0.36.0")

    pip_batch(tf_pkgs, f"TensorFlow {TENSORFLOW_VER} ecosystem")

    # !! CRITICAL: TF may upgrade numpy to 2.x — re-pin immediately
    info(f"Re-pinning numpy=={NUMPY_VER} (TF may have upgraded it)…")
    pip_install(f"numpy=={NUMPY_VER}", label="numpy re-pin")


# ════════════════════════════════════════════════════════════════════
#  6. AUDIO
# ════════════════════════════════════════════════════════════════════

def install_audio_packages():
    pip_batch(AUDIO_PACKAGES, "audio packages")


# ════════════════════════════════════════════════════════════════════
#  7. DL UTILITIES
# ════════════════════════════════════════════════════════════════════

def install_dl_packages():
    pip_batch(DL_PACKAGES, "DL utilities")


# ════════════════════════════════════════════════════════════════════
#  8. NLP
# ════════════════════════════════════════════════════════════════════

def install_nlp_packages():
    pip_batch(NLP_PACKAGES, "NLP packages")


def download_nlp_models():
    """Download spaCy model and NLTK corpora."""
    info("Downloading spaCy en_core_web_sm…")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True)
    ok("spaCy model ready")

    info("Downloading NLTK datasets…")
    try:
        import nltk
        for pkg in ["punkt", "stopwords", "wordnet",
                    "averaged_perceptron_tagger"]:
            nltk.download(pkg, quiet=True)
        ok("NLTK datasets ready")
    except ImportError:
        warn("NLTK not yet importable — data will download on first use")


# ════════════════════════════════════════════════════════════════════
#  9. CV / ML / EXPERIMENT / JUPYTER / UTILS
# ════════════════════════════════════════════════════════════════════

def install_cv_ml_packages():
    pip_batch(CV_PACKAGES,         "Computer Vision")
    pip_batch(ML_PACKAGES,         "Classical ML")
    pip_batch(EXPERIMENT_PACKAGES, "Experiment tracking & viz")
    pip_batch(UTIL_PACKAGES,       "Utilities")


def install_jupyter():
    pip_batch(JUPYTER_PACKAGES, "Jupyter Lab")


def register_kernel():
    info("Registering Jupyter kernel 'audio_ml'…")
    run_cmd(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "audio_ml",
         "--display-name", "Python (audio_ml — Colab)"],
        "Jupyter kernel registration",
        ignore_error=True)
