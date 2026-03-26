"""
All install steps — ordered exactly as the Colab Docker image builds them:
  1. System libs  →  2. Pre-pin numpy  →  3. PyTorch (hardware-aware)
  →  4. TensorFlow  →  5. Re-pin numpy  →  6. Profile groups
  →  7. NLP models  →  8. Jupyter kernel

Reference: us-docker.pkg.dev/colab-images/public/runtime
           us-docker.pkg.dev/colab-images/public/cpu-runtime
"""
import platform, shutil, subprocess, sys
from .utils    import ok, warn, info, step, run_cmd, pip_install, pip_batch
from .packages import (
    PRE, CORE, DL_CORE, JUPYTER,
    TORCH_VER, TORCHVISION_VER, TORCHAUDIO_VER,
    TORCHTEXT_VER, TORCHCODEC_VER,
    TENSORFLOW_VER, KERAS_VER,
    NUMPY_VER, CUDA_INDEX,
    get_profile_packages,
)

# ── System packages that Colab Docker image installs via apt ─────────
# Source: Colab apt-list.txt from googlecolab/backend-info
_APT = [
    # Build essentials
    "build-essential", "cmake", "ninja-build",
    "git", "curl", "wget", "unzip", "pkg-config",
    "python3-dev", "python3-pip",
    # Audio / multimedia (for torchaudio .ogg .mp3 .wav support)
    "ffmpeg",
    "libsndfile1", "libsndfile1-dev",
    "sox", "libsox-fmt-all",
    "libportaudio2", "portaudio19-dev",
    "libavcodec-extra", "libavformat-dev", "libavutil-dev",
    # Numeric / HDF5
    "libhdf5-dev", "libopenblas-dev", "liblapack-dev",
    # OpenCV / display
    "libgl1-mesa-glx", "libglib2.0-0",
    "libsm6", "libxrender1", "libxext6",
    # SSL / FFI
    "libffi-dev", "libssl-dev",
    # OCR
    "tesseract-ocr", "libtesseract-dev",
    # Spatial / GIS (Colab ships geopandas)
    "libgeos-dev", "libproj-dev", "libgdal-dev",
    # Misc
    "locales", "ca-certificates",
]

_YUM = [
    "gcc", "gcc-c++", "cmake", "ninja-build", "git", "curl", "wget",
    "ffmpeg", "libsndfile", "sox", "portaudio-devel",
    "openblas-devel", "lapack-devel", "hdf5-devel",
    "libffi-devel", "openssl-devel",
    "mesa-libGL", "tesseract",
]

_BREW = [
    "ffmpeg", "libsndfile", "sox", "portaudio", "cmake",
    "ninja", "libomp", "openblas", "hdf5", "tesseract",
    "geos", "proj",
]


# ════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY
# ════════════════════════════════════════════════════════════════════════

def run_all_steps(sys_info: dict, profile: str,
                  skip_system: bool, skip_nlp_dl: bool):

    profile_pkgs = get_profile_packages(profile)

    needs_torch = any(g in profile_pkgs for g in
                      ["dl_core", "cv", "nlp", "audio", "rl", "llm", "genai"])
    needs_tf    = "dl_core" in profile_pkgs
    needs_jax   = "dl_core" in profile_pkgs
    needs_nlp   = any(g in profile_pkgs for g in ["nlp", "llm", "genai"])
    needs_audio = "audio" in profile_pkgs

    TOTAL = 9
    n = 0

    n += 1; step(n, TOTAL, "Upgrading pip / setuptools / wheel")
    _upgrade_pip()

    n += 1; step(n, TOTAL, "System libraries  [Colab Docker apt-list]")
    if not skip_system:
        _system(sys_info)
    else:
        warn("--skip-system: skipping OS package install")

    n += 1; step(n, TOTAL, "Pre-install  [numpy pin · Cython · typing-extensions]")
    pip_batch(PRE, "pre-install")

    n += 1; step(n, TOTAL,
                 f"PyTorch {TORCH_VER} + torchaudio + torchvision + torchtext  [hardware-aware]")
    if needs_torch:
        _pytorch(sys_info, include_audio=needs_audio)
    else:
        info("Skipped — not required by this profile")

    n += 1; step(n, TOTAL,
                 f"TensorFlow {TENSORFLOW_VER} + Keras {KERAS_VER} + TF-Hub  [Colab image]")
    if needs_tf:
        _tensorflow(sys_info)
    else:
        info("Skipped — not required by this profile")

    n += 1; step(n, TOTAL, "numpy re-pin  [TF may have silently upgraded it]")
    pip_install(f"numpy=={NUMPY_VER}", label="numpy==1.26.4 re-pin")

    n += 1; step(n, TOTAL, f"Profile packages  [{profile.upper()}]")
    _skip_in_group = {"dl_core"}  # torch/tf handled above
    for gname, pkgs in profile_pkgs.items():
        if gname in _skip_in_group:
            _dl_core_remainder(pkgs, needs_jax, sys_info)
            continue
        label = gname.replace("_", " ").upper()
        pip_batch(pkgs, label)

    n += 1; step(n, TOTAL, "NLP model downloads  [spaCy · NLTK]")
    if needs_nlp and not skip_nlp_dl:
        _nlp_models()
    else:
        info("Skipped")

    n += 1; step(n, TOTAL, "Jupyter kernel registration  [Python (colab_mirror)]")
    _kernel()


# ════════════════════════════════════════════════════════════════════════
#  STEP IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════

def _upgrade_pip():
    run_cmd([sys.executable, "-m", "pip", "install",
             "--upgrade", "pip", "setuptools", "wheel"], "pip upgrade")


def _system(sys_info: dict):
    """
    Install OS-level packages that match what the Colab Docker image has.
    The Colab runtime image is built on Ubuntu 22.04 (apt).
    """
    os_name = platform.system()

    if os_name == "Linux":
        if shutil.which("apt-get"):
            info("apt-get  (Ubuntu/Debian — same as Colab Docker base)")
            subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
            subprocess.run(["apt-get", "install", "-y", "-q"] + _APT)
            ok("System packages installed (apt) — Colab Docker equivalent")

        elif shutil.which("yum"):
            info("yum  (RHEL/CentOS/Amazon Linux)")
            subprocess.run(["yum", "install", "-y", "-q", "epel-release"],
                           capture_output=True)
            subprocess.run(["yum", "install", "-y", "-q"] + _YUM)
            ok("System packages installed (yum)")

        elif shutil.which("dnf"):
            subprocess.run(["dnf", "install", "-y", "-q"] + _YUM)
            ok("System packages installed (dnf)")

        elif shutil.which("pacman"):
            subprocess.run([
                "pacman", "-Sy", "--noconfirm",
                "ffmpeg", "libsndfile", "sox", "portaudio",
                "cmake", "ninja", "tesseract", "geos"])
            ok("System packages installed (pacman)")

        else:
            warn("Unknown distro — install ffmpeg + libsndfile manually")

    elif os_name == "Darwin":
        if not shutil.which("brew"):
            info("Installing Homebrew…")
            subprocess.run(["/bin/bash", "-c",
                "$(curl -fsSL https://raw.githubusercontent.com/"
                "Homebrew/install/HEAD/install.sh)"])
        subprocess.run(["brew", "update", "-q"])
        subprocess.run(["brew", "install", "-q"] + _BREW)
        ok("System packages installed (Homebrew)")

    elif os_name == "Windows":
        info("Windows: verifying ffmpeg…")
        if not shutil.which("ffmpeg"):
            warn("ffmpeg not in PATH.\n"
                 "     Run: winget install -e --id Gyan.FFmpeg\n"
                 "     Then add to PATH and re-run this script.")
        else:
            ok("ffmpeg found in PATH")


def _pytorch(sys_info: dict, include_audio: bool = True):
    """
    Install torch family with the correct CUDA wheel.
    Colab uses cu121 wheels even when running CUDA 12.5 on the host.
    Reference: us-docker.pkg.dev/colab-images/public/runtime
    """
    pkgs = [
        f"torch=={TORCH_VER}",
        f"torchvision=={TORCHVISION_VER}",
    ]
    if include_audio:
        pkgs.append(f"torchaudio=={TORCHAUDIO_VER}")

    force_cpu = sys_info.get("force_cpu", False)

    if sys_info["cuda"] and not force_cpu:
        cv  = sys_info.get("cuda_version") or ""
        # Colab itself uses cu121 on CUDA 12.x hosts — replicate that
        idx = (CUDA_INDEX["12"] if cv.startswith("12") else
               CUDA_INDEX["11"] if cv.startswith("11") else
               CUDA_INDEX["12"])
        info(f"CUDA {cv} detected → wheel: {idx}  (mirrors Colab cu121 choice)")
        pip_batch(pkgs + ["--extra-index-url", idx],
                  f"PyTorch {TORCH_VER} CUDA")

    elif sys_info["mps"] and not force_cpu:
        info("Apple Silicon → standard PyTorch (MPS built-in)")
        pip_batch(pkgs, f"PyTorch {TORCH_VER} MPS")

    else:
        info(f"CPU-only → {CUDA_INDEX['cpu']}")
        pip_batch(pkgs + ["--extra-index-url", CUDA_INDEX["cpu"]],
                  f"PyTorch {TORCH_VER} CPU")

    pip_install(f"torchtext=={TORCHTEXT_VER}", label="torchtext")

    # torchcodec — Linux only (needs ffmpeg/libavcodec)
    if sys_info["os"] == "Linux":
        extra = (CUDA_INDEX["12"]
                 if (sys_info["cuda"] and not force_cpu)
                 else CUDA_INDEX["cpu"])
        pip_install(f"torchcodec=={TORCHCODEC_VER}",
                    extra_args=["--extra-index-url", extra],
                    label="torchcodec")
    else:
        warn(f"torchcodec: Linux only — skipped on {sys_info['os']}")


def _tensorflow(sys_info: dict):
    """
    Install TF ecosystem matching exact Colab runtime versions.
    Re-pin numpy immediately after — TF 2.18 may upgrade it to 2.x.
    """
    pkgs = [
        f"tensorflow=={TENSORFLOW_VER}",
        f"keras=={KERAS_VER}",
        "tensorflow-hub==0.16.1",
        "tensorflow-datasets==4.9.7",
    ]
    # Linux needs gcs filesystem driver for TF
    if sys_info["os"] == "Linux":
        pkgs.append("tensorflow-io-gcs-filesystem>=0.36.0")

    pip_batch(pkgs, f"TensorFlow {TENSORFLOW_VER} + Keras {KERAS_VER}")


def _dl_core_remainder(pkgs: list, include_jax: bool, sys_info: dict):
    """
    DL_CORE group contains TF (handled above) + JAX + utilities.
    Skip TF/Keras (already installed), handle JAX separately.
    """
    skip_start = ("tensorflow", "keras")
    jax_pkgs   = []
    rest       = []

    for p in pkgs:
        pl = p.lower()
        if any(pl.startswith(s) for s in skip_start):
            continue
        elif pl.startswith("jax"):
            jax_pkgs.append(p)
        else:
            rest.append(p)

    if rest:
        pip_batch(rest, "DL utilities")

    if include_jax and jax_pkgs:
        force_cpu = sys_info.get("force_cpu", False)
        if sys_info["cuda"] and not force_cpu:
            info("Installing JAX with CUDA 12 (Colab includes JAX)…")
            pip_install("jax[cuda12]", label="jax[cuda12]")
        else:
            info("Installing JAX CPU…")
            pip_install("jax[cpu]", label="jax[cpu]")


def _nlp_models():
    info("Downloading spaCy en_core_web_sm…")
    subprocess.run([sys.executable, "-m", "spacy", "download",
                    "en_core_web_sm"], capture_output=True)
    ok("spaCy model ready")

    info("Downloading NLTK data…")
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
                    "averaged_perceptron_tagger", "vader_lexicon"]:
            nltk.download(pkg, quiet=True)
        ok("NLTK data ready")
    except ImportError:
        warn("NLTK will download data on first import")


def _kernel():
    run_cmd([sys.executable, "-m", "ipykernel", "install",
             "--user", "--name", "colab_mirror",
             "--display-name", "Python (colab_mirror)"],
            "Jupyter kernel", ignore_error=True)
