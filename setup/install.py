"""
All installation steps. Called by colab_setup.py → run_all_steps().
"""
import platform, shutil, subprocess, sys
from .utils    import ok, warn, info, step, run_cmd, pip_install, pip_batch
from .packages import (
    PRE, CORE, DL_CORE, JUPYTER,
    TORCH_VER, TORCHVISION_VER, TORCHAUDIO_VER,
    TORCHTEXT_VER, TORCHCODEC_VER,
    TENSORFLOW_VER, KERAS_VER,
    NUMPY_VER, CUDA_INDEX,
    get_profile_packages, get_flat_list,
)

# ── System package lists per distro ─────────────────────────────────

_APT = [
    # Python build
    "python3-dev", "python3-pip", "build-essential", "cmake", "ninja-build",
    "git", "curl", "wget", "unzip", "pkg-config",
    # Audio / multimedia
    "ffmpeg", "libsndfile1", "libsndfile1-dev", "sox", "libsox-fmt-all",
    "libportaudio2", "portaudio19-dev", "libavcodec-extra",
    # Numeric / HDF5
    "libhdf5-dev", "libopenblas-dev", "liblapack-dev",
    # SSL / FFI
    "libffi-dev", "libssl-dev",
    # OpenCV / display
    "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6",
    # Tesseract OCR
    "tesseract-ocr", "libtesseract-dev",
    # Locale
    "locales",
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
]

# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_all_steps(sys_info: dict, profile: str,
                  skip_system: bool, skip_nlp_dl: bool):

    profile_pkgs = get_profile_packages(profile)
    needs_torch  = any(g in profile_pkgs for g in
                       ["dl_core", "cv", "nlp", "audio", "rl", "llm"])
    needs_tf     = any(g in profile_pkgs for g in ["dl_core"])
    needs_audio  = "audio" in profile_pkgs
    needs_nlp    = "nlp"   in profile_pkgs or "llm" in profile_pkgs

    total = sum([
        1,                    # pip upgrade
        1,                    # system deps
        1,                    # pre-install
        int(needs_torch),     # pytorch
        int(needs_tf),        # tensorflow
        len(profile_pkgs),    # profile groups
        int(needs_nlp and not skip_nlp_dl),   # nlp models
        1,                    # jupyter kernel
    ])

    n = 0

    # ── 1. pip upgrade ──────────────────────────────────────────────
    n += 1; step(n, total, "Upgrading pip / setuptools / wheel")
    _upgrade_pip()

    # ── 2. System deps ──────────────────────────────────────────────
    n += 1; step(n, total, "System dependencies (ffmpeg · cmake · libsndfile)")
    if not skip_system:
        _install_system_deps(sys_info)
    else:
        warn("--skip-system passed — skipping OS package install")

    # ── 3. Pre-install ──────────────────────────────────────────────
    n += 1; step(n, total, "Pre-install (numpy pin · Cython · typing-extensions)")
    pip_batch(PRE, "pre-install")

    # ── 4. PyTorch ──────────────────────────────────────────────────
    if needs_torch:
        n += 1; step(n, total,
                     f"PyTorch {TORCH_VER} + ecosystem (hardware-aware)")
        _install_pytorch(sys_info, include_audio="audio" in profile_pkgs)

    # ── 5. TensorFlow ───────────────────────────────────────────────
    if needs_tf:
        n += 1; step(n, total, f"TensorFlow {TENSORFLOW_VER} + Keras + TF-Hub")
        _install_tensorflow(sys_info)

    # ── 6. Profile package groups ────────────────────────────────────
    # Skip dl_core torch/tf (already done) — install remaining groups
    skip_in_group = {"dl_core"} if needs_torch or needs_tf else set()

    for group_name, pkgs in profile_pkgs.items():
        if group_name in skip_in_group:
            # DL_CORE contains TF packages; install the non-torch ones
            _install_dl_core_remainder(pkgs)
            continue
        n += 1
        label = group_name.replace("_", " ").upper()
        step(n, total, label)
        pip_batch(pkgs, label)

    # ── 7. NLP model downloads ───────────────────────────────────────
    if needs_nlp and not skip_nlp_dl:
        n += 1; step(n, total, "NLP model downloads (spaCy · NLTK)")
        _download_nlp_models()

    # ── 8. Jupyter kernel ────────────────────────────────────────────
    n += 1; step(n, total, "Registering Jupyter kernel")
    _register_kernel()


# ════════════════════════════════════════════════════════════════════════
#  STEP IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════

def _upgrade_pip():
    run_cmd(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        "pip upgrade")


def _install_system_deps(sys_info: dict):
    os_name = platform.system()

    if os_name == "Linux":
        if shutil.which("apt-get"):
            info("apt-get (Debian / Ubuntu / EC2 / RunPod)")
            subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
            subprocess.run(["apt-get", "install", "-y", "-q"] + _APT)
            ok("System packages installed (apt)")

        elif shutil.which("yum"):
            info("yum (RHEL / CentOS / Amazon Linux)")
            subprocess.run(["yum", "install", "-y", "-q",
                            "epel-release"], capture_output=True)
            subprocess.run(["yum", "install", "-y", "-q"] + _YUM)
            ok("System packages installed (yum)")

        elif shutil.which("dnf"):
            info("dnf (Fedora)")
            subprocess.run(["dnf", "install", "-y", "-q"] + _YUM)
            ok("System packages installed (dnf)")

        elif shutil.which("pacman"):
            info("pacman (Arch)")
            subprocess.run(
                ["pacman", "-Sy", "--noconfirm",
                 "ffmpeg", "libsndfile", "sox", "portaudio",
                 "cmake", "ninja", "tesseract"])
            ok("System packages installed (pacman)")

        else:
            warn("Unknown Linux distro — install ffmpeg + cmake manually")

    elif os_name == "Darwin":
        if not shutil.which("brew"):
            info("Homebrew not found — installing…")
            subprocess.run(["/bin/bash", "-c",
                            "$(curl -fsSL https://raw.githubusercontent.com/"
                            "Homebrew/install/HEAD/install.sh)"])
        subprocess.run(["brew", "update", "-q"])
        subprocess.run(["brew", "install", "-q"] + _BREW)
        ok("System packages installed (Homebrew)")

    elif os_name == "Windows":
        info("Windows: checking for ffmpeg…")
        if not shutil.which("ffmpeg"):
            warn("ffmpeg not in PATH.\n"
                 "     Install: winget install -e --id Gyan.FFmpeg\n"
                 "     Then add to PATH and re-run.")
        else:
            ok("ffmpeg found in PATH")


def _install_pytorch(sys_info: dict, include_audio: bool = True):
    """Install torch + vision + audio (if needed) with correct CUDA wheel."""
    base_pkgs = [
        f"torch=={TORCH_VER}",
        f"torchvision=={TORCHVISION_VER}",
    ]
    if include_audio:
        base_pkgs.append(f"torchaudio=={TORCHAUDIO_VER}")

    force_cpu = sys_info.get("force_cpu", False)

    if sys_info["cuda"] and not force_cpu:
        cv  = sys_info.get("cuda_version") or ""
        idx = (CUDA_INDEX["12"] if cv.startswith("12") else
               CUDA_INDEX["11"] if cv.startswith("11") else
               CUDA_INDEX["12"])
        info(f"CUDA {cv} → wheel index: {idx}")
        pip_batch(base_pkgs + ["--extra-index-url", idx],
                  f"PyTorch {TORCH_VER} + CUDA")

    elif sys_info["mps"] and not force_cpu:
        info("Apple Silicon — standard PyTorch with MPS")
        pip_batch(base_pkgs, f"PyTorch {TORCH_VER} + MPS")

    else:
        idx = CUDA_INDEX["cpu"]
        info(f"CPU-only → wheel index: {idx}")
        pip_batch(base_pkgs + ["--extra-index-url", idx],
                  f"PyTorch {TORCH_VER} CPU")

    # torchtext — must follow torch
    pip_install(f"torchtext=={TORCHTEXT_VER}", label="torchtext")

    # torchcodec — Linux only
    if sys_info["os"] == "Linux":
        extra_idx = (CUDA_INDEX["12"]
                     if (sys_info["cuda"] and not force_cpu)
                     else CUDA_INDEX["cpu"])
        pip_install(f"torchcodec=={TORCHCODEC_VER}",
                    extra_args=["--extra-index-url", extra_idx],
                    label="torchcodec")
    else:
        warn(f"torchcodec skipped — Linux only")


def _install_tensorflow(sys_info: dict):
    """Install TF ecosystem, then re-pin numpy."""
    tf_pkgs = [
        f"tensorflow=={TENSORFLOW_VER}",
        f"keras=={KERAS_VER}",
        "tensorflow-hub==0.16.1",
        "tensorflow-datasets==4.9.4",
        "tensorflow-addons==0.23.0",
    ]
    if sys_info["os"] == "Linux":
        tf_pkgs.append("tensorflow-io-gcs-filesystem>=0.36.0")

    pip_batch(tf_pkgs, f"TensorFlow {TENSORFLOW_VER}")

    # TF often upgrades numpy to 2.x — re-pin immediately
    info(f"Re-pinning numpy=={NUMPY_VER} (TF may have upgraded it)…")
    pip_install(f"numpy=={NUMPY_VER}", label="numpy re-pin")


def _install_dl_core_remainder(pkgs: list):
    """
    DL_CORE contains both torch and TF packages.
    torch/TF are already installed hardware-aware.
    This installs only the non-framework utilities from the group.
    """
    skip_prefixes = ("torch", "tensorflow", "keras")
    remainder = [p for p in pkgs
                 if not any(p.lower().startswith(s) for s in skip_prefixes)]
    if remainder:
        pip_batch(remainder, "DL utilities")


def _download_nlp_models():
    info("Downloading spaCy en_core_web_sm…")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True)
    ok("spaCy en_core_web_sm ready")

    info("Downloading NLTK data…")
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
                    "averaged_perceptron_tagger", "vader_lexicon"]:
            nltk.download(pkg, quiet=True)
        ok("NLTK data ready")
    except ImportError:
        warn("NLTK will download data on first use")


def _register_kernel():
    run_cmd(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "ml_env",
         "--display-name", "Python (ml_env — Colab)"],
        "Jupyter kernel registration",
        ignore_error=True)
