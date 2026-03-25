#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         Audio ML Environment Setup — Google Colab Mirror            ║
║  Project : Audio Signal Processing (RNN · CNN · Text · Multimodal)  ║
║  Stack   : PyTorch + TensorFlow + torchaudio + full audio libs       ║
║                                                                      ║
║  Usage   : python colab_setup.py                                     ║
║  OS      : Linux · macOS · Windows · EC2 · RunPod · Lightning AI     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, subprocess, platform, shutil, json, argparse
from pathlib import Path

# ── make sure we can find sibling modules ───────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "setup"))

from setup.detect   import detect_system, print_banner
from setup.install  import (
    upgrade_pip, install_system_audio_deps,
    install_pre_packages, install_pytorch,
    install_tensorflow, install_audio_packages,
    install_dl_packages, install_nlp_packages,
    install_cv_ml_packages, install_jupyter,
    download_nlp_models, register_kernel,
)
from setup.verify   import verify_all, save_report
from setup.packages import TORCH_VER, TENSORFLOW_VER
from setup.utils    import step, ok, warn, info, G, R, BOLD, C

TOTAL_STEPS = 10


def parse_args():
    p = argparse.ArgumentParser(
        description="Audio ML Environment Setup — Colab Mirror")
    p.add_argument("--skip-system",  action="store_true",
                   help="Skip apt/brew system package install")
    p.add_argument("--skip-nlp",     action="store_true",
                   help="Skip NLP model downloads (spaCy, NLTK)")
    p.add_argument("--cpu-only",     action="store_true",
                   help="Force CPU-only PyTorch even if GPU detected")
    p.add_argument("--verify-only",  action="store_true",
                   help="Only run verification, skip installs")
    return p.parse_args()


def main():
    args = parse_args()

    sys_info = detect_system(force_cpu=args.cpu_only)
    print_banner(sys_info, TORCH_VER, TENSORFLOW_VER)

    # Python version guard
    if sys.version_info < (3, 9):
        print(f"\n  ❌  Python 3.9+ required. You have "
              f"{sys.version_info.major}.{sys.version_info.minor}\n")
        sys.exit(1)

    if args.verify_only:
        results = verify_all(sys_info)
        save_report(sys_info, results)
        return

    step(1,  TOTAL_STEPS, "Upgrading pip & build tools")
    upgrade_pip()

    step(2,  TOTAL_STEPS, "System audio libraries (ffmpeg · libsndfile · sox)")
    if not args.skip_system:
        install_system_audio_deps()
    else:
        warn("--skip-system passed — skipping system package install")

    step(3,  TOTAL_STEPS, "Pre-install (numpy pin · Cython · typing-extensions)")
    install_pre_packages()

    step(4,  TOTAL_STEPS, f"PyTorch {TORCH_VER} + torchaudio + torchvision + torchtext + torchcodec")
    install_pytorch(sys_info)

    step(5,  TOTAL_STEPS, f"TensorFlow {TENSORFLOW_VER} + Keras + TF-Hub + TF-Addons")
    install_tensorflow(sys_info)

    step(6,  TOTAL_STEPS, "Audio packages (librosa · soundfile · pydub · speechbrain · opensmile)")
    install_audio_packages()

    step(7,  TOTAL_STEPS, "Deep Learning utilities (torchmetrics · lightning · timm · einops)")
    install_dl_packages()

    step(8,  TOTAL_STEPS, "NLP (transformers · Wav2Vec2 · Whisper · spaCy · NLTK)")
    install_nlp_packages()
    if not args.skip_nlp:
        download_nlp_models()

    step(9,  TOTAL_STEPS, "CV · ML · Experiment Tracking · Jupyter")
    install_cv_ml_packages()
    install_jupyter()
    register_kernel()

    step(10, TOTAL_STEPS, "Verification")
    results = verify_all(sys_info)
    save_report(sys_info, results)

    print(f"""{G}{BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║  ✅  Setup Complete — Audio ML environment ready!                    ║
║                                                                      ║
║  Kernel     :  Python (audio_ml — Colab)                            ║
║  Jupyter    :  jupyter lab                                           ║
║  Script     :  python your_notebook.py                              ║
║                                                                      ║
║  Audio      :  .wav  .ogg  .mp3  .flac  .aiff  ← all supported     ║
║  Load audio :  torchaudio.load("file.ogg")                          ║
╚══════════════════════════════════════════════════════════════════════╝{R}
""")


if __name__ == "__main__":
    main()
