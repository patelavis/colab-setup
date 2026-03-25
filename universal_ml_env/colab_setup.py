#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Universal DS / ML / AI Environment — Google Colab Mirror          ║
║                                                                              ║
║  Works on  : Linux · macOS · Windows · EC2 · RunPod · Lightning AI          ║
║              Databricks · Kaggle · Paperspace · Vast.ai · GCP · Azure       ║
║  Hardware  : NVIDIA GPU (CUDA 11/12) · Apple Silicon (MPS) · CPU            ║
║                                                                              ║
║  USAGE     :  python colab_setup.py                   ← full setup          ║
║               python colab_setup.py --profile nlp     ← NLP only            ║
║               python colab_setup.py --profile cv      ← Computer Vision     ║
║               python colab_setup.py --profile tabular ← Tabular / DS        ║
║               python colab_setup.py --profile rl      ← Reinforcement Lrng  ║
║               python colab_setup.py --profile audio   ← Audio / Speech      ║
║               python colab_setup.py --profile llm     ← LLMs / GenAI        ║
║               python colab_setup.py --profile full    ← Everything (default)║
║               python colab_setup.py --verify-only     ← check installs      ║
║               python colab_setup.py --cpu-only        ← force CPU           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from setup.detect   import detect_system, print_banner
from setup.install  import run_all_steps
from setup.verify   import verify_all, save_report
from setup.packages import TORCH_VER, TENSORFLOW_VER
from setup.utils    import step, ok, warn, G, R, BOLD, C

VALID_PROFILES = ["full", "nlp", "cv", "tabular", "rl", "audio", "llm", "timeseries"]

PROFILE_DESC = {
    "full"      : "Everything — complete Colab mirror (default)",
    "nlp"       : "NLP / Text  — transformers, tokenizers, spaCy, NLTK, HuggingFace",
    "cv"        : "Computer Vision — CNN, object detection, segmentation, GANs",
    "tabular"   : "Tabular / DS — pandas, sklearn, XGBoost, LightGBM, feature eng",
    "rl"        : "Reinforcement Learning — Gym, Stable-Baselines3, RLlib",
    "audio"     : "Audio / Speech — torchaudio, librosa, Whisper, speechbrain",
    "llm"       : "LLMs / GenAI — transformers, PEFT, LoRA, vLLM, LangChain",
    "timeseries": "Time Series — statsmodels, Prophet, sktime, darts, tsfresh",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Universal DS/ML/AI Environment Setup — Google Colab Mirror",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--profile", "-p",
        default="full",
        choices=VALID_PROFILES,
        help="\n".join(f"  {k:<12} {v}" for k, v in PROFILE_DESC.items()),
    )
    p.add_argument("--cpu-only",    action="store_true",
                   help="Force CPU-only PyTorch even if GPU detected")
    p.add_argument("--skip-system", action="store_true",
                   help="Skip apt/brew system package install")
    p.add_argument("--skip-nlp-dl", action="store_true",
                   help="Skip spaCy / NLTK model downloads")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify existing installs, no new installs")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print what would be installed without installing")
    return p.parse_args()


def print_profile_banner(profile: str):
    desc = PROFILE_DESC.get(profile, "")
    print(f"\n  {BOLD}Profile:{R}  {C}{profile.upper()}{R}  —  {desc}\n")


def main():
    args   = parse_args()
    profile = args.profile

    sys_info = detect_system(force_cpu=args.cpu_only)
    print_banner(sys_info, TORCH_VER, TENSORFLOW_VER)
    print_profile_banner(profile)

    # Python version guard
    if sys.version_info < (3, 9):
        print(f"\n  ❌  Python 3.9+ required. "
              f"You have {sys.version_info.major}.{sys.version_info.minor}\n")
        sys.exit(1)

    if args.verify_only:
        results = verify_all(sys_info, profile)
        save_report(sys_info, results, profile)
        return

    if args.dry_run:
        _dry_run(profile, sys_info)
        return

    # ── Run all install steps ────────────────────────────────────────
    run_all_steps(
        sys_info    = sys_info,
        profile     = profile,
        skip_system = args.skip_system,
        skip_nlp_dl = args.skip_nlp_dl,
    )

    # ── Verify + report ─────────────────────────────────────────────
    results = verify_all(sys_info, profile)
    save_report(sys_info, results, profile)

    print(f"""{G}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅  Setup Complete!  Profile: {profile.upper():<47}║
║                                                                              ║
║  Activate  :  source venv/bin/activate        (Linux/macOS)                 ║
║              venv\\Scripts\\activate             (Windows)                   ║
║  Jupyter   :  jupyter lab                                                    ║
║  Kernel    :  Python (ml_env — Colab)                                       ║
║  Verify    :  python colab_setup.py --verify-only                           ║
╚══════════════════════════════════════════════════════════════════════════════╝{R}
""")


def _dry_run(profile: str, sys_info: dict):
    """Print what would be installed for this profile."""
    from setup.packages import get_profile_packages
    pkgs = get_profile_packages(profile)
    hw   = ("CUDA " + (sys_info.get("cuda_version") or "")) if sys_info["cuda"] \
           else "MPS" if sys_info["mps"] else "CPU"
    print(f"\n  {BOLD}[DRY RUN] Profile: {profile.upper()}  |  Hardware: {hw}{R}\n")
    for group, plist in pkgs.items():
        print(f"  {C}{group}{R}")
        for p in plist:
            print(f"    • {p}")
    print()


if __name__ == "__main__":
    main()
