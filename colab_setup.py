#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       Universal DS / ML / AI Environment  —  Google Colab Mirror            ║
║                                                                              ║
║  Docker  :  us-docker.pkg.dev/colab-images/public/runtime    (GPU)          ║
║             us-docker.pkg.dev/colab-images/public/cpu-runtime (CPU)         ║
║  Ref     :  github.com/googlecolab/backend-info  pip-freeze.txt             ║
║                                                                              ║
║  Runtime :  Python 3.12 · CUDA 12.5 · torch 2.5.1 · tensorflow 2.18.0      ║
║             numpy 1.26.4 · keras 3.8.0 · scikit-learn 1.6.1                ║
║                                                                              ║
║  USAGE                                                                       ║
║    python colab_setup.py                    full setup (all profiles)        ║
║    python colab_setup.py -p nlp             NLP / Text models only           ║
║    python colab_setup.py -p cv              Computer Vision only             ║
║    python colab_setup.py -p tabular         DS / Classical ML only           ║
║    python colab_setup.py -p audio           Audio / Speech only              ║
║    python colab_setup.py -p rl              Reinforcement Learning only      ║
║    python colab_setup.py -p llm             LLMs / GenAI / RAG only          ║
║    python colab_setup.py -p timeseries      Time Series / Forecasting only   ║
║    python colab_setup.py -p genai           GenAI + extras (Gradio, etc.)    ║
║                                                                              ║
║    python colab_setup.py --verify-only      check what's installed           ║
║    python colab_setup.py --dry-run          print packages, no install       ║
║    python colab_setup.py --cpu-only         force CPU-only PyTorch           ║
║    python colab_setup.py --skip-system      skip apt/brew installs           ║
║    python colab_setup.py --skip-nlp-dl      skip spaCy/NLTK downloads       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from setup.detect   import detect_system, print_banner
from setup.install  import run_all_steps
from setup.verify   import verify_all, save_report
from setup.packages import (
    TORCH_VER, TENSORFLOW_VER, NUMPY_VER, KERAS_VER,
    COLAB_PYTHON, COLAB_CUDA, PROFILE_GROUPS,
    get_profile_packages,
)
from setup.utils import G, R, BOLD, C, Y, warn, info

PROFILE_DESC = {
    "full"      : "Everything — complete Colab mirror (default)",
    "nlp"       : "NLP/Text  — transformers, Whisper, BERT, spaCy, NLTK",
    "cv"        : "Computer Vision — CNN, YOLO, diffusers, OCR",
    "tabular"   : "Tabular/DS — pandas, sklearn, XGBoost, SHAP, Optuna",
    "rl"        : "Reinforcement Learning — gymnasium, stable-baselines3",
    "audio"     : "Audio/Speech — torchaudio, librosa, speechbrain, Whisper",
    "llm"       : "LLMs/GenAI — PEFT, LoRA, LangChain, vLLM, RAG",
    "timeseries": "Time Series — Prophet, sktime, darts, tsfresh",
    "genai"     : "GenAI + Extras — LLMs + gradio + google-genai + duckdb",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Universal DS/ML/AI — Google Colab Mirror",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--profile", "-p", default="full",
        choices=list(PROFILE_GROUPS.keys()),
        help="\n".join(f"  {k:<12} {v}" for k, v in PROFILE_DESC.items()))
    p.add_argument("--cpu-only",    action="store_true")
    p.add_argument("--skip-system", action="store_true")
    p.add_argument("--skip-nlp-dl", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    profile  = args.profile
    sys_info = detect_system(force_cpu=args.cpu_only)

    print_banner(sys_info)
    print(f"  {BOLD}Profile  :{R}  {C}{profile.upper()}{R}  —  {PROFILE_DESC.get(profile,'')}\n")

    if sys.version_info < (3, 9):
        print(f"\n  ❌  Python 3.9+ required. Colab uses {COLAB_PYTHON}.\n")
        sys.exit(1)

    if args.verify_only:
        results = verify_all(sys_info, profile)
        save_report(sys_info, results, profile)
        return

    if args.dry_run:
        _dry_run(profile, sys_info)
        return

    run_all_steps(
        sys_info    = sys_info,
        profile     = profile,
        skip_system = args.skip_system,
        skip_nlp_dl = args.skip_nlp_dl,
    )

    results = verify_all(sys_info, profile)
    save_report(sys_info, results, profile)

    print(f"""{G}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅  Done!  Profile [{profile.upper():<10}] mirrors Google Colab runtime.           ║
║                                                                              ║
║  Activate   :  source venv/bin/activate    (Linux/macOS)                    ║
║               venv\\Scripts\\activate        (Windows)                       ║
║  Jupyter    :  jupyter lab                                                   ║
║  Kernel     :  Python (colab_mirror)                                        ║
║  Verify     :  python colab_setup.py --verify-only                          ║
╚══════════════════════════════════════════════════════════════════════════════╝{R}
""")


def _dry_run(profile: str, sys_info: dict):
    pkgs = get_profile_packages(profile)
    hw   = (f"CUDA {sys_info.get('cuda_version','?')}"
            if sys_info["cuda"] else
            "MPS" if sys_info["mps"] else "CPU")
    print(f"\n  {BOLD}[DRY RUN]  Profile: {profile.upper()}  |  Hardware: {hw}{R}\n")
    total = 0
    for group, plist in pkgs.items():
        count = len(plist)
        total += count
        print(f"  {C}{group.upper()}{R}  ({count} packages)")
        for p in plist:
            print(f"    · {p}")
    print(f"\n  Total: {total} packages  (plus torch/tf installed hardware-aware)\n")


if __name__ == "__main__":
    main()
