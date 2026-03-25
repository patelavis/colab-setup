"""
System detection: OS, architecture, CUDA/MPS GPU, runtime environment.
"""
import os, platform, shutil, subprocess, urllib.request
from .utils import ok, warn, info, G, Y, R, BOLD, C


def detect_system(force_cpu: bool = False) -> dict:
    d = {
        "os"           : platform.system(),           # Windows / Linux / Darwin
        "os_version"   : platform.version(),
        "arch"         : platform.machine(),           # x86_64 / arm64
        "python"       : f"{platform.python_version()}",
        "cuda"         : False,
        "cuda_version" : None,
        "mps"          : False,
        "gpu_name"     : None,
        "env"          : _detect_runtime_env(),
        "force_cpu"    : force_cpu,
    }

    if force_cpu:
        warn("--cpu-only flag set — skipping GPU detection")
        return d

    # ── NVIDIA CUDA ──────────────────────────────────────────────
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                d["cuda"]     = True
                d["gpu_name"] = r.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

        if d["cuda"]:
            try:
                r = subprocess.run(["nvcc", "--version"],
                                   capture_output=True, text=True, timeout=10)
                if "release" in r.stdout:
                    d["cuda_version"] = (
                        r.stdout.split("release ")[-1].split(",")[0].strip())
            except Exception:
                d["cuda_version"] = "unknown"

    # ── Apple Silicon MPS ────────────────────────────────────────
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        d["mps"] = True

    return d


def _detect_runtime_env() -> str:
    """Identify the compute platform we're running on."""
    if os.path.exists("/content"):
        return "google_colab"
    if os.path.exists("/databricks"):
        return "databricks"
    if os.environ.get("LIGHTNING_CLOUD_PROJECT_ID"):
        return "lightning_ai"
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    if os.environ.get("VAST_CONTAINERLABEL"):
        return "vast_ai"
    if os.environ.get("PAPERSPACE_FQDN"):
        return "paperspace"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    # AWS EC2 metadata probe
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/", timeout=1)
        return "aws_ec2"
    except Exception:
        pass
    return "local"


def print_banner(sys_info: dict, torch_ver: str, tf_ver: str):
    print(f"""{C}{BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║         Audio ML Environment Setup — Google Colab Mirror            ║
║  Stack  : PyTorch {torch_ver:<8}  +  TensorFlow {tf_ver:<8}              ║
╚══════════════════════════════════════════════════════════════════════╝{R}
  OS        : {sys_info['os']} {sys_info['os_version'][:35]}
  Arch      : {sys_info['arch']}
  Python    : {sys_info['python']}
  Platform  : {sys_info['env']}""")

    if sys_info["cuda"]:
        print(f"  GPU       : {G}{sys_info['gpu_name']}  |  CUDA {sys_info['cuda_version']}{R}")
    elif sys_info["mps"]:
        print(f"  GPU       : {G}Apple Silicon MPS{R}")
    else:
        print(f"  GPU       : {Y}None detected — CPU mode{R}")

    if sys_info.get("force_cpu"):
        print(f"  Mode      : {Y}CPU-only (forced){R}")

    print()
