"""
System detection: OS, architecture, CUDA/MPS GPU, runtime environment.
"""
import os, platform, shutil, subprocess, urllib.request
from .utils import ok, warn, info, G, Y, R, BOLD, C, B


def detect_system(force_cpu: bool = False) -> dict:
    d = {
        "os"          : platform.system(),       # Windows / Linux / Darwin
        "os_version"  : platform.version(),
        "arch"        : platform.machine(),       # x86_64 / arm64 / AMD64
        "python"      : platform.python_version(),
        "cuda"        : False,
        "cuda_version": None,
        "mps"         : False,
        "gpu_name"    : None,
        "gpu_count"   : 0,
        "env"         : _detect_runtime_env(),
        "force_cpu"   : force_cpu,
        "ram_gb"      : _ram_gb(),
        "disk_free_gb": _disk_free_gb(),
    }

    if not force_cpu:
        _probe_cuda(d)
        _probe_mps(d)

    return d


def _probe_cuda(d: dict):
    if not shutil.which("nvidia-smi"):
        return
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().splitlines()
            d["cuda"]      = True
            d["gpu_name"]  = lines[0].split(",")[0].strip()
            d["gpu_count"] = len(lines)
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


def _probe_mps(d: dict):
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        d["mps"] = True


def _detect_runtime_env() -> str:
    """Identify the compute platform."""
    checks = [
        (lambda: os.path.exists("/content"),
         "google_colab"),
        (lambda: os.path.exists("/databricks"),
         "databricks"),
        (lambda: bool(os.environ.get("LIGHTNING_CLOUD_PROJECT_ID")),
         "lightning_ai"),
        (lambda: bool(os.environ.get("RUNPOD_POD_ID")),
         "runpod"),
        (lambda: bool(os.environ.get("VAST_CONTAINERLABEL")),
         "vast_ai"),
        (lambda: bool(os.environ.get("PAPERSPACE_FQDN")),
         "paperspace"),
        (lambda: bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
         "kaggle"),
        (lambda: bool(os.environ.get("COLAB_BACKEND_VERSION")),
         "google_colab"),
        (lambda: bool(os.environ.get("AZURE_ML_MODEL_DIR")),
         "azure_ml"),
        (lambda: bool(os.environ.get("SM_MODEL_DIR")),
         "aws_sagemaker"),
        (lambda: bool(os.environ.get("GCP_PROJECT")),
         "gcp_vertex"),
    ]
    for condition, name in checks:
        try:
            if condition():
                return name
        except Exception:
            pass

    # EC2 metadata probe
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/", timeout=1)
        return "aws_ec2"
    except Exception:
        pass

    return "local"


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    return round(kb / 1e6, 1)
    except Exception:
        pass
    return 0.0


def _disk_free_gb() -> float:
    try:
        import shutil as sh
        total, used, free = sh.disk_usage("/")
        return round(free / 1e9, 1)
    except Exception:
        return 0.0


def print_banner(sys_info: dict, torch_ver: str, tf_ver: str):
    gpu_line = ""
    if sys_info["cuda"]:
        cnt  = sys_info["gpu_count"]
        name = sys_info["gpu_name"]
        cv   = sys_info["cuda_version"] or "?"
        gpu_line = f"  GPU       : {G}{cnt}× {name}  |  CUDA {cv}{R}"
    elif sys_info["mps"]:
        gpu_line = f"  GPU       : {G}Apple Silicon MPS{R}"
    else:
        gpu_line = f"  GPU       : {Y}None detected — CPU mode{R}"

    print(f"""{C}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║           Universal DS / ML / AI Environment — Google Colab Mirror          ║
║     PyTorch {torch_ver:<10}  ·  TensorFlow {tf_ver:<10}  ·  Python {sys_info['python']:<8}    ║
╚══════════════════════════════════════════════════════════════════════════════╝{R}
  OS        : {sys_info['os']} {sys_info['os_version'][:40]}
  Arch      : {sys_info['arch']}
  Platform  : {sys_info['env']}
  RAM       : {sys_info['ram_gb']} GB
  Disk free : {sys_info['disk_free_gb']} GB
{gpu_line}""")

    if sys_info.get("force_cpu"):
        print(f"  Mode      : {Y}CPU-only (forced){R}")
    print()
