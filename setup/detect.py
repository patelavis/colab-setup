"""
System detection: OS, GPU (CUDA/MPS), RAM, disk, runtime platform.
Matches what the Colab Docker image itself would detect.
"""
import os, platform, shutil, subprocess, urllib.request
from .utils import ok, warn, info, G, Y, R, BOLD, C
from .packages import (
    COLAB_PYTHON, COLAB_CUDA, COLAB_NVIDIA_DRIVER,
    TENSORFLOW_VER, TORCH_VER,
)


def detect_system(force_cpu: bool = False) -> dict:
    d = {
        "os"          : platform.system(),
        "os_version"  : platform.version(),
        "arch"        : platform.machine(),
        "python"      : platform.python_version(),
        "cuda"        : False,
        "cuda_version": None,
        "mps"         : False,
        "gpu_name"    : None,
        "gpu_count"   : 0,
        "env"         : _env(),
        "force_cpu"   : force_cpu,
        "ram_gb"      : _ram(),
        "disk_gb"     : _disk(),
    }
    if not force_cpu:
        _cuda(d)
        _mps(d)
    return d


def _cuda(d):
    if not shutil.which("nvidia-smi"):
        return
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().splitlines()
            d["cuda"]     = True
            d["gpu_name"] = lines[0].split(",")[0].strip()
            d["gpu_count"]= len(lines)
    except Exception: pass

    if d["cuda"]:
        try:
            r = subprocess.run(["nvcc", "--version"],
                               capture_output=True, text=True, timeout=10)
            if "release" in r.stdout:
                d["cuda_version"] = (
                    r.stdout.split("release ")[-1].split(",")[0].strip())
        except Exception:
            d["cuda_version"] = "unknown"


def _mps(d):
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        d["mps"] = True


def _env() -> str:
    checks = [
        (lambda: os.path.exists("/content"),                          "google_colab"),
        (lambda: os.path.exists("/databricks"),                       "databricks"),
        (lambda: bool(os.environ.get("LIGHTNING_CLOUD_PROJECT_ID")), "lightning_ai"),
        (lambda: bool(os.environ.get("RUNPOD_POD_ID")),              "runpod"),
        (lambda: bool(os.environ.get("VAST_CONTAINERLABEL")),        "vast_ai"),
        (lambda: bool(os.environ.get("PAPERSPACE_FQDN")),            "paperspace"),
        (lambda: bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),     "kaggle"),
        (lambda: bool(os.environ.get("AZURE_ML_MODEL_DIR")),         "azure_ml"),
        (lambda: bool(os.environ.get("SM_MODEL_DIR")),               "aws_sagemaker"),
        (lambda: bool(os.environ.get("GCP_PROJECT")),                "gcp_vertex"),
    ]
    for fn, name in checks:
        try:
            if fn(): return name
        except Exception: pass
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/", timeout=1)
        return "aws_ec2"
    except Exception: pass
    return "local"


def _ram() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception: pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    return round(int(line.split()[1]) / 1e6, 1)
    except Exception: pass
    return 0.0


def _disk() -> float:
    try:
        import shutil as sh
        return round(sh.disk_usage("/").free / 1e9, 1)
    except Exception: return 0.0


def print_banner(sys_info: dict):
    gpu = ""
    if sys_info["cuda"]:
        gpu = (f"  GPU       : {G}{sys_info['gpu_count']}× "
               f"{sys_info['gpu_name']}  |  CUDA {sys_info['cuda_version']}{R}")
    elif sys_info["mps"]:
        gpu = f"  GPU       : {G}Apple Silicon MPS{R}"
    else:
        gpu = f"  GPU       : {Y}None — CPU mode{R}"

    print(f"""{C}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║       Universal DS/ML/AI Environment — Exact Google Colab Mirror            ║
║                                                                              ║
║  Source  :  us-docker.pkg.dev/colab-images/public/runtime  (GPU image)      ║
║  Ref     :  github.com/googlecolab/backend-info  pip-freeze.txt             ║
║                                                                              ║
║  Colab runtime:  Python {COLAB_PYTHON}  ·  CUDA {COLAB_CUDA}  ·  Driver {COLAB_NVIDIA_DRIVER}       ║
║  Frameworks  :  torch {TORCH_VER}  ·  tensorflow {TENSORFLOW_VER}                  ║
╚══════════════════════════════════════════════════════════════════════════════╝{R}
  This machine:
  OS        : {sys_info['os']} {sys_info['os_version'][:35]}
  Arch      : {sys_info['arch']}
  Python    : {sys_info['python']}
  Platform  : {sys_info['env']}
  RAM       : {sys_info['ram_gb']} GB    Disk free: {sys_info['disk_gb']} GB
{gpu}""")
    if sys_info.get("force_cpu"):
        print(f"  Mode      : {Y}CPU-only (--cpu-only flag){R}")
    print()
