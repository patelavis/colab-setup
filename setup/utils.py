"""
Shared terminal output utilities.
"""
import platform, subprocess, sys

# ── ANSI colour codes ────────────────────────────────────────────────
R    = "\033[0m"
BOLD = "\033[1m"
G    = "\033[92m"
Y    = "\033[93m"
RED  = "\033[91m"
C    = "\033[96m"
B    = "\033[94m"

def _enable_win_ansi():
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_win_ansi()


def ok(msg):
    print(f"{G}  ✅  {msg}{R}")

def warn(msg):
    print(f"{Y}  ⚠️   {msg}{R}")

def err(msg):
    print(f"{RED}  ❌  {msg}{R}")
    sys.exit(1)

def info(msg):
    print(f"{B}  ℹ   {msg}{R}")

def step(n: int, total: int, msg: str):
    print(f"{BOLD}{C}\n  ── [{n}/{total}] {msg} ──{R}")


def run_cmd(cmd: list, label: str, ignore_error: bool = False) -> bool:
    """Run a subprocess command, print result."""
    result = subprocess.run(cmd)
    if result.returncode != 0:
        if ignore_error:
            warn(f"'{label}' failed but continuing…")
        else:
            warn(f"'{label}' exited with code {result.returncode}")
        return False
    ok(f"'{label}' done")
    return True


def pip_install(*packages, extra_args: list = None, label: str = None,
                ignore_error: bool = True):
    """Install one or more packages via pip (quiet mode)."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(packages)
    if extra_args:
        cmd += extra_args
    lbl = label or (packages[0] if packages else "pip install")
    info(f"Installing: {lbl}")
    return run_cmd(cmd, lbl, ignore_error=ignore_error)


def pip_batch(package_list: list, label: str):
    """
    Install a list of packages in one pip call for better dependency
    resolution.  Falls back to one-by-one on failure.
    """
    if not package_list:
        return
    info(f"Installing batch [{label}]  ({len(package_list)} packages)…")
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + package_list
    result = subprocess.run(cmd)
    if result.returncode != 0:
        warn(f"Batch '{label}' had issues — retrying one by one…")
        for pkg in package_list:
            pip_install(pkg, label=pkg)
    else:
        ok(f"Batch '{label}' done")
