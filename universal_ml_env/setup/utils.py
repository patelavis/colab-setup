"""
Shared terminal helpers: colours, logging, pip wrappers.
"""
import platform, subprocess, sys

# ── ANSI colours ─────────────────────────────────────────────────────
R    = "\033[0m"
BOLD = "\033[1m"
G    = "\033[92m"
Y    = "\033[93m"
RED  = "\033[91m"
C    = "\033[96m"
B    = "\033[94m"
DIM  = "\033[2m"

def _enable_win_ansi():
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_win_ansi()


def ok(msg):   print(f"{G}  ✅  {msg}{R}")
def warn(msg): print(f"{Y}  ⚠️   {msg}{R}")
def err(msg):  print(f"{RED}  ❌  {msg}{R}"); sys.exit(1)
def info(msg): print(f"{B}  ℹ   {msg}{R}")
def dim(msg):  print(f"{DIM}      {msg}{R}")

def step(n: int, total: int, msg: str):
    bar = "█" * n + "░" * (total - n)
    print(f"\n{BOLD}{C}  [{n:02d}/{total:02d}] {bar}  {msg}{R}")


def run_cmd(cmd: list, label: str, ignore_error: bool = True) -> bool:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        warn(f"'{label}' failed (exit {result.returncode})")
        return False
    ok(f"'{label}' done")
    return True


def pip_install(*packages, extra_args: list = None,
                label: str = None, ignore_error: bool = True) -> bool:
    if not packages:
        return True
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(packages)
    if extra_args:
        cmd += extra_args
    lbl = label or packages[0]
    dim(f"pip install {lbl}")
    return run_cmd(cmd, lbl, ignore_error=ignore_error)


def pip_batch(package_list: list, label: str):
    """
    Install a list of packages in one pip call.
    Falls back to one-by-one if the batch fails.
    """
    if not package_list:
        return
    # Strip flag strings (e.g. --extra-index-url ...) from display
    display = [p for p in package_list if not p.startswith("-")]
    info(f"Installing [{label}]  →  {len(display)} packages")
    cmd    = [sys.executable, "-m", "pip", "install", "-q"] + package_list
    result = subprocess.run(cmd)
    if result.returncode != 0:
        warn(f"Batch '{label}' failed — retrying one by one…")
        flags = [p for p in package_list if p.startswith("-")]
        for pkg in display:
            pip_install(pkg, extra_args=flags if flags else None, label=pkg)
    else:
        ok(f"[{label}] done")
