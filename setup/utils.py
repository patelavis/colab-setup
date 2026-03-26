"""Shared terminal helpers."""
import platform, subprocess, sys

R    = "\033[0m";  BOLD = "\033[1m"
G    = "\033[92m"; Y    = "\033[93m"
RED  = "\033[91m"; C    = "\033[96m"
B    = "\033[94m"; DIM  = "\033[2m"

def _win_ansi():
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception: pass
_win_ansi()

def ok(m):   print(f"{G}  ✅  {m}{R}")
def warn(m): print(f"{Y}  ⚠️   {m}{R}")
def err(m):  print(f"{RED}  ❌  {m}{R}"); sys.exit(1)
def info(m): print(f"{B}  ℹ   {m}{R}")
def dim(m):  print(f"{DIM}      {m}{R}")

def step(n: int, total: int, msg: str):
    done = "█" * n + "░" * (total - n)
    print(f"\n{BOLD}{C}  [{n:02d}/{total:02d}] {done}  {msg}{R}")

def run_cmd(cmd: list, label: str, ignore_error=True) -> bool:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        warn(f"'{label}' exited {r.returncode}")
        return False
    ok(f"'{label}' done")
    return True

def pip_install(*pkgs, extra_args=None, label=None, ignore_error=True):
    if not pkgs: return True
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs)
    if extra_args: cmd += extra_args
    lbl = label or pkgs[0]
    dim(f"pip install {lbl}")
    return run_cmd(cmd, lbl, ignore_error=ignore_error)

def pip_batch(pkg_list: list, label: str):
    if not pkg_list: return
    display = [p for p in pkg_list if not p.startswith("-")]
    info(f"Installing [{label}]  {len(display)} packages")
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkg_list
    r   = subprocess.run(cmd)
    if r.returncode != 0:
        warn(f"Batch '{label}' failed — retrying one by one…")
        flags = [p for p in pkg_list if p.startswith("-")]
        for p in display:
            pip_install(p, extra_args=flags or None, label=p)
    else:
        ok(f"[{label}] installed")
