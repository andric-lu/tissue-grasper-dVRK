#!/usr/bin/env python3
"""
verify_host.py -- confirm the macOS-native environment can reach the GPU.

Run this after creating the conda environment:

    conda activate tissue-host
    python host/verify_host.py

Every check prints PASS, WARN, or FAIL with an explanation. The two checks that
really matter are PyTorch/MPS and Taichi/Metal -- if either says FAIL you are
silently running on CPU, which works but is roughly 10-30x slower and you will
not notice until you wonder why training takes all night.
"""

import platform
import subprocess
import sys

results = []


def check(name):
    """Decorator that runs a check function and records PASS/WARN/FAIL."""
    def wrap(fn):
        try:
            status, detail = fn()
        except ImportError as e:
            status, detail = "FAIL", f"not installed ({e.name}). Is the conda env active?"
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results.append((status, name, detail))
        icon = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
        print(f"[{icon}] {name}\n         {detail}")
        return fn
    return wrap


print("=" * 72)
print("HOST ENVIRONMENT CHECK  (macOS-native, GPU side)")
print("=" * 72)


@check("Python is ARM64-native, not running under Rosetta")
def _():
    m = platform.machine()
    # If this says x86_64 on an M-series Mac, you installed an Intel build of
    # conda and everything downstream will be emulated and slow, and Metal
    # backends may refuse to initialise. Fix: reinstall Miniforge.
    if m == "arm64":
        return "PASS", f"machine={m}, python={platform.python_version()}"
    return "FAIL", (f"machine={m} -- this is an Intel build running under Rosetta. "
                    "Uninstall this conda and install Miniforge (ARM64).")


@check("macOS version supports Metal compute")
def _():
    v = platform.mac_ver()[0]
    if not v:
        return "FAIL", "not running on macOS"
    major = int(v.split(".")[0])
    if major >= 13:
        return "PASS", f"macOS {v}"
    return "WARN", f"macOS {v} -- MPS needs 12.3+, and 13+ is far less buggy"


@check("PyTorch is installed and the MPS (GPU) backend is available")
def _():
    import torch
    if not torch.backends.mps.is_available():
        why = ("built without MPS support"
               if not torch.backends.mps.is_built()
               else "built with MPS but the runtime is unavailable")
        return "FAIL", f"torch {torch.__version__} -- {why}. You would be on CPU only."
    return "PASS", f"torch {torch.__version__}, MPS available"


@check("A real tensor computation runs on the GPU")
def _():
    import torch
    if not torch.backends.mps.is_available():
        return "FAIL", "skipped, MPS unavailable"
    # "Available" and "actually works" are different things. Multiply two
    # 2048x2048 matrices on the GPU and check the result against CPU.
    a = torch.randn(2048, 2048, device="mps")
    b = torch.randn(2048, 2048, device="mps")
    c = (a @ b).cpu()
    ref = a.cpu() @ b.cpu()
    err = (c - ref).abs().max().item()
    if err > 1e-2:
        return "FAIL", f"GPU and CPU results disagree by {err:.3g}"
    return "PASS", f"2048x2048 matmul on MPS, max error {err:.2e}"


@check("Taichi is installed and the Metal backend initialises")
def _():
    import taichi as ti
    # arch=ti.metal asks specifically for the Apple GPU. If Metal were
    # unavailable Taichi would fall back to CPU with only a log line, so the
    # backend is checked explicitly afterwards.
    ti.init(arch=ti.metal, log_level=ti.ERROR)
    backend = str(ti.lang.impl.current_cfg().arch)
    if "metal" not in backend.lower():
        return "FAIL", f"Taichi fell back to {backend} -- MPM will run on CPU"
    return "PASS", f"taichi {ti.__version__}, backend={backend}"


@check("A Taichi kernel actually executes on Metal")
def _():
    import taichi as ti
    f = ti.field(ti.f32, shape=1024)

    @ti.kernel
    def fill():
        for i in f:
            f[i] = i * 2.0

    fill()
    v = f.to_numpy()
    if abs(v[100] - 200.0) > 1e-4:
        return "FAIL", f"kernel produced wrong value: f[100]={v[100]}"
    return "PASS", f"kernel ran, f[100]={v[100]:.1f} as expected"


@check("The shared trajectory format round-trips")
def _():
    # Imports src/trajectory_io.py -- the file both this environment and the
    # container use. If this fails, the two halves cannot exchange data.
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import trajectory_io  # noqa: F401
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "src", "trajectory_io.py")],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return "FAIL", r.stderr.strip().splitlines()[-1] if r.stderr else "self-test failed"
    return "PASS", "src/trajectory_io.py self-test passed"


@check("Docker is installed and its daemon is running")
def _():
    # Not strictly part of the host env, but this is the thing most likely to
    # be quietly not-running when you try to start the container.
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "WARN", "docker not responding -- is Docker Desktop open?"
    return "PASS", f"docker daemon {r.stdout.strip()}"


print("=" * 72)
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
n_warn = sum(1 for s, _, _ in results if s == "WARN")
if n_fail:
    print(f"{n_fail} check(s) FAILED, {n_warn} warning(s).")
    print("Fix failures before continuing -- see SETUP_GUIDE.md, Troubleshooting.")
    sys.exit(1)
print(f"All checks passed ({n_warn} warning(s)). Host environment is ready.")
