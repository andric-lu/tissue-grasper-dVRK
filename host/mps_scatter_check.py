#!/usr/bin/env python3
"""
mps_scatter_check.py -- is scatter-add on Apple's GPU correct, and is it fast enough?

    conda activate tissue-host
    python host/mps_scatter_check.py              # ~1 min
    python host/mps_scatter_check.py --json runs/phase0/mps_scatter.json

DECISION_LOG.md section 13.5 (brief study 0.4). HOST ONLY: imports torch.

WHY THIS EXISTS
---------------
Message passing is, computationally, a scatter-add: every edge computes a message
and every node sums the messages addressed to it. With torch that is
`index_add_` / `scatter_add_`. The usual graph libraries (torch_geometric,
torch_scatter) wrap compiled CUDA kernels and have no Metal path, so on this
machine message passing is hand-written on top of those two ops.

A scatter with DUPLICATE indices is exactly where a GPU kernel can go wrong: if
two parallel writes to the same node race, one is silently lost. The result is
a finite, plausible, wrong tensor. A network trained on it plateaus with no
error -- the same class of hazard as section 3.2's KMP_DUPLICATE_LIB_OK: a
defect that corrupts arithmetic quietly is worse than a crash. So this is
checked against the CPU before anything is built on it, not assumed.

WHAT "CORRECT" MEANS HERE
-------------------------
Float sums depend on order, so MPS and CPU may legitimately differ in the last
bits. Two tests handle that from opposite sides:

  * EXACT: small integers stored as float32 sum exactly in ANY order (up to
    2^24). So with integer-valued data MPS must equal CPU bit for bit, and any
    difference is a lost or doubled write -- no tolerance to argue about.
  * FLOAT: realistic (E, 128) messages compared at a relative tolerance.

And the comparator is shown FAILING: a deliberately permuted index must be
reported as a mismatch, or the "match" results above prove nothing.

THE BUDGET, AND A CORRECTION TO THE BRIEF
-----------------------------------------
The brief states the MPC requirement as "S x H x 30 = 230,000 node updates per
second" (S = 256 samples, H = 30 steps, 30 Hz control). 230,400 is the number of
MODEL FORWARD PASSES per second, not node updates. Each forward pass evaluates
every edge in every one of M message-passing blocks, so the requirement in the
unit this script can measure is

    edge-block evaluations / s  =  S * H * f_control * E * M

At n = 600 nodes, ~14 neighbours (E = 8,400) and M = 10 that is ~1.9e10 / s --
five orders of magnitude above the brief's figure. The measured throughput is
reported against the corrected number.

Every check prints PASS / WARN / FAIL / SKIP and exits non-zero on any FAIL.
"""

import argparse
import json
import os
import platform
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = []
facts = {}


def record(status, name, detail):
    results.append((status, name, detail))
    icon = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{icon}] {name}\n         {detail}")


def check(name):
    """Run a check, turning any exception into a FAIL rather than a traceback."""
    def wrap(fn):
        try:
            status, detail = fn()
        except ImportError as e:
            status, detail = "FAIL", f"not installed ({e.name}). Is the conda env active?"
        except Exception as e:  # noqa: BLE001 -- a check must report, not crash
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        record(status, name, detail)
        return fn
    return wrap


try:
    import torch
except ImportError:
    torch = None

MPS = torch is not None and torch.backends.mps.is_available()


def _sync():
    if MPS:
        torch.mps.synchronize()


def _index_add(n, index, src, device):
    out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=device)
    return out.index_add_(0, index.to(device), src.to(device))


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

@check("torch imports and MPS is available")
def _():
    import torch as t  # noqa: F401 -- ImportError becomes a FAIL via check()
    facts["torch"] = torch.__version__
    facts["platform"] = platform.platform()
    if not MPS:
        why = ("not built into this wheel" if not torch.backends.mps.is_built()
               else "built, but no Metal device reachable")
        return "FAIL", f"torch {torch.__version__}: MPS {why}. Every check below SKIPs."
    return "PASS", f"torch {torch.__version__}, MPS available, {platform.machine()}"


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------

@check("index_add_ with duplicate indices is EXACT on integer data")
def _():
    # WHY: catches a lost or doubled write on duplicate indices -- the race a
    # parallel scatter kernel can have. Integers < 2^24 sum exactly in float32 in
    # any order, so ANY difference from the CPU is a wrong write, not rounding.
    if not MPS:
        return "SKIP", "no MPS device"
    g = torch.Generator().manual_seed(0)
    cases = []
    for E, n in ((1_000, 50), (100_000, 1_000), (1_000_000, 600), (1_000_000, 1)):
        idx = torch.randint(0, n, (E,), generator=g)
        # 0..7 so the all-into-one-node case peaks at 7e6 < 2^24 = 16.8e6
        val = torch.randint(0, 8, (E,), generator=g).to(torch.float32)
        a = _index_add(n, idx, val, "cpu")
        b = _index_add(n, idx, val, "mps").cpu()
        if not torch.equal(a, b):
            bad = int((a != b).sum())
            return "FAIL", (f"E={E}, n={n}: {bad} nodes differ from CPU on EXACT "
                            f"integer data (max |diff| {float((a-b).abs().max())}). "
                            "Duplicate-index writes are being lost or doubled.")
        cases.append(f"E={E:,}/n={n}")
    return "PASS", "bit-identical to CPU: " + ", ".join(cases) + " (n=1: every edge into one node)"


# MPS's float32 error against a float64 reference may be at most this multiple
# of the CPU's own float32 error on the same sum.
#
# WHY A RATIO AND NOT A FIXED TOLERANCE. The first version of this check used a
# hand-picked relative tolerance of 1e-5 and FAILED at 3.6e-5 on E=1e6, n=600.
# Measured against float64 that case gives CPU float32 error 7.7e-5 relative and
# MPS 6.9e-5: the CPU was further from the truth than the GPU. Summing ~1,700
# float32 values per node accumulates rounding that grows with the number of
# terms, so any fixed tolerance is wrong at some E. The honest standard is "no
# worse than float32 on the CPU", measured on the same data each run.
FLOAT_ERROR_RATIO = 4.0


def _err_vs(ref64, x32):
    return float((x32.double() - ref64).abs().max())


@check("index_add_ / scatter_add_ on (E, 128) float messages: MPS no worse than CPU float32")
def _():
    # WHY: catches wrong results on realistic message tensors -- 2-D sources,
    # non-contiguous views, empty edge sets -- which the 1-D exact test does
    # not exercise. Both devices are scored against a float64 reference, so
    # order-dependent rounding is measured rather than guessed at.
    if not MPS:
        return "SKIP", "no MPS device"
    g = torch.Generator().manual_seed(1)
    worst, lines = 0.0, []
    for E, n in ((1_000, 80), (100_000, 1_800), (1_000_000, 600)):
        idx = torch.randint(0, n, (E,), generator=g)
        msg = torch.randn(E, 128, generator=g)
        ex = idx[:, None].expand(E, 128)
        nc_i, nc_m = idx[::2], msg[::2]
        assert not nc_m.is_contiguous()
        variants = {
            "index_add_": lambda dev, dt: torch.zeros(n, 128, dtype=dt, device=dev)
                .index_add_(0, idx.to(dev), msg.to(dev, dt)),
            "scatter_add_": lambda dev, dt: torch.zeros(n, 128, dtype=dt, device=dev)
                .scatter_add_(0, ex.to(dev), msg.to(dev, dt)),
            "non-contiguous": lambda dev, dt: torch.zeros(n, 128, dtype=dt, device=dev)
                .index_add_(0, nc_i.to(dev), nc_m.to(dev, dt)),
        }
        for name, op in variants.items():
            ref = op("cpu", torch.float64)
            e_cpu = _err_vs(ref, op("cpu", torch.float32))
            e_mps = _err_vs(ref, op("mps", torch.float32).cpu())
            ratio = e_mps / max(e_cpu, 1e-12)
            worst = max(worst, ratio)
            if ratio > FLOAT_ERROR_RATIO:
                return "FAIL", (f"{name} E={E:,} n={n}: MPS error {e_mps:.2e} vs CPU "
                                f"float32 {e_cpu:.2e} ({ratio:.1f}x > {FLOAT_ERROR_RATIO}x)")
        lines.append(f"E={E:,}: CPU-f32 err {e_cpu:.1e}, MPS {e_mps:.1e}")

        # THE CRITERION IS SHOWN TO FAIL, in place: drop ONE message on the MPS
        # side. That is an O(1) error against O(1e-4) rounding, and the ratio
        # must flag it -- or the ratio is too lax to mean anything.
        if E == 1_000_000:
            ref = variants["index_add_"]("cpu", torch.float64)
            e_cpu = _err_vs(ref, variants["index_add_"]("cpu", torch.float32))
            dropped = torch.zeros(n, 128, device="mps").index_add_(
                0, idx[1:].to("mps"), msg[1:].to("mps")).cpu()
            if _err_vs(ref, dropped) <= FLOAT_ERROR_RATIO * e_cpu:
                return "FAIL", ("ONE dropped message was NOT detected by the float "
                                "criterion -- it is not discriminating at this E")
            lines.append(f"one dropped message detected ({_err_vs(ref, dropped) / e_cpu:.0f}x CPU rounding)")

    empty = _index_add(10, torch.empty(0, dtype=torch.long), torch.empty(0, 128), "mps")
    if float(empty.abs().sum().cpu()) != 0.0:
        return "FAIL", "empty edge set produced a nonzero aggregate on MPS"
    return "PASS", (f"worst MPS/CPU error ratio {worst:.2f} (limit {FLOAT_ERROR_RATIO}); "
                    + "; ".join(lines) + "; empty edge set gives zeros")


@check("int32 indices behave the same on MPS as on CPU")
def _():
    # WHY: catches a device-dependent dtype path. Graph builders often emit int32
    # to halve index memory; if CPU accepts it and MPS silently reinterprets it,
    # the edge list is wrong only on the GPU.
    if not MPS:
        return "SKIP", "no MPS device"
    idx = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int32)
    val = torch.ones(6)
    outcomes = {}
    for dev in ("cpu", "mps"):
        try:
            outcomes[dev] = _index_add(3, idx, val, dev).cpu().tolist()
        except Exception as e:  # noqa: BLE001
            outcomes[dev] = f"{type(e).__name__}"
    if outcomes["cpu"] != outcomes["mps"]:
        return "FAIL", f"CPU gave {outcomes['cpu']}, MPS gave {outcomes['mps']}"
    if isinstance(outcomes["cpu"], str):
        return "SKIP", f"int32 index rejected on both devices ({outcomes['cpu']}); use int64"
    return "PASS", f"int32 accepted on both, identical result {outcomes['cpu']}"


@check("gradients through index_select + index_add_: MPS no worse than CPU float32")
def _():
    # WHY: training runs backward, and the backward of a scatter is a gather
    # (and vice versa) -- a different kernel from the forward. A correct forward
    # with a wrong gradient trains to a plateau with no error.
    #
    # Same standard as the float forward check, for the same measured reason:
    # the first version used a hand-picked 1e-4 and FAILED at 1.9e-4. dW sums
    # 8,400 edge contributions per entry, so float32 rounding on ANY device
    # grows past a fixed tolerance. Score both devices against float64 instead.
    if not MPS:
        return "SKIP", "no MPS device"
    g = torch.Generator().manual_seed(2)
    n, E, H = 600, 8_400, 128
    senders = torch.randint(0, n, (E,), generator=g)
    receivers = torch.randint(0, n, (E,), generator=g)
    x0 = torch.randn(n, H, generator=g)
    W = torch.randn(H, 2 * H, generator=g) * 0.05
    probe = torch.randn(n, H, generator=g)

    def grads(dev, dtype, recv=receivers):
        # detach().clone() BEFORE requires_grad_: `x0.to("cpu")` returns x0
        # itself, so without the clone the CPU pass flips requires_grad on the
        # shared x0/W, and a later `.to("mps")` yields a NON-leaf whose .grad is
        # never populated. The first run crashed exactly that way.
        x = x0.detach().clone().to(dev, dtype).requires_grad_(True)
        w = W.detach().clone().to(dev, dtype).requires_grad_(True)
        s, r = senders.to(dev), recv.to(dev)
        m = torch.relu(torch.cat([x.index_select(0, s), x.index_select(0, r)], 1) @ w.T)
        agg = torch.zeros(n, H, device=dev, dtype=dtype).index_add_(0, r, m)
        (agg * probe.to(dev, dtype)).sum().backward()
        return x.grad.cpu(), w.grad.cpu()

    ref = grads("cpu", torch.float64)
    cpu32 = grads("cpu", torch.float32)
    mps32 = grads("mps", torch.float32)
    e_cpu = max(_err_vs(a, b) for a, b in zip(ref, cpu32))
    e_mps = max(_err_vs(a, b) for a, b in zip(ref, mps32))
    ratio = e_mps / max(e_cpu, 1e-12)
    if ratio > FLOAT_ERROR_RATIO:
        return "FAIL", (f"gradient error MPS {e_mps:.2e} vs CPU float32 {e_cpu:.2e} "
                        f"({ratio:.1f}x > {FLOAT_ERROR_RATIO}x)")

    # Shown to FAIL: route ONE edge to the wrong receiver on the MPS side.
    bad = receivers.clone()
    bad[0] = (bad[0] + 1) % n
    e_bad = max(_err_vs(a, b) for a, b in zip(ref, grads("mps", torch.float32, bad)))
    if e_bad <= FLOAT_ERROR_RATIO * e_cpu:
        return "FAIL", "ONE mis-routed edge was NOT detected in the gradients"
    return "PASS", (f"d/dx, d/dW: MPS err {e_mps:.1e}, CPU-f32 err {e_cpu:.1e} "
                    f"(ratio {ratio:.2f}); one mis-routed edge detected "
                    f"({e_bad / e_cpu:.0f}x CPU rounding); n={n}, E={E:,}, H={H}")


@check("the comparator itself detects a wrong scatter")
def _():
    # WHY: a check only ever shown passing has not been shown to work (CLAUDE.md).
    # Scatter with a PERMUTED index on MPS -- a stand-in for a kernel that sends
    # messages to the wrong nodes -- and the same comparison used above must
    # report a mismatch. If it doesn't, every PASS above is meaningless.
    if not MPS:
        return "SKIP", "no MPS device"
    g = torch.Generator().manual_seed(3)
    E, n = 100_000, 1_000
    idx = torch.randint(0, n, (E,), generator=g)
    val = torch.randint(0, 8, (E,), generator=g).to(torch.float32)
    ref = _index_add(n, idx, val, "cpu")
    wrong = _index_add(n, idx[torch.randperm(E, generator=g)], val, "mps").cpu()
    if torch.equal(ref, wrong):
        return "FAIL", "permuted index was NOT detected -- the exact comparison is not discriminating"
    return "PASS", f"permuted index detected: {int((ref != wrong).sum())}/{n} nodes differ"


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------

def _block(n, E, H, device, train):
    """One GNS-shaped message-passing block: edge MLP on [e, v_s, v_r], scatter,
    node MLP on [v, agg]. 2 hidden layers of width H, LayerNorm, residuals."""
    import torch.nn as nn

    def mlp(i):
        return nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
                             nn.Linear(H, H), nn.LayerNorm(H)).to(device)

    edge_mlp, node_mlp = mlp(3 * H), mlp(2 * H)
    g = torch.Generator().manual_seed(4)
    s = torch.randint(0, n, (E,), generator=g).to(device)
    r = torch.randint(0, n, (E,), generator=g).to(device)
    v = torch.randn(n, H, generator=g).to(device)
    e = torch.randn(E, H, generator=g).to(device)
    params = list(edge_mlp.parameters()) + list(node_mlp.parameters())

    def step():
        if train:
            for p in params:
                p.grad = None
        with torch.set_grad_enabled(train):
            e2 = e + edge_mlp(torch.cat([e, v.index_select(0, s), v.index_select(0, r)], 1))
            agg = torch.zeros(n, H, device=device).index_add_(0, r, e2)
            v2 = v + node_mlp(torch.cat([v, agg], 1))
            if train:
                (v2.square().mean() + e2.square().mean()).backward()
    return step


def _edges_per_second(n, E, device, train, min_time=1.0, max_iter=30, H=128):
    step = _block(n, E, H, device, train)
    for _ in range(3):
        step()
    _sync()
    t0, it = time.perf_counter(), 0
    while it < max_iter and (time.perf_counter() - t0) < min_time:
        step()
        it += 1
    _sync()
    return E * it / (time.perf_counter() - t0)


@check("throughput: one message-passing block, edges per second")
def _():
    # WHY: sets Phase 6's MPC budget with a measured number instead of an
    # estimate, and tells section 13 early whether a graph at this node count can
    # close the control loop at all -- before a model is built around it.
    if not MPS:
        return "SKIP", "no MPS device"
    sizes = [("single graph n=600, deg 14", 600, 8_400),
             ("single graph n=1800, deg 20", 1_800, 36_000),
             ("MPPI batch S=256 x n=600", 256 * 600, 256 * 8_400)]
    rows = []
    for label, n, E in sizes:
        row = {"label": label, "n": n, "E": E}
        for dev in ("mps", "cpu"):
            for train in (False, True):
                if dev == "cpu" and E > 100_000:
                    continue            # minutes on CPU and not a decision input
                key = f"{dev}_{'train' if train else 'infer'}"
                try:
                    row[key] = _edges_per_second(n, E, dev, train)
                except RuntimeError as ex:       # out of memory on the batch row
                    row[key] = f"{type(ex).__name__}: {str(ex)[:80]}"
        rows.append(row)
    facts["throughput"] = rows

    # The corrected MPC requirement (module docstring): S*H*f*E*M.
    S, Hh, f, M = 256, 30, 30, 10
    E_ctrl = 8_400
    need = S * Hh * f * E_ctrl * M
    have = rows[2].get("mps_infer")
    facts["mpc_requirement_edge_blocks_per_s"] = need

    lines = []
    for row in rows:
        parts = [f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in row.items() if k.startswith(("mps", "cpu"))]
        lines.append(f"{row['label']}: " + ", ".join(parts))
    if not isinstance(have, float):
        return "WARN", " | ".join(lines) + " | batched inference did not run"
    ratio = have / need
    facts["mpc_budget_ratio"] = ratio
    summary = (f"MPC needs S*H*f*E*M = {need:.2e} edge-blocks/s (S={S}, H={Hh}, "
               f"f={f} Hz, E={E_ctrl}, M={M}); batched MPS inference gives "
               f"{have:.2e} -> {ratio:.2%} of budget")
    detail = "\n         ".join(lines + [summary])
    return ("PASS" if ratio >= 1.0 else "WARN"), detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", default="", help="also write results + facts here")
    args = ap.parse_args(argv)

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"results": results, "facts": facts}, fh, indent=2, default=str)
        print(f"wrote {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
