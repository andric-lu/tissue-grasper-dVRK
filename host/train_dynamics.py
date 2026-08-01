#!/usr/bin/env python3
"""
train_dynamics.py -- closes the loop: trains a dynamics model on the GPU from
data the container produced.

Run on the HOST (not in the container -- this is the half that uses the M3 GPU):

    conda activate tissue-host
    python host/train_dynamics.py --data data --epochs 40

WHAT THIS IS FOR
----------------
This is a deliberately plain baseline whose job is to prove the pipeline works
end to end: container writes .npz -> host reads it -> a model trains on Metal.
It is NOT the architecture you want for this problem.

WHY AN MLP IS THE WRONG FINAL ANSWER
------------------------------------
Flattening every node into one long vector throws away the mesh. The network has
to rediscover, from data, that node 47 is adjacent to node 48 -- something you
already know for free from the topology. It also means the model is locked to
one specific mesh resolution: retrain from scratch for a finer mesh.

The right family is a graph network: nodes are mesh vertices, edges are mesh
connectivity, and the model learns local interaction rules that generalise
across resolutions and geometries (see MeshGraphNets, Pfaff et al. 2020, which
is the standard starting point for learned deformable dynamics). The trajectory
format already stores `tissue_faces`/`tissue_tets` so you have the graph
structure waiting when you get there.

THE BASELINE MATTERS MORE THAN THE MODEL
----------------------------------------
This script always reports a constant-velocity baseline: "assume every node
keeps moving as it was". Soft tissue at 30 Hz is smooth, so that baseline is
strong. A neural network that does not clearly beat it has learned nothing,
and it is very easy to celebrate a small loss number without checking.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from trajectory_io import list_trajectories, load_trajectory  # noqa: E402


def pick_device() -> torch.device:
    """Prefer the Apple GPU; fall back to CPU with a loud warning."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: MPS unavailable, training on CPU. Run host/verify_host.py.")
    return torch.device("cpu")


def build_dataset(data_dir: str, horizon: int):
    """Load every episode and assemble one big supervised dataset.

    Predicts the DELTA in node positions rather than absolute positions. This
    matters more than it looks: absolute targets force the network to spend
    capacity memorising where the tissue sits in world coordinates, and the
    targets have a huge constant offset that dominates the loss. Deltas are
    zero-centred and small, which is what neural networks are good at.
    """
    files = list_trajectories(data_dir)
    if not files:
        raise SystemExit(
            f"No .npz files in {data_dir}. Collect some first:\n"
            "  docker compose run --rm surrol python container/collect_retraction.py --episodes 5")

    X, U, Y, ep_ids = [], [], [], []
    for i, f in enumerate(files):
        tr = load_trajectory(f)
        T, N = len(tr), tr.n_nodes
        if T <= horizon:
            print(f"  skipping {os.path.basename(f)}: only {T} steps")
            continue
        pos = tr.tissue_pos.astype(np.float32)
        vel = tr.tissue_vel.astype(np.float32)

        state = np.concatenate([pos.reshape(T, N * 3),
                                vel.reshape(T, N * 3),
                                tr.ee_pose.astype(np.float32)], axis=1)
        X.append(state[:-horizon])
        U.append(tr.action.astype(np.float32)[:-horizon])
        Y.append((pos[horizon:] - pos[:-horizon]).reshape(T - horizon, N * 3))
        ep_ids.append(np.full(T - horizon, i, np.int64))
        print(f"  {os.path.basename(f)}: {T} steps, {N} nodes")

    X, U, Y = np.concatenate(X), np.concatenate(U), np.concatenate(Y)
    return X, U, Y, np.concatenate(ep_ids), len(files)


class DynamicsMLP(nn.Module):
    """state + action -> change in node positions."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            # LayerNorm rather than BatchNorm: batch statistics are unreliable
            # here because consecutive samples come from the same trajectory
            # and are highly correlated, which is exactly what BatchNorm hates.
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=1,
                    help="predict this many steps ahead")
    ap.add_argument("--out", default="models/dynamics_mlp.pt")
    args = ap.parse_args()

    dev = pick_device()
    print(f"device: {dev}\nloading episodes from {args.data}/")
    X, U, Y, ep, n_eps = build_dataset(args.data, args.horizon)
    print(f"{len(X)} transitions from {n_eps} episode(s); "
          f"state={X.shape[1]}, action={U.shape[1]}, target={Y.shape[1]}")

    # Split by EPISODE, never by random shuffling of timesteps. Consecutive
    # frames are nearly identical, so a random split leaks almost-identical
    # frames into validation and produces a beautiful, meaningless val loss.
    n_val = max(1, n_eps // 5)
    val_mask = ep >= (n_eps - n_val)
    print(f"holding out {n_val} episode(s) for validation")

    inp = np.concatenate([X, U], axis=1)
    # Standardise inputs. Node coordinates are ~0.1 and velocities ~1.0; without
    # this the optimiser is fighting badly scaled gradients from step one.
    mu, sd = inp[~val_mask].mean(0), inp[~val_mask].std(0) + 1e-6
    inp = (inp - mu) / sd
    # Targets are scaled by a single scalar, not per-dimension: per-dimension
    # scaling would distort the relative importance of x/y/z and of nodes that
    # happen to move little.
    y_scale = float(np.abs(Y[~val_mask]).mean() + 1e-9)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    xtr, ytr = t(inp[~val_mask]), t(Y[~val_mask] / y_scale)
    xva, yva = t(inp[val_mask]), t(Y[val_mask] / y_scale)

    # Constant-velocity baseline, in metres, on the validation set.
    # Predicted displacement = velocity * dt, read straight out of the state.
    n_nodes = Y.shape[1] // 3
    vel_va = X[val_mask][:, n_nodes * 3: n_nodes * 6]
    dt = float(load_trajectory(list_trajectories(args.data)[0]).dt)
    baseline_rmse = float(np.sqrt(((vel_va * dt * args.horizon - Y[val_mask]) ** 2).mean()))
    zero_rmse = float(np.sqrt((Y[val_mask] ** 2).mean()))
    print(f"\nbaselines on validation (RMSE, metres):"
          f"\n  predict no motion      : {zero_rmse*1000:.4f} mm"
          f"\n  constant velocity      : {baseline_rmse*1000:.4f} mm")

    model = DynamicsMLP(xtr.shape[1], ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    print(f"\ntraining {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot, t0 = 0.0, time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Deformable data has occasional spikes (a node snaps, a contact
            # resolves violently). Without clipping, one bad batch can wreck
            # the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(xva)
            val_rmse = float(torch.sqrt(((pred - yva) ** 2).mean()).item()) * y_scale
        flag = ""
        if val_rmse < best:
            best, flag = val_rmse, "  *"
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            # Save the normalisation constants alongside the weights. A model
            # without them is useless -- you cannot reproduce the input scaling.
            torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                        "y_scale": y_scale, "horizon": args.horizon,
                        "n_nodes": n_nodes}, args.out)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {tot/len(xtr):.5f}  "
                  f"val {val_rmse*1000:7.4f} mm  ({time.time()-t0:.1f}s){flag}")

    print(f"\nbest validation RMSE: {best*1000:.4f} mm")
    if best < baseline_rmse:
        print(f"beats constant-velocity baseline by "
              f"{100*(1-best/baseline_rmse):.1f}%  -- the model learned something")
    else:
        print("DOES NOT beat the constant-velocity baseline. The model has not\n"
              "learned useful dynamics. Usual causes: too few episodes, the\n"
              "logging rate is so high that motion per step is near zero, or\n"
              "an MLP cannot exploit this mesh -- move to a graph network.")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
