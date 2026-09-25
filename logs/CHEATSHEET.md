# Cheatsheet

Day-to-day commands, once setup is done. Run everything from `~/tissue-dynamics`.

---

## Which side am I on?

| | Host (macOS) | Container (Linux) |
|---|---|---|
| Prompt | `(tissue-host) andriclu@… %` | `root@a3f2b1c8:/work#` |
| `uname -s` | `Darwin` | `Linux` |
| Use for | PyTorch, Taichi, plots, notebooks | PyBullet, SurRoL, ROS |

**Rule:** PyBullet/SurRoL → container. PyTorch/Taichi → host.

---

## Starting a work session

**Host side:**
```bash
cd ~/tissue-dynamics
conda activate tissue-host
```

**Container side** (Docker Desktop must be running):
```bash
cd ~/tissue-dynamics
docker compose run --rm surrol              # interactive shell
docker compose run --rm surrol python container/some_script.py   # one command
```

`--rm` deletes the container on exit. Always use it. Anything outside `/work`
is lost — that is intentional.

---

## The common commands

```bash
# collect data
docker compose run --rm surrol python container/collect_retraction.py --episodes 20

# train
python host/train_dynamics.py --data data --epochs 100

# health checks
python host/verify_host.py
docker compose run --rm surrol python container/verify_container.py

# notebooks (host)
jupyter lab
```

---

## conda

```bash
conda activate tissue-host      # enter
conda deactivate                # leave
conda env list                  # what environments exist
conda list                      # packages in the active one

# after editing host/environment.yml
conda env update -f host/environment.yml --prune

# start over
conda env remove -n tissue-host
conda env create -f host/environment.yml
```

Installed something with `pip install` on the host and it worked? **Add it to
`host/environment.yml`** or it vanishes the next time you rebuild.

---

## Docker

```bash
docker compose build                # rebuild after editing the Dockerfile
docker compose build --no-cache     # rebuild ignoring the cache (slow, thorough)

docker ps                           # running containers
docker ps -a                        # including stopped ones
docker images                       # built images
docker system df                    # disk usage

docker system prune -a              # delete everything unused (frees GBs)
```

Same rule: anything you `pip install` inside a container is gone on exit. If it
helped, **add it to `docker/Dockerfile`** and rebuild.

---

## git

```bash
git status                  # what changed
git diff                    # exactly how it changed
git add -A                  # stage everything
git commit -m "message"     # save a snapshot
git log --oneline           # history

git checkout -- path/to/file    # discard changes to one file
git stash                       # shelve all changes temporarily
git stash pop                   # bring them back
```

Commit when something **works**, not when it is finished.

---

## Terminal

| Keys | Does |
|---|---|
| `Tab` | Complete a path — also verifies it exists |
| `Ctrl-C` | Stop the running command |
| `Ctrl-D` | Exit the shell (leaves the container) |
| `↑` / `↓` | Previous commands |
| `Ctrl-R` | Search command history |
| `Ctrl-A` / `Ctrl-E` | Jump to start / end of line |

---

## Fast diagnostics

```bash
# host: is the GPU reachable?
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "import taichi as ti; ti.init(arch=ti.metal)"

# host: am I ARM-native? (must print arm64)
python -c "import platform; print(platform.machine())"

# is Docker up?
docker info --format '{{.ServerVersion}} / {{.Architecture}}'

# container: quick physics check
docker compose run --rm surrol python -c \
  "import pybullet as p; p.connect(p.DIRECT); print('pybullet ok')"

# inspect a trajectory (host)
python -c "
import sys; sys.path.insert(0,'src')
from trajectory_io import load_trajectory, list_trajectories
t = load_trajectory(list_trajectories('data')[0]); print(t); print(t.notes)"
```

---

## When it is thoroughly broken

```bash
docker compose down
docker system prune -a
docker compose build --no-cache
```

Nothing in `~/tissue-dynamics` is affected — that lives on your Mac, not in the
container.

---

## Reference

- `SETUP_GUIDE.md` — the full walkthrough, with a troubleshooting section
- `src/trajectory_io.py` — the data format both sides share; read the docstring
- `docker/Dockerfile` — the container environment, fully commented
- `host/environment.yml` — the host environment
