# Setting up a soft-tissue dynamics research environment on an M3 Max MacBook Pro

A complete walkthrough, assuming no prior command-line experience. Every command
is explained: what it does, why it's needed, and what should happen.

**Time:** 60–90 minutes, most of it waiting for downloads and compiles.
**You need:** the laptop, an internet connection, and your Mac login password.

---

## How to use this guide

Work through it in order. Each part ends with a **Checkpoint** — a command that
proves the part worked. Do not move on past a failed checkpoint; every later step
assumes the earlier ones succeeded, and debugging is far easier when you know
exactly which step broke.

**Reading the commands.** Command blocks look like this:

```bash
cd ~/tissue-dynamics
```

Type or paste exactly that. You will see other tutorials online write `$ cd ...`
— the `$` is a prompt symbol representing "the terminal", not something you type.
This guide omits it to avoid the confusion.

**When something fails,** read the *last* line of the error first. Errors print
a stack of context, and the actual problem is usually at the bottom, not the top.
Part 7 covers the failures you're most likely to hit.

---

## Part 0 — What you are building, and why it looks complicated

You are building **two separate Python environments** on one laptop.

```
   ┌─────────────────────── your MacBook ────────────────────────┐
   │                                                             │
   │   HOST environment                CONTAINER environment     │
   │   (macOS-native)                  (Linux, inside Docker)    │
   │   ─────────────────                ────────────────────     │
   │   conda env "tissue-host"          Ubuntu 22.04 (ARM64)     │
   │   Python 3.11                      Python 3.10              │
   │                                                             │
   │   PyTorch  → Apple GPU             PyBullet physics         │
   │   Taichi   → Apple GPU             SurRoL + dVRK models     │
   │   analysis, plots, notebooks       ROS later                │
   │                                                             │
   │   CAN reach the M3 Max GPU         CANNOT reach the GPU     │
   │                                                             │
   │              ↓ writes                    ↓ reads            │
   │        ┌──────────────────────────────────────────┐         │
   │        │  ~/tissue-dynamics/data/*.npz            │         │
   │        │  shared folder, visible to both sides    │         │
   │        └──────────────────────────────────────────┘         │
   └─────────────────────────────────────────────────────────────┘
```

**Why two, and not one?** Two constraints pull in opposite directions.

1. *SurRoL wants Linux.* It's built and tested on Ubuntu against a specific,
   now-dated set of package versions. Reproducing that on macOS means fighting
   the package manager repeatedly. A container gives you Ubuntu exactly.

2. *GPU acceleration only works outside the container.* Apple's GPU is reached
   through Metal, and Metal is a macOS API. A Linux container — even one running
   natively on your ARM chip — has no path to it. Taichi's Metal backend and
   PyTorch's MPS backend both require running directly on macOS.

So: physics in the container, learning on the host, one shared folder between.

**What a container actually is.** Not a virtual machine. A VM boots a whole
second operating system with its own kernel — slow to start, heavy on memory. A
container is a bundle of files (an Ubuntu filesystem, Python, your packages) that
runs as *ordinary processes on your Mac's kernel*, isolated so they only see
their own files. Startup is a second or two, and CPU speed is native.

The one thing that makes containers useful here: the container's contents are
described entirely by a text file (the `Dockerfile`). Delete the container,
rebuild from that file, and you get a byte-identical environment. When something
breaks beyond repair, you throw it away and rebuild instead of debugging.

**Three words you'll see constantly:**

| Term | What it is | Analogy |
|---|---|---|
| **Dockerfile** | Text recipe listing what to install | A recipe |
| **image** | The built, frozen filesystem | A cake you baked |
| **container** | A running instance of an image | A slice you're eating |

You build an image once. You start and throw away containers constantly.

---

## Part 1 — Terminal basics (15 minutes, skip if you're comfortable)

Everything below happens in Terminal. Open it: press `⌘ Space`, type
`Terminal`, press Return.

You'll see something like `andriclu@MacBook-Pro ~ %`. That's the **prompt**:
your username, your computer, your current folder, then `%`. It's the computer
saying "ready".

The `~` is shorthand for your home folder, `/Users/andriclu`.

### The five commands that matter

```bash
pwd
```
**p**rint **w**orking **d**irectory — "where am I?" The terminal is always
"inside" one folder, and commands act relative to it. When a command mysteriously
can't find a file, this is the first thing to check.

```bash
ls
```
**l**i**s**t — what's in this folder. Add `-la` (`ls -la`) to also show hidden
files, which on macOS means anything starting with a dot. Several important
config files are hidden, so get used to `ls -la`.

```bash
cd ~/tissue-dynamics
```
**c**hange **d**irectory — move into a folder. Special targets:
- `cd ~` → home folder
- `cd ..` → up one level
- `cd -` → back to where you just were

```bash
mkdir -p ~/tissue-dynamics/data
```
**m**a**k**e **dir**ectory. The `-p` flag means "create parent folders as needed,
and don't complain if it already exists". Almost always what you want.

```bash
cat somefile.txt
```
Print a file's contents. Useful for quickly checking a config file.

### Four keyboard habits worth building now

- **Tab completion.** Type `cd ~/tis` and press Tab — the shell completes it.
  This isn't just speed; it *verifies the path exists*. If Tab doesn't complete,
  you typed something wrong. Use it constantly.
- **Ctrl-C** stops whatever is currently running. Your escape hatch when
  something hangs or runs away.
- **Up arrow** cycles through previous commands. `Ctrl-R` then typing searches
  them.
- **Ctrl-D** or typing `exit` closes the current shell. You'll use this to leave
  the container.

### Two conventions that will confuse you once

**Flags** are the `-p`, `-la`, `--platform` bits. Single dash + one letter
(`-p`), or double dash + a word (`--platform`). They modify a command's
behaviour. `ls -la` is `ls` with flags `l` and `a` combined.

**Everything is case-sensitive.** `Data` and `data` are different folders. macOS
Finder hides this from you; the terminal does not.

> **Checkpoint 1.** Run these three:
> ```bash
> pwd
> ls
> echo "terminal works"
> ```
> You should see your home path, a list of folders (Desktop, Documents…), then
> `terminal works`. If so, you have everything you need.

---

## Part 2 — Foundations

Three pieces of software the rest of the setup depends on.

### 2.1 Xcode Command Line Tools

```bash
xcode-select --install
```

A dialog box appears — click **Install**, accept the licence. Takes 5–15 minutes.

**What it is:** Apple's C/C++ compilers (`clang`), `make`, `git`, and the system
header files. Roughly 1.5 GB.

**Why you need it:** several Python packages have no prebuilt version for Apple
Silicon and ship as C source code that gets compiled during installation. Without
a compiler, `pip install` fails with a wall of errors that never mention the
actual cause. This also installs `git`, which you need in Part 3.

If it says "already installed", good, move on.

Verify:

```bash
xcode-select -p
```
Should print a path ending in `CommandLineTools` or `Xcode.app/…/Developer`.

### 2.2 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It'll ask for your Mac password (the screen shows nothing as you type — that's
normal, not a broken keyboard) and print what it's about to do. Press Return to
continue.

**What it is:** a package manager for macOS. macOS has no built-in way to install
developer software; Homebrew fills that gap. `brew install X` handles downloading,
placing, and updating X.

**Why:** you could install everything by hand from websites, but then nothing
tracks versions and upgrades become archaeology.

**The step everyone misses.** On Apple Silicon, Homebrew installs to
`/opt/homebrew`, which your shell doesn't search by default. Run these two lines
so it does:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

The first appends a line to `~/.zprofile`, a script your shell runs at every
login, so this is permanent. The second applies it to the terminal you're in
right now, so you don't have to restart.

*(`>>` means "append to file". Be careful: a single `>` means "overwrite file",
and typing `>` where you meant `>>` will silently erase your shell config.)*

Verify:

```bash
brew --version
```
Should print `Homebrew 4.x.x`. If it says "command not found", the two lines
above didn't take — close Terminal, reopen it, try again.

### 2.3 Tell git who you are

Git ships with the Xcode tools, but needs your identity before it will record
anything:

```bash
git config --global user.name "Andric Lu"
git config --global user.email "andriclu00@gmail.com"
```

**Why now:** git refuses to make a commit without these, and it's irritating to
discover that mid-task later. `--global` means "for all projects on this machine".

> **Checkpoint 2.**
> ```bash
> brew --version && git --version && clang --version | head -1
> ```
> Three version lines, no errors. (`&&` means "run the next command only if the
> previous one succeeded".)

---

## Part 3 — The project folder

Put the downloaded `tissue-dynamics` folder in your home directory, so its path
is exactly `~/tissue-dynamics`. In Finder, drag it there; or if it's in
Downloads:

```bash
mv ~/Downloads/tissue-dynamics ~/tissue-dynamics
cd ~/tissue-dynamics
ls -la
```

You should see:

```
.gitignore          docker/             host/
CHEATSHEET.md       docker-compose.yml  src/
README.md           container/
SETUP_GUIDE.md
```

### What each folder is for

| Path | Runs where | Contents |
|---|---|---|
| `host/` | macOS | GPU-side code: training, analysis, `environment.yml` |
| `container/` | Linux | Simulation code: data collection, verification |
| `src/` | **both** | Shared code, above all the trajectory format |
| `docker/` | — | The `Dockerfile` describing the Linux environment |
| `data/` | **both** | Trajectories. Created on first run. |

`src/` being shared is the important one. `src/trajectory_io.py` defines the file
format both halves speak. Because it's a single file used by both, the two sides
cannot drift apart — if you change the format, both sides change together.

### Put it under version control

```bash
cd ~/tissue-dynamics
git init
git add .
git commit -m "Initial setup: host env, container, trajectory format"
```

**What just happened.** `git init` created a hidden `.git` folder that will
record the history of this project. `git add .` staged every file (`.` = "this
folder, recursively") except those listed in `.gitignore`. `git commit` saved a
permanent snapshot with a message.

**Why bother before writing any research code:** three months in, you will change
a physics parameter, get different results, and need to know exactly what changed.
Without version control that question is unanswerable. It also means you can
experiment fearlessly — anything committed can be recovered.

The habit to build: commit whenever something *works*. Not when it's finished —
when it works.

```bash
git status      # what's changed since the last commit
git log --oneline   # history so far
```

> **Checkpoint 3.** `git log --oneline` shows one commit. `git status` says
> "nothing to commit, working tree clean".

---

## Part 4 — The host environment (the GPU side)

### 4.1 Install Miniforge

```bash
brew install --cask miniforge
```

**What conda is:** a tool that creates isolated Python environments. Each
environment has its own Python and its own packages, so a project needing
NumPy 1.23 and a project needing NumPy 2.0 can coexist. Without isolation,
installing one project's dependencies breaks another's — the single most common
way scientific Python setups fall apart.

**Why Miniforge specifically, and not Anaconda:** Miniforge is the ARM64-native,
community-maintained build that defaults to the `conda-forge` package channel,
which has much better Apple Silicon coverage. Anaconda's installer will happily
give you an Intel build that runs under Rosetta emulation — everything appears to
work, but you're running translated x86 code and **Metal GPU backends may refuse
to initialise entirely**. That failure is quiet and expensive.

Now activate conda in your shell:

```bash
conda init zsh
```

This appends a block to `~/.zshrc` (the config file for interactive shells).
**Close Terminal and open a new one** — this change only takes effect in a fresh
shell.

Your prompt now starts with `(base)`. That's conda telling you which environment
is active. `base` is the default one; you're about to make your own.

Optional but recommended:

```bash
conda config --set auto_activate_base false
```
Stops conda auto-activating `base` in every terminal. Being explicit about which
environment you're in prevents a whole category of "why isn't this installed"
confusion.

### 4.2 Create the project environment

```bash
cd ~/tissue-dynamics
conda env create -f host/environment.yml
```

This reads `host/environment.yml` and builds an environment named `tissue-host`.
Takes 5–15 minutes. Open `host/environment.yml` in a text editor and read it —
it's commented, and it's the authoritative record of what this environment
contains.

Then activate it:

```bash
conda activate tissue-host
```

Your prompt changes to `(tissue-host)`. **You must run this in every new terminal
where you want to use this environment.** Forgetting is the most common beginner
stumble: you run a script, get `ModuleNotFoundError: No module named 'torch'`,
and the cause is simply that the environment isn't active.

### 4.3 Verify

```bash
python host/verify_host.py
```

Eight checks. What each one is really testing:

- **ARM64-native** — that you aren't secretly running emulated x86. If this
  fails, everything else is compromised; uninstall conda and reinstall Miniforge.
- **PyTorch MPS available** — that PyTorch can see the GPU.
- **A real matmul on MPS** — that it can *use* it. "Available" and "works" are
  different, and the difference has cost people days.
- **Taichi Metal backend** — that Taichi initialises on the GPU rather than
  silently falling back to CPU, which it does with only a log line.
- **A Taichi kernel executes** — same reasoning as the matmul.
- **Trajectory format round-trips** — that the shared data code works here.
- **Docker daemon** — warns if Docker isn't running yet. Expected to warn at
  this point; you install it next.

> **Checkpoint 4.** Everything says `ok` except possibly the Docker check.
> If PyTorch or Taichi says `FAIL`, stop and fix it — see Part 7.

---

## Part 5 — The container (the simulation side)

### 5.1 Install Docker Desktop

```bash
brew install --cask docker-desktop
```

If Homebrew says that cask doesn't exist (the name has changed over time), try
`brew install --cask docker`, or download the **Apple Silicon** `.dmg` from
docker.com directly.

Then **open the Docker application** from Applications. It'll ask for permissions
and your password. A whale icon appears in your menu bar. Wait for it to say
"Docker Desktop is running".

**This is not optional and not obvious:** Docker is a background service. If the
app isn't running, every `docker` command fails with "Cannot connect to the
Docker daemon". When a Docker command mysteriously fails, check the menu bar
first.

**Licensing note:** Docker Desktop is free for personal use, education, and small
businesses, but requires a paid subscription at larger companies. If that's a
problem, [OrbStack](https://orbstack.dev) or [Colima](https://github.com/abiosoft/colima)
are drop-in alternatives; every command in this guide works unchanged with either.

### 5.2 Give it enough memory

Docker runs a lightweight Linux VM under the hood with a conservative default
memory limit. You have 128 GB; don't let it use 8.

Docker Desktop → **Settings** (gear icon) → **Resources**:

- **Memory:** 32 GB (deformable simulation with a fine mesh is memory-hungry)
- **CPUs:** 12 (leave headroom for macOS and your host-side work)
- **Disk image size:** 100 GB or more

Click **Apply & restart**.

Verify:

```bash
docker --version
docker info --format '{{.ServerVersion}} / {{.Architecture}}'
```

The architecture should say `aarch64`. That confirms Docker is running ARM
natively rather than emulating x86 — which would cost you roughly 5x in physics
throughput.

### 5.3 Read the Dockerfile before building it

```bash
cat docker/Dockerfile
```

Take five minutes on this. It's the most valuable file to understand, because
it's a *complete, executable description of a working environment* — the thing
that would otherwise live only in your memory and a pile of half-remembered
terminal commands.

The structure of every Dockerfile:

- `FROM ubuntu:22.04` — the starting filesystem
- `RUN <shell command>` — run something during the build; each `RUN` produces a
  cached layer
- `ENV` — set an environment variable
- `WORKDIR` — change directory
- `CMD` — what to run when a container starts

**Layer caching** is why the ordering looks odd. Docker caches each `RUN` and
reuses the cache if that instruction and everything before it are unchanged. So
slow, stable things (system packages) go first and fast, volatile things (SurRoL)
go last. Edit the last line, and only the last line rebuilds.

Three decisions in that file are worth knowing about, because they're the ones
that will bite you if you ever change them:

1. **`numpy==1.23.5`, pinned.** NumPy 1.24 removed the deprecated aliases
   `np.bool`, `np.int`, `np.float`. Gym 0.21 and SurRoL still use them. A newer
   NumPy produces `AttributeError` deep inside library code.

2. **`gym==0.21.0`, pinned.** Gym 0.26 changed its core API — `reset()` began
   returning a tuple, `step()` five values instead of four. SurRoL is written
   against the old API. Newer gym imports fine and then fails at runtime, which
   is the worst kind of incompatibility.

3. **`pip install --no-deps -e .` for SurRoL.** SurRoL's `setup.py` asks for
   `gym>=0.15.6` with no upper bound. Without `--no-deps`, pip helpfully upgrades
   you to a modern gym and undoes pin #2. Everything SurRoL needs is already
   installed by then, so skipping dependency resolution is safe.

That's the general lesson: **research code is usually pinned to the software
landscape of its publication date.** "Just install the latest version" is
reasonable-sounding advice that breaks research code constantly.

### 5.4 Build the image

```bash
cd ~/tissue-dynamics
docker compose build
```

**Expect 15–30 minutes.** Most of that is compiling PyBullet from source —
there's no prebuilt package for ARM Linux, so pip downloads C++ and builds it.
You'll see a long period with no output. That's compilation, not a hang.

What you're watching: each `RUN` from the Dockerfile executes in order, with
`CACHED` next to steps reused from a previous build.

Success ends with something like `naming to docker.io/tissue-dynamics/surrol:latest`.

If it fails, note **which step number** failed — that maps directly to a line in
the Dockerfile, which tells you what to fix. See Part 7.

### 5.5 Verify

```bash
docker compose run --rm surrol python container/verify_container.py
```

Unpacking that command, because you'll type it a hundred times:

- `docker compose run` — start a container from the service defined in
  `docker-compose.yml`
- `--rm` — delete the container when the command finishes. Containers are
  disposable; without this you accumulate hundreds of dead ones.
- `surrol` — the service name from `docker-compose.yml`
- everything after — the command to run inside

Nine checks, testing:

- **ARM64 Linux, not emulated** — `aarch64`, not `x86_64`
- **numpy < 1.24 and gym 0.21** — that the pins survived
- **PyBullet starts a physics server**
- **Rigid-body physics is correct** — drops a sphere and compares against
  `z = 1 − ½gt²`. Catches wrong units and broken integration.
- **Deformable bodies work** — loads a cloth. This is the one that matters most:
  soft-body support is the reason this project exists.
- **SurRoL imports, and the dVRK PSM loads** — exercises URDFs, asset paths, and
  kinematics together
- **`/work` is writable** — that the shared folder actually connects to your Mac.
  If this fails, everything you collect vanishes when the container exits.

> **Checkpoint 5.** All nine `ok`.

### 5.6 Look around inside

Worth doing once, so the container stops being abstract:

```bash
docker compose run --rm surrol
```

No command means you get a bash shell inside Ubuntu. Try:

```bash
uname -a          # Linux, aarch64 — you are in a different OS
ls /work          # your Mac project folder, mounted here
python -c "import surrol, os; print(os.path.dirname(surrol.__file__))"
ls /opt/SurRoL/surrol/tasks/     # the ten built-in surgical tasks
exit              # back to macOS
```

Two things to internalise:

**`/work` is the same folder as `~/tissue-dynamics`.** Not a copy — the same
bytes on the same disk. Create a file in `/work` inside the container, and it's
in Finder immediately. This is called a *bind mount*, and it's why the two
environments can collaborate.

**Everything outside `/work` is temporary.** `exit` destroys the container
(because of `--rm`). Anything you `pip install` interactively is gone. That feels
hostile at first and turns out to be the best feature: it forces every dependency
into the Dockerfile, so the environment is always reproducible. **If you install
something interactively and it helps, add it to the Dockerfile immediately.**

---

## Part 6 — The end-to-end run

Both halves work. Now use them together.

### 6.1 Collect data (container)

```bash
cd ~/tissue-dynamics
docker compose run --rm surrol python container/collect_retraction.py --episodes 5
```

A deformable sheet is grasped, lifted, retracted, held, and released — five
times, each with randomised grasp point, retraction direction, and material
stiffness. Roughly 30 seconds to a few minutes per episode.

Read `container/collect_retraction.py` afterwards. Two things it does are
deliberate and worth understanding:

**It uses a rigid block, not the dVRK arm.** Introducing soft-body physics and
a 7-DOF robot simultaneously means that when something looks wrong you can't tell
whether it's the tissue model, the inverse kinematics, or contact handling.
Isolate the uncertain part. Swapping in `surrol.robots.psm.Psm1` afterwards is a
small change and the logging code doesn't move.

**It randomises material parameters per episode.** A model trained on one
stiffness learns that stiffness, not the underlying dynamics. Varying it from the
start is much cheaper than regenerating a dataset later.

Check the output landed on your Mac:

```bash
ls -lh data/
```

Five `.npz` files. These were written by Linux and are sitting in your macOS
home folder — that's the bind mount working.

### 6.2 Train a model (host)

```bash
conda activate tissue-host
python host/train_dynamics.py --data data --epochs 40
```

Reads the trajectories, trains an MLP on the Apple GPU, reports validation error
in millimetres.

**Read the baseline numbers before the model numbers.** The script always reports
a *constant-velocity baseline*: assume every node keeps moving exactly as it was.
Soft tissue at 30 Hz is smooth, so that baseline is strong. A model that doesn't
clearly beat it has learned nothing, no matter how small the loss looks.

With five episodes it may well not beat the baseline. That's fine and expected —
the point of this step is that the pipeline runs end to end, not that the model
is good.

Two design choices in that script generalise well beyond this project:

**It predicts position deltas, not absolute positions.** Absolute targets force
the network to memorise where the tissue sits in world coordinates, and that
large constant offset dominates the loss. Deltas are small and zero-centred,
which is what neural networks handle well.

**It splits train/validation by episode, never by random timestep.** Consecutive
frames are nearly identical. A random split leaks near-duplicate frames into
validation and produces a beautiful, completely meaningless validation curve.
This mistake is extremely common in learned-dynamics work.

### 6.3 Commit

```bash
git add -A
git commit -m "Working end-to-end: PyBullet retraction -> MLP dynamics baseline"
```

`data/*.npz` and `models/*.pt` are excluded by `.gitignore` — data is regenerable
from code plus a seed, and git handles large binaries badly. Version the script
that produces the data, not the data.

> **Checkpoint 6.** You collected data in Linux and trained a model on the Apple
> GPU, in one project folder, in two environments. The setup is complete.

---

## Part 7 — Troubleshooting

### `command not found: brew` / `conda` / `docker`

The program is installed but your shell doesn't know where to find it. The
`PATH` variable lists folders the shell searches.

```bash
echo $PATH        # should contain /opt/homebrew/bin
which brew        # where the shell thinks brew is
```

Fix: close and reopen Terminal (config changes only apply to new shells). If it
persists, re-run the two `brew shellenv` lines from Part 2.2.

### `ModuleNotFoundError: No module named 'torch'`

Almost always: the conda environment isn't active. Check your prompt for
`(tissue-host)`.

```bash
conda activate tissue-host
which python      # should be inside .../envs/tissue-host/bin/python
```

If `which python` points at `/usr/bin/python3`, you're on macOS's system Python,
which has none of your packages.

### `Cannot connect to the Docker daemon`

Docker Desktop isn't running. Open it from Applications, wait for the whale icon
in the menu bar to stop animating.

### The Docker build fails

Note the step number in the error — it maps directly to a Dockerfile instruction.

- **`roboticstoolbox-python` fails.** This is the most fragile dependency, which
  is why it has its own line. Comment it out (put `#` at the start of the line),
  rebuild, and continue. Most of SurRoL works without it, and you'll get a clear
  `ImportError` if you reach a part that needs it.
- **Network timeouts.** Just re-run `docker compose build`. Cached layers are
  reused, so it resumes rather than restarting.
- **`no space left on device`.** Docker's disk image filled up.
  ```bash
  docker system prune -a     # deletes unused images/containers — frees a lot
  ```
  and raise the disk limit in Docker Desktop → Settings → Resources.

### Verification says `x86_64` instead of `arm64` / `aarch64`

You're running emulated. On the host, it means an Intel conda — uninstall it and
install Miniforge. In the container, check that `platform: linux/arm64` is
present in `docker-compose.yml` and rebuild with `docker compose build --no-cache`.

### Taichi falls back to CPU instead of Metal

```bash
python -c "import taichi as ti; ti.init(arch=ti.metal)"
```
Read the log lines. Usual causes: macOS below 12.3, or an emulated Python (see
above). Confirm with `python -c "import platform; print(platform.machine())"` —
must print `arm64`.

### PyBullet `loadSoftBody` says the method isn't supported

You forgot `p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)`. Deformables need a
different world type and you must request it explicitly. This catches everyone
once.

### The simulation runs but the tissue explodes or vibrates

Classic numerical instability, and you'll meet it repeatedly in soft-body work.
In rough order of what to try:

1. Reduce the timestep (`DT = 1/480` instead of `1/240`)
2. Reduce `springElasticStiffness` — stiff springs need small timesteps
3. Increase `springDampingStiffness`
4. Check units. Coordinates should be in metres. Modelling a 3 cm object as
   `3.0` instead of `0.03` makes it a 3-metre object with wildly wrong dynamics.

### Nuclear option

The container is disposable by design:

```bash
docker compose down
docker system prune -a
docker compose build --no-cache
```

Nothing in `~/tissue-dynamics` is touched — that's on your Mac, not in the
container. This is the payoff of keeping all state in the mounted folder.

---

## Part 8 — What you actually learned

The transferable ideas, separate from the specific commands:

**Environments are described by files, not by memory.** `environment.yml` and
`Dockerfile` are the real artifacts. If your environment only exists as a
sequence of commands you once typed, it doesn't survive a laptop change, and a
collaborator can't reproduce your results.

**Isolation prevents a whole class of problems.** Separate conda environments per
project, separate containers per stack. The cost is remembering to activate; the
benefit is that project A can't break project B.

**Version pins in research code are load-bearing.** "Install the latest version"
is wrong far more often than it's right when working with published research
code. That code was pinned to its publication-era landscape.

**Verify at boundaries, not just at the end.** Each part of this guide had a
checkpoint. When the whole pipeline fails, checkpoints tell you *where*. The
verification scripts test physical correctness (does a falling sphere match
`½gt²`?), not just "did the import succeed" — an environment can import fine and
still compute nonsense.

**Commit when things work.** Not when finished. When they work.

---

## Part 9 — Where to go next

Roughly in order:

**1. Swap the block for the real dVRK arm.** Replace the kinematic block in
`collect_retraction.py` with `surrol.robots.psm.Psm1`, driven through
`psm.move()`. Read `/opt/SurRoL/tests/test_psm.ipynb` inside the container first
— it's the clearest introduction to the control API, which mirrors the real dVRK
Python interface.

**2. Scale up data collection.** Hundreds of episodes with varied grasp points,
retraction directions, stiffnesses, and mesh resolutions. Run several containers
in parallel — you have the cores for it.

**3. Replace the MLP with a graph network.** The flattened-vector MLP throws away
the mesh structure and locks you to one resolution. MeshGraphNets (Pfaff et al.,
2020) is the standard starting point for learned deformable dynamics. The
trajectory format already stores `tissue_faces` and `tissue_tets` so the graph
is waiting for you.

**4. Move to MPM / Neo-Hookean for ground truth.** The mass-spring cloth is a
pipeline test, not a tissue model — it has no volume and therefore can't capture
the incompressibility that dominates real tissue response. This is where the host
environment's Taichi Metal backend earns its place, and it's the point at which
your data starts being physically meaningful.

**5. Then Isaac Sim,** on rented cloud GPU or a Linux workstation. Because
everything reads and writes the same trajectory format, this becomes a new data
*source* rather than a rewrite. That was the point of the whole two-environment
structure.

---

## Appendix — Which environment am I in?

The single most common source of confusion. Three ways to tell:

| | Host | Container |
|---|---|---|
| Prompt | `(tissue-host) andriclu@… %` | `root@a3f2b1c8:/work#` |
| `uname -s` | `Darwin` | `Linux` |
| `platform.machine()` | `arm64` | `aarch64` |

Rule of thumb: **anything touching PyBullet or SurRoL runs in the container;
anything touching PyTorch or Taichi runs on the host.**

Day-to-day commands are in `CHEATSHEET.md`.
