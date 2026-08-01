# =============================================================================
# Dockerfile -- the recipe for the Linux machine that runs SurRoL.
#
# A Dockerfile is read top to bottom. Each instruction produces a cached layer,
# so if you edit line 60 only lines 60+ are rebuilt. That is why the slow,
# rarely-changing steps (apt packages) come first and the fast, frequently-
# changing steps (SurRoL itself) come last.
# =============================================================================

# FROM = the starting point. `ubuntu:22.04` is a minimal Ubuntu filesystem.
# 22.04 rather than 24.04 because SurRoL's dependency stack is from that era
# and 22.04 ships Python 3.10, which is the newest version the old `gym` API
# still works with.
FROM ubuntu:22.04

# ENV sets environment variables for every later command AND for the running
# container. DEBIAN_FRONTEND=noninteractive stops apt from opening interactive
# prompts (there is no human at the keyboard during a build, so a prompt would
# hang forever). PYTHONUNBUFFERED makes Python print immediately instead of
# buffering, so you see output in real time.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# -----------------------------------------------------------------------------
# System packages
# -----------------------------------------------------------------------------
# `apt-get update` refreshes the package index; `install` fetches packages.
# They MUST be in the same RUN because each RUN is a separate cached layer --
# a stale index from an old cached layer causes "package not found" errors.
#
# What each group is for:
#   python3.10, -dev, python3-pip : the interpreter, its C headers (needed to
#                                   COMPILE extensions), and the installer
#   build-essential, cmake, git   : C/C++ compiler + build tools. Required
#                                   because pybullet has NO prebuilt package for
#                                   ARM Linux -- pip downloads source and
#                                   compiles it here.
#   libgl1-mesa-dev, libglu1-mesa-dev, libx11-dev, libxext-dev
#                                 : graphics headers pybullet links against at
#                                   build time, even for headless use
#   ffmpeg                        : imageio-ffmpeg writes videos with it
#   ca-certificates               : lets pip/git verify HTTPS certificates
#
# `--no-install-recommends` skips optional extras; `rm -rf /var/lib/apt/lists/*`
# deletes the package index afterwards. Both just keep the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential cmake git \
        libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxext-dev \
        ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make `python` and `pip` mean python3.10 / pip3. Ubuntu deliberately does not
# create a bare `python`, and half the tutorials you will read assume it exists.
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
 && ln -sf /usr/bin/pip3       /usr/local/bin/pip

# -----------------------------------------------------------------------------
# Python packaging tools -- PINNED, and this pin is load-bearing
# -----------------------------------------------------------------------------
# gym 0.21.0 was published before a packaging-standards change. Modern
# setuptools rejects its metadata with a cryptic error. Downgrading setuptools
# and wheel first is the standard, widely-used fix.
#
# You need gym 0.21 specifically: gym 0.26+ changed the environment API
# (`reset()` returns a tuple, `step()` returns five values instead of four),
# and SurRoL is written against the old API. Installing "the latest gym" here
# produces code that imports fine and then fails at runtime.
RUN pip install --upgrade "pip==23.3.2" \
 && pip install "setuptools==65.5.0" "wheel==0.38.4"

# -----------------------------------------------------------------------------
# Scientific Python stack
# -----------------------------------------------------------------------------
# numpy is pinned to 1.23.5 because 1.24 removed the deprecated aliases
# `np.bool`, `np.int`, `np.float`, which this generation of gym/SurRoL code
# still uses. It is also installed BEFORE pybullet, because pybullet's build
# script looks for numpy headers and silently disables its numpy interface if
# they are missing -- which would make every `getMeshData` call return slow
# Python lists instead of arrays.
RUN pip install \
        "numpy==1.23.5" \
        "scipy" "pandas" \
        "imageio" "imageio-ffmpeg" \
        "opencv-python-headless"

# pybullet compiles from source here; expect 3-8 minutes with no output.
# That is normal, not a hang.
RUN pip install "pybullet==3.2.6"

# gym, with dependency resolution disabled so it cannot drag numpy back up.
RUN pip install --no-deps "gym==0.21.0" "cloudpickle" "importlib-metadata<5.0"

# roboticstoolbox-python is imported by parts of SurRoL's robot code. It is the
# most fragile dependency in this file, so it gets its own layer: if the build
# fails HERE, comment this line out and rebuild. Most of SurRoL still works, and
# you will get a clear ImportError if you touch a part that needs it.
RUN pip install "roboticstoolbox-python" && pip install "numpy==1.23.5"

# -----------------------------------------------------------------------------
# SurRoL
# -----------------------------------------------------------------------------
# --depth 1 clones only the latest commit instead of the full history: faster,
# smaller, and you do not need SurRoL's history.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/med-air/SurRoL.git

WORKDIR /opt/SurRoL
# `-e` = editable install: Python is pointed at /opt/SurRoL rather than copying
# files, so you can edit SurRoL's source and see the change immediately.
# `--no-deps` is essential: SurRoL's setup.py asks for "gym>=0.15.6" with no
# upper bound, so without this flag pip would helpfully "upgrade" you to a
# modern gym and undo the pin above. Everything it needs is already installed.
RUN pip install --no-deps -e .

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
# /work is where your project folder gets mounted from the Mac. Putting
# /work/src on PYTHONPATH means `import trajectory_io` works from any directory
# inside the container.
ENV PYTHONPATH=/work/src

WORKDIR /work

# What runs when the container starts with no other command: an interactive
# shell. This makes the container behave like "a Linux terminal I can type in".
CMD ["/bin/bash"]
