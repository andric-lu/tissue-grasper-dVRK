#!/usr/bin/env bash
# setup_tissue_sim.sh -- the SIMULATION environment for the Classic arm (SurRoL).
#
#     bash host/setup_tissue_sim.sh          # from ~/tissue-dynamics, conda base active
#
# WHY A SECOND ENVIRONMENT. SurRoL `main` (the Classic Psm1, to-do Item 6) uses
# `np.float`/`np.int`/`np.bool`, removed in numpy 1.24, and gym 0.21, which no
# longer builds under modern setuptools. `tissue-host` has numpy 2.x for torch.
# So: tissue-sim collects data (numpy 1.23), tissue-host trains on it (numpy 2);
# the .npz schema is the only thing they share (DECISION_LOG 14.15, and the
# 25 Sep handoff's "two conda envs, coupled only through the .npz schema").
#
# Every pin below is the one that made `host/check_surrol_interface.py`'s import
# step pass on 22 Oct 2026 (Linux x86_64, Python 3.10.20): numpy 1.23.5,
# taichi 1.7.4, pybullet 3.2.6, gym 0.21.0, roboticstoolbox-python 1.4.4,
# SurRoL main a903fa6. The macOS-specific parts are the CFLAGS line (PyBullet
# does not compile on this SDK without it, DECISION_LOG 9.4) and Metal, which
# Taichi picks automatically. Each step stops the script on failure (set -e).
set -euo pipefail

ENV=tissue-sim
SURROL_DIR="${SURROL_DIR:-$HOME/SurRoL}"

# PyBullet builds from source on macOS arm64 and needs this BEFORE pip runs.
export CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"

conda create -n "$ENV" python=3.10 -y
# `conda run` executes inside the env without needing `conda activate` in a script.
RUN="conda run -n $ENV --no-capture-output"

# gym 0.21's setup.py breaks on setuptools >= 66: pin the build tools first.
$RUN pip install "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4"

# numpy pinned IN THE SAME COMMAND as everything that depends on it, so the
# resolver cannot upgrade it past 1.23 to satisfy a newer package.
$RUN pip install "numpy==1.23.5" "taichi==1.7.4" "pybullet==3.2.6" \
    scikit-image trimesh pytest "roboticstoolbox-python" matplotlib

# --no-deps: gym's declared deps would pull a newer numpy. --no-build-isolation:
# build with the pinned setuptools above rather than a fresh, too-new one.
$RUN pip install --no-deps --no-build-isolation "gym==0.21.0"
$RUN pip install --no-deps cloudpickle "importlib-metadata<5.0"

if [ ! -d "$SURROL_DIR" ]; then
    git clone --branch main https://github.com/med-air/SurRoL.git "$SURROL_DIR"
fi
git -C "$SURROL_DIR" log --oneline -1
# Editable install, no deps: SurRoL's setup.py asks for unpinned gym/numpy.
$RUN pip install --no-deps -e "$SURROL_DIR"

echo
echo "check with:  conda activate $ENV && python host/check_surrol_interface.py"
