#!/bin/bash
set -euo pipefail

# Detect active environment
CURRENT_ENV=${CONDA_DEFAULT_ENV:-"base"}
TARGET_ENV="yor-nero"

# If we are in an active non-base environment, use it
if [ "$CURRENT_ENV" != "base" ]; then
    echo "Active environment '$CURRENT_ENV' detected. Installing into '$CURRENT_ENV'..."
    TARGET_ENV=$CURRENT_ENV
else
    echo "No active environment detected (base). Defaulting to '$TARGET_ENV'..."
fi

# Find mamba or conda
MAMBA_EXE=$(which mamba || which micromamba || true)
CONDA_EXE=$(which conda || true)

if [ -n "$MAMBA_EXE" ]; then
    SOLVER_EXE=$MAMBA_EXE
    echo "Using mamba/micromamba at $SOLVER_EXE"
elif [ -n "$CONDA_EXE" ]; then
    SOLVER_EXE=$CONDA_EXE
    echo "Using conda at $SOLVER_EXE"
    # Try to enable libmamba solver for faster solving on constrained hardware
    if $CONDA_EXE config --show-sources | grep -q "solver: libmamba"; then
        echo "libmamba solver already enabled."
    else
        echo "Attempting to enable libmamba solver..."
        $CONDA_EXE install -n base conda-libmamba-solver -y || echo "Could not install libmamba solver, falling back to default."
        $CONDA_EXE config --set solver libmamba || echo "Could not set libmamba solver."
    fi
else
    echo "Neither mamba nor conda found. Please ensure they are in your PATH."
    exit 1
fi

# Install/Update dependencies into target environment
echo "Updating environment '$TARGET_ENV'..."
if $SOLVER_EXE info --envs | grep -q "^$TARGET_ENV "; then
    $SOLVER_EXE env update -n "$TARGET_ENV" -f environment.yml --prune
else
    $SOLVER_EXE env create -f environment.yml -n "$TARGET_ENV"
fi

# Get environment prefix directly (no conda run)
ENV_PREFIX=$($SOLVER_EXE info --envs | grep -E "^$TARGET_ENV " | awk '{print $NF}')
if [ -z "$ENV_PREFIX" ]; then
    # Fallback: look for active env path
    ENV_PREFIX=$($SOLVER_EXE info --envs | grep "\*" | awk '{print $NF}')
fi
echo "ENV_PREFIX=$ENV_PREFIX"

# Locate compilers directly in the environment prefix
C_COMPILER=$(ls "$ENV_PREFIX/bin/"*-gcc 2>/dev/null | head -n1 || echo "")
CXX_COMPILER=$(ls "$ENV_PREFIX/bin/"*-g++ 2>/dev/null | head -n1 || echo "")

if [ -z "$C_COMPILER" ] || [ -z "$CXX_COMPILER" ]; then
    echo "ERROR: Could not find gcc/g++ in $ENV_PREFIX/bin/"
    echo "       Make sure gxx_linux-aarch64 is installed in the environment."
    exit 1
fi

echo "Using C compiler:   $C_COMPILER"
echo "Using C++ compiler: $CXX_COMPILER"

# All binaries in the environment prefix
PIP="$ENV_PREFIX/bin/pip"
CMAKE="$ENV_PREFIX/bin/cmake"

# Install ruckig
echo "Installing ruckig..."
if [ ! -d "ruckig" ]; then
    echo "ruckig directory not found!"
    exit 1
fi

cd ruckig
mkdir -p build && cd build
# Build using environment compilers and install into environment prefix
unset CC CXX  # clear any inherited compiler vars
"$CMAKE" -DCMAKE_INSTALL_PREFIX="$ENV_PREFIX" \
         -DCMAKE_C_COMPILER="$C_COMPILER" \
         -DCMAKE_CXX_COMPILER="$CXX_COMPILER" ..
make -j"$(nproc)"
# Use cmake --install to explicitly enforce the prefix (avoids /usr/local fallback)
"$CMAKE" --install . --prefix "$ENV_PREFIX"
cd ../..

# Install nerolib Python package
echo "Installing nerolib..."
NEROLIB_CONDA_ENV="$ENV_PREFIX" "$PIP" install -e .

echo "Installation complete!"
