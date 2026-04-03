# YOR Robot Repository

This repository contains the codebase for the YOR robot, supporting both high-fidelity MuJoCo simulation and physical robot control via CAN bus.

## 1. Installation & Setup

### 1.1 Prerequisites
- **OS:** Linux (Ubuntu 22.04+ recommended) for physical robot control. macOS/Windows supported for Simulation only.
- **Python:** Version 3.10 or higher.
- **Hardware (Physical only):**
  - CANable or compatible USB-to-CAN adapter.
  - AgileX Nero arm(s).
  - SparkFlex/SparkMax motor controllers.

### 1.2 Cloning the Repository
This repository uses submodules. Clone recursively:
```bash
git clone -b nero --recursive https://github.com/YOR-robot/YOR.git
cd YOR
```
If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 1.3 Environment Setup (Recommended: Conda)

For **Linux systems with physical robot support**, we recommend using conda to manage dependencies, as it provides better control over C++ library versions.

**1. Create Conda Environment:**
```bash
conda create -n yor-nero python=3.10 -y
conda activate yor-nero
```

**2. Install System Dependencies:**
```bash
conda install -y -c conda-forge pinocchio spdlog catch2 boost pybind11 gxx cxx-compiler
```

**3. Install Build Tools:**
```bash
pip install scikit-build-core cmake ruckig
```

**4. Install Hardware Drivers (Linux Only):**

Install `sparkcan_py`:
```bash
cd sparkcan_py
pip install .
cd ..
```

Install `nerolib` into the active environment:
```bash
cd nerolib
bash install.sh
cd ..
```
*(This will automatically install dependencies into your active `yor-nero` environment.)*

**5. Install Main Robot Package:**
```bash
pip install -e .
```

### 1.4 Environment Setup (Alternative: `uv`)

For **simulation-only** or **macOS/Windows** setups, `uv` provides a fast, lightweight alternative.

**1. Install `uv` (if not installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Sync Environment:**
```bash
uv sync
```
This will automatically create a virtual environment (`.venv`), install the correct Python version, and sync all dependencies from `uv.lock`.

**3. Activate Environment:**
```bash
source .venv/bin/activate
```

> [!NOTE]
> The `uv` method installs only simulation dependencies. Physical robot drivers (`sparkcan_py` and `nerolib`) must be installed separately following section 1.3 steps 4-5.

### 1.5 Verify Installation

Test that all packages import correctly:
```bash
python -c "import robot; print('Robot package installed successfully')"
```

For physical robot setup, also verify:
```bash
python -c "import nerolib; import sparkcan_py; print('Hardware drivers installed successfully')"
```

### 1.6 Navigation & Mapping Setup (Optional)

For dynamic mapping and navigation (e.g., using `robot/zed_pub_node.py` or `robot/nav/`), additional dependencies are required.

**1. Install PyTorch (Linux Conda Recommended):**
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
*(Adjust the CUDA version to match your system based on [pytorch.org](https://pytorch.org)).*

**2. Install Navigation Python Packages:**
You can install all necessary python dependencies using the `nav` optional group:
```bash
pip install -e ".[nav]"
```

**3. Install ZED SDK (for physical cameras):**
Download and install the [ZED SDK for Linux](https://www.stereolabs.com/developers/release) (matching your CUDA version), then install its Python wrapper:
```bash
wget -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/4.1/cu118/ubuntu22
chmod +x ZED_SDK_Linux.run
./ZED_SDK_Linux.run -- silent skip_tools
python -m pyzed.sl.get_python_api
```

---

## 2. Running Simulation (Works on macOS/Linux/Windows)

The simulation uses MuJoCo and does not require hardware drivers.

**1. Launch Simulation:**
```bash
python robot/yor_mujoco.py
```
This will launch a passive MuJoCo viewer.

**2. Control the Robot:**
In a separate terminal (with the same environment activated), run a teleop script:
```bash
python robot/teleop/telestick.py
```

---

## 3. Running Physical Robot (Linux Only)

**1. Automated Setup (Recommended):**

Run the following script to set up the CAN interface, launch the robot driver, and start the joystick controller in a tmux session:
```bash
./create_windows.sh
```

**2. Manual Setup:**

If you prefer to run components individually:

- **Hardware Setup:**
  Connect CAN adapter/motors and configure interfaces:
  ```bash
  ./extra/setup.sh
  ```

- **Launch Robot Driver:**
  ```bash
  python robot/yor.py
  ```

- **Control the Robot:**
  In a separate terminal:
  ```bash
  python robot/teleop/joystick.py
  ```

---

## 4. Teleoperation Options

Teleoperation scripts communicate with either Simulation or Physical Robot via `commlink` (ZeroMQ). Run these in a separate terminal while the robot/sim is running.

**Keyboard / Test Controls:**
```bash
python robot/teleop/telestick.py
```

**Joystick (Gamepad):**
```bash
python robot/teleop/joystick.py
```

**Oculus / VR Teleop:**
Requires Oculus controller connected via Bluetooth or link.
```bash
# Single arm + base
python robot/teleop/oculus_teleop.py

# Whole body (Base + Arm)
python robot/teleop/oculus_wb_teleop.py

# Bimanual
python robot/teleop/oculus_bimanual_teleop.py
```

---

## 5. Troubleshooting

**"ModuleNotFoundError: No module named 'sparkcan_py'"**
- You are likely on macOS/Windows or skipped the hardware driver installation. This is expected if you are only running simulation. Physical robot control (`robot/yor.py`) will not work without this.

**"ImportError: ... linux/can.h not found"**
- You are trying to install hardware drivers on a non-Linux OS. Use the simulation (`robot/yor_mujoco.py`) instead.

**"CMake Error: Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS)"**
- When installing `nerolib`, use `bash install.sh` from the `nerolib/` directory
- Ensure your conda environment is activated before running the install script

**"undefined reference to GLIBCXX_3.4.31"**
- Install the conda C++ compiler: `conda install -y -c conda-forge gxx cxx-compiler`
- This ensures the C++ standard library version matches the boost libraries

**"ZMQ / RPC Connection Failed"**
- Ensure `robot/yor.py` or `robot/yor_mujoco.py` is running in another terminal
- Check if port 5557 is blocked or already in use

**Import errors after installation**
- Verify your environment is activated: `conda activate yor-nero` or `source .venv/bin/activate`
- Try reinstalling in editable mode: `pip install -e .`
