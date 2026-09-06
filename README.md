# ArUco Robot Navigation

A real-time computer-vision pipeline that detects ArUco markers, estimates
their 3D pose, predicts motion with a Kalman filter, checks for collisions
against defined zones, and drives a simple state-machine navigator toward a
target marker — all visualised live with an on-screen HUD and a bird's-eye
(top-down) view.

Built with **OpenCV 4.7+** (new `cv::aruco::ArucoDetector` API).

> **Scope:** this project produces navigation *commands*
> (`STOP`/`MOVE_FORWARD`/`MOVE_BACKWARD`/`TURN_LEFT`/`TURN_RIGHT`/`HOLD`) from
> a vision pipeline — it does not drive any physical motors or hardware. To
> control a real robot, wire the `NavOutput::cmd` value from
> `RobotNavigator::tick()` to your own motor-control code (serial link to a
> microcontroller, ROS topic, etc.).

## Features

- **Marker detection** — `cv::aruco::ArucoDetector` (DICT_4X4_100) with
  sub-pixel corner refinement.
- **Pose estimation** — `cv::solvePnP` (`SOLVEPNP_IPPE_SQUARE`), a drop-in,
  version-proof replacement for the removed `estimatePoseSingleMarkers()`.
- **Trajectory prediction** — a per-marker 6-state (position + velocity)
  Kalman filter that smooths noisy pose estimates and forecasts future
  positions.
- **Collision detection** — configurable 3D danger/warning zones with
  time-to-impact estimation and a live top-down radar view.
- **Robot navigator** — a finite-state machine (`SEARCHING → APPROACHING →
  ALIGNING → HOLDING`, plus `EMERGENCY`) that outputs simple drive commands
  toward a target marker ID.
- **Performance logging** — per-frame timings written to
  `evaluation/metrics.csv`, with a summary printed on exit.

## Project layout

```
.
├── src/
│   └── main.cpp        # full pipeline (detector, pose, predictor,
│                        # collision, navigator, logger, main)
├── .github/
│   └── workflows/
│       └── build.yml    # CI: builds OpenCV 4.9 from source, then the project
├── CMakeLists.txt       # CMake build
├── Makefile             # simple g++/pkg-config build
├── .gitignore
└── LICENSE
```

> The `markers/` (generated marker PNGs) and `evaluation/` (metrics CSV)
> folders are created at runtime and are git-ignored.

## Requirements

- A C++17 compiler (GCC, Clang, or MSVC via MSYS2/MinGW)
- OpenCV **4.7+** built with the `opencv_contrib` `aruco` module
- A webcam, or a video file to run against

### Installing OpenCV

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev cmake build-essential
```
> ⚠️ On many Ubuntu releases (e.g. 22.04/24.04) this installs OpenCV 4.5–4.6,
> which **predates** `cv::aruco::ArucoDetector` and won't compile this
> project. Check with `pkg-config --modversion opencv4`; if it's below 4.7,
> build OpenCV from source instead (see the CI workflow in
> `.github/workflows/build.yml` for a working from-source recipe), or install
> a newer version via [conda-forge](https://anaconda.org/conda-forge/opencv)
> or [vcpkg](https://vcpkg.io/).

**Windows (MSYS2 / MinGW)**
```bash
pacman -S mingw-w64-x86_64-opencv mingw-w64-x86_64-cmake mingw-w64-x86_64-toolchain
```

**macOS (Homebrew)**
```bash
brew install opencv cmake
```

## Build

### Option A — CMake (recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/aruco_navigation --help
```

### Option B — Makefile / g++ directly

```bash
make
# or manually:
g++ -std=c++17 src/main.cpp -o aruco_navigation $(pkg-config --cflags --libs opencv4)
```

## Usage

**1. Generate printable markers**
```bash
./aruco_navigation --generate-markers
# writes markers/marker_0.png ... marker_9.png
```
Print `marker_0.png` (or any marker) on paper.

**2. Run**
```bash
./aruco_navigation                     # webcam
./aruco_navigation --video myvideo.mp4 # video file
./aruco_navigation --target 2          # navigate toward marker ID 2
./aruco_navigation --marker-len 0.08   # marker is 8 cm wide
./aruco_navigation --calib calib.yaml  # use a real camera calibration
./aruco_navigation --no-topdown        # disable the bird's-eye window
```

**Full option list**
```
--video <file>       video file (default: webcam)
--calib <yaml>       camera calibration file
--marker-len <m>     marker side length in metres, 0 < m <= 5.0 (default 0.05)
--target <id>        marker ID to navigate to, 0-99 (default 0)
--pred <n>           number of future steps to predict, 1-100 (default 10)
--no-topdown         disable bird's-eye window
--generate-markers   save marker PNGs and exit
--help               show usage
```

**Keys while running**

| Key       | Action                              |
|-----------|--------------------------------------|
| `q` / Esc | Quit                                  |
| `r`       | Reset navigator + Kalman filters      |
| `s`       | Save the current frame as a PNG       |

## Camera calibration

Without `--calib`, the app falls back to an approximate default intrinsic
matrix for a generic 640×480 webcam — good enough for demos, but not for
accurate metric distance. For real measurements, calibrate your camera with
OpenCV's [`calibrateCamera`](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
workflow (a checkerboard pattern works well) and save the result as a YAML
file with `camera_matrix`, `distortion_coefficients`, `image_width`, and
`image_height` keys — see `CameraParams::fromFile` / `::save` in `src/main.cpp`.

## Performance metrics

Every run **appends** per-frame detection/pose/prediction/total timings to
`evaluation/metrics.csv` (the header row is written once, only if the file
doesn't already have data), and prints an average-FPS summary on exit —
handy for comparing performance across machines or parameter tweaks. Each
row is tagged with a `run_id` (the run's start time as a Unix timestamp) so
rows from different runs remain distinguishable after appending, since the
per-run `frame` counter restarts from 0 each time.

## License

Released under the [MIT License](LICENSE).
