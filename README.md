# Ping Pong Robot

[English](README.md) | [한국어](README_KR.md)

**1st Place — 2026 AX-Challenge**

## Overview

A table tennis robot developed for the 2026 AX-Challenge.

The system tracks a table tennis ball in 3D using two cameras and predicts its trajectory, impact position, and arrival time. The predicted results are used to determine the robot's configuration and swing timing.

The entire pipeline, from camera calibration to robot control, is integrated into a single application.

## System Architecture

```text
Stereo Cameras
      ↓
Ball Detection & 3D Tracking
      ↓
Trajectory Prediction
      ↓
Shot Planning & Inverse Kinematics
      ↓
Real-time Motion Control
```

## Implementation

### Camera Calibration & Ball Tracking

- Camera intrinsic calibration using a ChArUco board
- Extrinsic calibration using the table as a reference
- HSV-based ball detection and 3D reconstruction through stereo triangulation
- Asynchronous frame compensation and Kalman filtering

### Trajectory Prediction

- Linear regression for horizontal motion and parabolic fitting for vertical motion
- Pre- and post-bounce trajectory modeling with restitution-based prediction
- Continuous updates of the predicted impact position and arrival time

### Robot Kinematics & Motion Control

- Forward and inverse kinematics based on the robot's mechanical structure
- Shot planning within the paddle's reachable workspace
- Swing scheduling based on actuator velocity profiles and travel times
- Multithreaded vision processing and motor control
- Real-time target updates based on the latest predictions

### Integrated GUI

- Live camera feeds and ball detection results
- 3D visualization of measured trajectories, predicted trajectories, and robot configurations
- Parameter configuration, data logging, and simulation
- Robot control and emergency stop

## Repository Structure

| File / Directory | Description |
|---|---|
| `calibrate_intrinsics.py`, `setup_extrinsics.py` | Camera calibration |
| `track_ball*.py` | Ball detection, tracking, and integrated system |
| `trajectory_predictor.py` | Trajectory and impact prediction |
| `shot_planner.py`, `robot_kinematics.py` | Shot planning and robot kinematics |
| `motion_control.py`, `turn_state.py` | Real-time motion control and rally state management |
| `robot_backend.py`, `*_link.py` | Hardware control interfaces |
| `settings/` | System configurations |
| `native/` | Native linear motor control module |

## Development Environment

- **Language:** Python, C/C++
- **Libraries:** OpenCV, NumPy, Tkinter
- **Hardware:** Stereo Cameras, Dynamixel, Ajin Linear Motor
- **OS:** Windows

## Usage

```bash
# Camera calibration
python calibrate_intrinsics.py --cam 0
python calibrate_intrinsics.py --cam 1
python setup_extrinsics.py

# Simulation
python track_ball_robot.py --sim

# Integrated system
python track_ball_robot.py
```

Camera calibration and actuator communication settings are required before operating the physical robot.

## Team & Contributions

| Name | Role | Contributions |
|---|---|---|
| [Donghyeok Kwak](https://github.com/dhkki) | Team Leader / Project Manager | Overall project management and coordination of hardware and software development |
| [Youngtaek Lee](https://imytle.github.io/) | Mechanical Designer / Table Tennis Player | Robot design, 3D modeling, and table tennis player |
| [Geonhyoung Kim](https://geon0407.github.io/) | Software Engineer | End-to-end software development, including computer vision, trajectory prediction, robot control, and system integration |
