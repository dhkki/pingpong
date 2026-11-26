# Ping Pong Robot

[English](README.md) | [한국어](README_KR.md)

**2026 창의융합 AX-Challenge — 우승**

## Overview

2026 창의융합 AX-Challenge 참가를 위해 개발한 탁구 로봇이다.

두 대의 카메라를 이용하여 탁구공의 3차원 위치를 실시간으로 추적하고, 공의 이동 궤적과 예상 타점을 계산한다. 예측된 타점과 타격 시각을 바탕으로 로봇의 자세 및 스윙을 제어한다.

카메라 캘리브레이션부터 로봇 제어에 이르는 전 과정을 하나의 시스템으로 통합하였다.

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

- ChArUco Board 기반 카메라 내부 파라미터 추정
- 탁구대를 기준으로 카메라 외부 파라미터 설정
- HSV 기반 공 검출 및 Stereo Triangulation을 통한 3차원 좌표 복원
- 비동기 카메라 프레임 간 시간 차이 보정 및 Kalman Filtering

### Trajectory Prediction

- 수평 방향의 선형 회귀 및 수직 방향의 포물선 근사
- 바운스 전후의 궤적 모델링 및 반발계수 기반 예측
- 실시간 관측 데이터를 이용한 예상 타점 및 타격 시각 갱신

### Robot Kinematics & Motion Control

- 로봇의 기구학적 구조를 고려한 Forward / Inverse Kinematics
- 라켓의 도달 가능 영역을 고려한 타격 계획 수립
- 모터의 속도 프로파일 및 이동 시간을 고려한 스윙 제어
- 멀티스레딩 기반 비전 처리 및 모터 제어
- 최신 예측값을 반영하는 실시간 목표 위치 갱신

### Integrated GUI

- 카메라 영상 및 공 검출 결과 출력
- 실측 궤적, 예측 궤적 및 로봇 자세의 3차원 시각화
- 시스템 파라미터 설정, 데이터 기록 및 시뮬레이션
- 로봇 제어 및 비상 정지

## Repository Structure

| File / Directory | Description |
|---|---|
| `calibrate_intrinsics.py`, `setup_extrinsics.py` | 카메라 캘리브레이션 |
| `track_ball*.py` | 공 검출, 위치 추적 및 통합 시스템 |
| `trajectory_predictor.py` | 궤적 및 타점 예측 |
| `shot_planner.py`, `robot_kinematics.py` | 타격 계획 및 기구학 연산 |
| `motion_control.py`, `turn_state.py` | 실시간 로봇 제어 및 랠리 상태 관리 |
| `robot_backend.py`, `*_link.py` | 하드웨어 제어 인터페이스 |
| `settings/` | 시스템 설정 |
| `native/` | 리니어 모터 제어 모듈 |

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

실제 로봇 구동에 앞서 카메라 캘리브레이션 및 모터 통신 설정이 필요하다.

## Team & Contributions

| Name | Role | Contributions |
|---|---|---|
| [곽동혁](https://github.com/dhkki) | Team Leader / Project Manager | 프로젝트 총괄 및 하드웨어·소프트웨어 개발 전반 관리 |
| [이영택](https://imytle.github.io/) | Mechanical Designer / Table Tennis Player | 로봇 설계, 3D 모델링 및 탁구 선수 |
| [김건형](https://geon0407.github.io/) | Software Engineer | 컴퓨터 비전, 궤적 예측, 로봇 제어 및 시스템 통합 등 소프트웨어 개발 전반 |
