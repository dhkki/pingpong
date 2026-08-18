#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_predictor.py
=======================
track_ball.py 용 **궤적 예측 창** (2-포물선 근사 / 단일 바운스 가정).

ball_tracker 계열 코드의 ParabolaFitView 알고리즘을 그대로 옮겨 오되,
track_ball.py 의 좌표계·단위(미터)에 맞게만 바꾼 것이다.

  [track_ball 좌표계]
    X = 탁구대 폭축(m). 0 = 중앙, -hw ~ +hw
    Y = 탁구대 길이축(m). 0 ~ table_length
    Z = 상판 기준 높이(m). 0 = 상판, 위가 +

  따라서 원본의 "X(길이축)" 은 여기서 Y,
           원본의 "Y(폭축)"   은 여기서 X 가 된다.

  진행축(along) = Y,  횡축(lateral) = X,  수직 = Z

알고리즘 요약
-------------
  구간 A (바운스 전) : 실측 샘플 최소자승 포물선 피팅
  바운스 지점        : 관측되면 실측 최저점, 아직이면 A 의 Z=0 해로 예측
  구간 B (바운스 후) : 관측되면 실측 피팅, 아직이면 반발/마찰 계수로 생성
  수평 성분(X, Y)    : 바운스와 무관하게 '직선 하나'로만 근사
                       (트랙 시작에서 line_window_m 이내 샘플로 1회 회귀)

이 파일은 track_ball.py 를 import 하지 않는다. 완전히 독립이며,
(t, [x, y, z]) 샘플만 넣어 주면 동작한다.

    pred = PredictorWindow(master_tk, geo_dict)
    pred.push(t_sec, np.array([x, y, z]))     # 실측 1점
    pred.render()                             # 주기적으로 호출

geo_dict 는 track_ball 의 settings/table_geometry.json 형식:
    table_length, table_width, net_height, net_overhang, roi{x_min..z_max}
"""

import csv
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


GRAVITY = 9.81                      # m/s^2 (양수 크기)

# ============================================================================
# 기본 설정 - 여기만 만지면 된다
# ============================================================================

DEFAULT_FIT = {
    # ---- 진행 방향 ----
    "direction": -1,            # -1 : Y 가 작아지는 방향으로 오는 공만 근사
                                # +1 : Y 가 커지는 방향 / 0 : 방향 무시
    "target_y": 0.0,            # 이 Y 라인 통과 시각/위치를 근사식으로 계산 (m)
    "dir_min_travel": 0.06,     # 방향 판정 최소 이동(m)
    #  역방향 누적 이동이 이만큼 넘으면 새 트랙(m).
    #  로봇이 친 공은 방향이 반대이므로 이 문턱을 넘으면 곧바로 정리된다.
    #  너무 작으면 검출 잡음 한두 샘플에도 트랙이 끊긴다.  [예측] 탭 슬라이더.
    "dir_rev_m": 0.05,

    # ---- 트랙 관리 ----
    "min_points": 4,            # 포물선 하나당 최소 점 개수
    "buffer": 400,              # 트랙당 최대 샘플 수
    "gap_sec": 0.15,            # 이 시간 이상 끊기면 새 트랙(s)
    "auto_clear_sec": 1.5,      # 이 시간 이상 갱신 없으면 자동 초기화(0=비활성)

    # ---- 수평(직선 1개) 근사 ----
    "line_only": True,          # X, Y 를 바운스와 무관한 직선 하나로
    "line_window_m": 0.80,      # 트랙 시작에서 이 거리 안의 샘플만 회귀(m)
    "line_min_points": 3,
    "lat_gain": 1.00,           # 횡(X) 기울기 보정 배수. 실측으로 조정
    "lat_gain_step": 0.02,
    "lat_gain_min": 0.20,
    "lat_gain_max": 2.00,

    # ---- 바운스(극소점) 판정 ----
    "bounce_z": 0.20,           # 최저점 높이 상한(m)
    "bounce_z_slack": 0.08,     # 트랙 최저점 + 이 값 이내면 높이 조건 통과(m)
    "bounce_win": 3,            # 극소 판정에 볼 앞뒤 샘플 수
    "bounce_min_fall": 0.012,   # 최저점 직전 하강량(m)
    "bounce_min_rise": 0.012,   # 최저점 직후 상승량(m)
    "bounce_min_before": 3,
    "bounce_min_after": 2,
    "bounce_debug": True,       # 미검출 시 '가장 근접한 후보' 표시

    # ---- 예측 ----
    "gravity_fixed": True,      # False 면 g 까지 피팅 (캘리브레이션 점검용)
    #  바운스 수직 반발계수.  아직 관측되지 않은 바운스 뒤의 궤적을 만들 때만
    #  쓰인다(관측된 바운스는 실측으로 피팅한다).  1.0 = 튀어오른 뒤에도
    #  같은 속도.  실측으로 맞출 값이라 UI [예측] 탭에 슬라이더로 빼 두었다.
    "restitution": 1.00,
    "friction": 0.85,           # 바운스 수평 감쇠 (line_only 면 무시됨)
    "extrapolate_t": 0.35,      # 실측 포물선을 앞으로 더 연장(s)
    "pred_max_t": 1.00,         # 예측 포물선 최대 길이(s)

    # ---- 콘솔 출력 ----
    "print_on_end": True,       # 궤적이 끝날 때마다 좌표를 한 번 출력
    "print_min_points": 3,      # 이보다 짧은 조각은 출력하지 않는다

    # 콘솔에 찍을 '통과 평면'. 근사식이 이 평면을 지나는 순간의 좌표를 낸다.
    #   "along"   : 진행축(world Y) = end_report_value 인 면   <- 기본
    #   "lateral" : 횡축(world X)   = end_report_value 인 면
    "end_report_axis": "along",
    "end_report_value": 0.0,    # (m)

    # 마지막 실측점에서 통과 평면까지 '외삽'한 거리/시간의 한계.
    #   이 값을 넘으면 예측을 신뢰할 수 없다고 보고 [!] 경고를 붙인다.
    #   end_report_drop_over = True 면 아예 좌표를 출력하지 않는다.
    "end_report_warn_m": 0.60,      # 외삽 거리 경고 기준(m)
    "end_report_warn_t": 0.25,      # 외삽 시간 경고 기준(s)
    "end_report_drop_over": False,  # 경고 기준 초과 시 출력 생략
}

# ---- 그리기 ----
PX_PER_M = 210.0                # m -> px
MARGIN = 46
Z_MAX_M = 1.00
Z_MIN_M = -0.25
RENDER_MIN_DT = 0.04            # 창 갱신 최소 간격(s) ~25 fps

C_A = (0, 255, 180)             # 구간 A (바운스 전)
C_B = (0, 200, 255)             # 구간 B (바운스 후, 실측)
C_PRED = (255, 0, 255)          # 아직 관측 안 된 예측 구간
C_RAW = (150, 150, 150)         # 실측점
C_TARGET = (60, 220, 60)
BG_IMG = (32, 26, 21)

BG = "#1e1e22"
FG = "#e8e8ec"
ACC = "#4ea3ff"
DIM = "#8b8b95"
PANEL = "#26262c"
FONT = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 9, "bold")
FONT_M = ("Consolas", 9)


# ============================================================================
# 예측 엔진 (Tk 와 무관한 순수 계산부)
# ============================================================================

class ParabolaPredictor:
    """(t, x, y, z) 샘플을 받아 2-포물선 근사/예측을 유지한다.

    buf 원소 = (t, x_lat, y_along, z)
    """

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_FIT)
        if cfg:
            self.cfg.update(cfg)
        self.direction = int(self.cfg["direction"])
        self.target_y = float(self.cfg["target_y"])
        self.gain = float(self.cfg["lat_gain"])

        self.buf = []
        self.res = None
        self.arrival = None         # {"X","Y","Z","v",...} 또는 None
        self.recording = True
        self.total_points = 0
        self.rev_accum = 0.0
        self.wrong_dir_t = 0.0
        self.wrong_dir_dy = 0.0
        self.last_push_t = time.perf_counter()
        self.auto_cleared = True
        self.last_arrival = None    # 트랙 중 마지막으로 성립한 도착 예측
        self.last_report = None     # 콘솔 출력용: 통과 평면에서의 좌표/속도
        self.track_no = 0           # 콘솔 출력용 트랙 번호

    # ------------------------------------------------------------ 입력
    def push(self, t, X):
        """실측 1점. X = (x, y, z) (m), t = perf_counter 기준 시각(s)"""
        if not self.recording:
            return
        x, y, z = (float(X[0]), float(X[1]), float(X[2]))
        self.last_push_t = time.perf_counter()
        self.auto_cleared = False

        if self.buf:
            t_prev, _, y_prev, _ = self.buf[-1]
            dy = y - y_prev
            if t - t_prev > self.cfg["gap_sec"]:
                self.end_track("샘플 끊김")
                self.buf = []
                self.rev_accum = 0.0
            elif self.direction != 0:
                # 정방향이면 누적이 깎이고 역방향이면 쌓인다.
                # 잡음 한두 샘플로는 트랙이 끊기지 않는다.
                self.rev_accum = max(0.0, self.rev_accum - dy * self.direction)
                if self.rev_accum > self.cfg["dir_rev_m"]:
                    self.end_track("역방향 진입")
                    self.buf = []
                    self.rev_accum = 0.0
                    self.wrong_dir_t = t
                    self.wrong_dir_dy = dy

        self.buf.append((float(t), x, y, z))
        if len(self.buf) > int(self.cfg["buffer"]):
            self.buf.pop(0)
        self.total_points += 1
        self.res = self._analyze()
        rep = self._cross_report()
        if rep is not None:
            self.last_report = rep

    def tick(self):
        """샘플이 없어도 주기적으로 불러 자동 초기화를 처리한다."""
        ac = float(self.cfg["auto_clear_sec"])
        if ac > 0 and not self.auto_cleared \
                and time.perf_counter() - self.last_push_t > ac:
            self.end_track("공 미검출")
            self.clear(quiet=True)

    def _cross_report(self):
        """근사식이 '통과 평면'(기본: 진행축 Y = 0)을 지나는 순간의
        좌표와 속도를 구한다. 아직 못 지나거나 근사가 없으면 None.

        v   = 3D 속도 크기
        v_h = Z 성분을 뺀 '나아가는' 수평 속도  sqrt(vX^2 + vY^2)
        """
        r = self.res
        if r is None or not r.get("ok"):
            return None
        axis = str(self.cfg.get("end_report_axis", "along"))
        val = float(self.cfg.get("end_report_value", 0.0))
        fA, fB = r["fitA"], r["fitB"]
        tb = r["bounce"][0] if r["bounce"] else None
        t_now = r["t_now"]
        for f, tag in ((fA, "A"), (fB, "B")):
            if f is None:
                continue
            lo = t_now if tag == "A" else max(t_now, f["t_off"])
            if axis == "lateral":
                if abs(f["vx"]) < 1e-6:
                    continue
                tc = f["t_off"] + (val - f["x0"]) / f["vx"]
                if tc < lo:
                    continue
            else:
                tc = self._cross_y(f, val, lo)
                if tc is None:
                    continue
            if tb is not None and tag == "A" and tc > tb:
                continue                       # 바운스 이후는 B 담당
            x, y, z = self._eval(f, tc)
            if z < -0.05:
                continue
            vx, vy, vz, v = self._curve_velocity(f, tc)
            # 마지막 실측점에서 통과 평면까지 얼마나 '외삽'했는가
            lx, ly, lz = self.buf[-1][1], self.buf[-1][2], self.buf[-1][3]
            gap_m = abs(ly - y) if axis != "lateral" else abs(lx - x)
            t0 = r["t0"]
            return {"axis": axis, "value": val, "seg": tag,
                    "eta": tc - t_now, "pred": bool(f["pred"]),
                    # 절대 시각(perf_counter). 모션 계층은 이걸 쓴다.
                    "t_sample": t0 + t_now, "t_impact": t0 + tc,
                    "track_no": self.track_no + 1,
                    "X": x, "Y": y, "Z": z,
                    "vX": vx, "vY": vy, "vZ": vz,
                    "v": v, "v_h": math.hypot(vx, vy),
                    "last_X": lx, "last_Y": ly, "last_Z": lz,
                    "extrap_m": gap_m, "extrap_t": max(tc - t_now, 0.0)}
        return None

    def end_track(self, reason=""):
        """한 궤적이 끝났을 때 X, Y, Z 를 콘솔에 한 번만 출력한다.

        - 마지막 실측 좌표 (cm)
        - 트랙 중 마지막으로 성립한 타겟 라인 도착 예측이 있으면 함께
        호출 시점: 샘플 끊김 / 역방향 진입으로 새 트랙 시작 / 자동 초기화
        """
        buf, self.buf = self.buf, []
        if not self.cfg.get("print_on_end", True):
            self.last_arrival = None
            return
        if len(buf) < int(self.cfg.get("print_min_points", 3)):
            self.last_arrival = None
            return
        self.track_no += 1
        t_first, t_last = buf[0][0], buf[-1][0]
        _, x, y, z = buf[-1]
        print("[궤적 #%d 종료] (%d점, %.2fs, %s)"
              % (self.track_no, len(buf), t_last - t_first, reason or "-"))
        rep = self.last_report
        if rep is not None:
            axname = "X" if rep["axis"] == "lateral" else "Y"
            far = (rep["extrap_m"] > float(self.cfg["end_report_warn_m"])
                   or rep["extrap_t"] > float(self.cfg["end_report_warn_t"]))
            if far and bool(self.cfg.get("end_report_drop_over", False)):
                print("   [!] 외삽 과다 (%s=%+.1fcm 에서 %.1fcm / %.0fms) "
                      "- 통과 좌표 생략"
                      % (axname,
                         (rep["last_X"] if rep["axis"] == "lateral"
                          else rep["last_Y"]) * 100.0,
                         rep["extrap_m"] * 100.0, rep["extrap_t"] * 1000.0))
            else:
                print("   %s%s=%.1fcm 통과  X=%+7.1f  Y=%+7.1f  Z=%+7.1f cm   "
                      "v=%6.1f cm/s   수평(XY) v=%6.1f cm/s"
                      % ("[!] " if far else "", axname, rep["value"] * 100.0,
                         rep["X"] * 100.0, rep["Y"] * 100.0, rep["Z"] * 100.0,
                         rep["v"] * 100.0, rep["v_h"] * 100.0))
                print("        (마지막 실측 %s=%+.1fcm 에서 %.1fcm / %.0fms "
                      "외삽%s)"
                      % (axname,
                         (rep["last_X"] if rep["axis"] == "lateral"
                          else rep["last_Y"]) * 100.0,
                         rep["extrap_m"] * 100.0, rep["extrap_t"] * 1000.0,
                         ", 바운스 예측 구간" if rep["pred"] else ""))
        else:
            print("   통과 예측 없음 - 마지막 실측  "
                  "X=%+7.1f  Y=%+7.1f  Z=%+7.1f cm"
                  % (x * 100.0, y * 100.0, z * 100.0))
        self.last_arrival = None
        self.last_report = None

    def clear(self, quiet=False):
        self.buf = []
        self.res = None
        self.arrival = None
        self.last_arrival = None
        self.last_report = None
        self.total_points = 0
        self.rev_accum = 0.0
        self.auto_cleared = True

    def toggle_pause(self):
        self.recording = not self.recording

    def adjust_gain(self, d):
        self.gain = max(self.cfg["lat_gain_min"],
                        min(self.cfg["lat_gain_max"], self.gain + d))
        if self.buf:
            self.res = self._analyze()

    def set_target_y(self, v):
        self.target_y = float(v)
        if self.buf:
            self.res = self._analyze()

    def set_dir_rev(self, v):
        """역방향 누적 문턱을 실행 중에 바꾼다(m).

        작을수록 로봇이 친 뒤 트랙이 빨리 정리되지만, 검출 잡음 한두 샘플에
        트랙이 끊길 위험이 커진다.
        """
        self.cfg["dir_rev_m"] = max(0.0, float(v))

    def set_restitution(self, v):
        """바운스 반발계수를 실행 중에 바꾼다."""
        self.cfg["restitution"] = max(0.1, min(1.5, float(v)))
        if self.buf:
            self.res = self._analyze()

    def set_friction(self, v):
        self.cfg["friction"] = max(0.1, min(1.5, float(v)))
        if self.buf:
            self.res = self._analyze()

    def toggle_direction(self):
        self.direction = -1 if self.direction >= 0 else 1
        self.buf = []
        self.res = None
        self.arrival = None
        self.last_arrival = None
        self.last_report = None

    # ------------------------------------------------------ 최소자승 피팅
    def _fit(self, samples, t0, t_off=0.0):
        """samples -> 계수 dict. 계수는 (t - t0 - t_off) 기준."""
        if len(samples) < int(self.cfg["min_points"]):
            return None
        T = np.array([p[0] - t0 - t_off for p in samples], float)
        Lx = np.array([p[1] for p in samples], float)      # 횡 (world X)
        Ay = np.array([p[2] for p in samples], float)      # 진행 (world Y)
        Z = np.array([p[3] for p in samples], float)
        A1 = np.column_stack([np.ones_like(T), T])

        y0, vy = np.linalg.lstsq(A1, Ay, rcond=None)[0]
        x0, vx = np.linalg.lstsq(A1, Lx, rcond=None)[0]

        if self.cfg["gravity_fixed"]:
            z0, vz = np.linalg.lstsq(A1, Z + 0.5 * GRAVITY * T ** 2,
                                     rcond=None)[0]
            g = GRAVITY
        else:
            A2 = np.column_stack([np.ones_like(T), T, T ** 2])
            c0, c1, c2 = np.linalg.lstsq(A2, Z, rcond=None)[0]
            z0, vz, g = c0, c1, -2.0 * c2
            if g < 1.0:                     # 직선에 가까운 비정상 해 방어
                g = GRAVITY
                z0, vz = np.linalg.lstsq(A1, Z + 0.5 * g * T ** 2,
                                         rcond=None)[0]

        rz = Z - (z0 + vz * T - 0.5 * g * T ** 2)
        ry = Ay - (y0 + vy * T)
        return {"x0": float(x0), "vx": float(vx),
                "y0": float(y0), "vy": float(vy),
                "z0": float(z0), "vz": float(vz), "g": float(g),
                "t_off": float(t_off),
                "rms_y": float(np.sqrt(np.mean(ry ** 2))),
                "rms_z": float(np.sqrt(np.mean(rz ** 2))),
                "t_lo": float(T[0] + t_off), "t_hi": float(T[-1] + t_off),
                "n": len(samples), "pred": False}

    @staticmethod
    def _lin(T, V):
        if len(T) < 2 or (max(T) - min(T)) < 1e-6:
            return None
        c0, c1 = np.linalg.lstsq(np.column_stack([np.ones_like(T), T]),
                                 V, rcond=None)[0]
        return float(c0), float(c1)

    def _fit_line(self, t0):
        """트랙 시작에서 line_window_m 이내의 샘플만으로 X(t), Y(t) 를
        각각 직선 회귀. 횡(X) 기울기에는 gain 을 곱한다."""
        if not self.buf:
            return None
        y_start = self.buf[0][2]
        sel = [p for p in self.buf
               if abs(p[2] - y_start) <= self.cfg["line_window_m"]]
        if len(sel) < int(self.cfg["line_min_points"]):
            return None
        T = np.array([p[0] - t0 for p in sel], float)
        ry = self._lin(T, np.array([p[2] for p in sel], float))
        rx = self._lin(T, np.array([p[1] for p in sel], float))
        if ry is None and rx is None:
            return None
        out = {"y0": None, "vy": None, "x0": None,
               "vx": None, "vx_raw": None, "gain": float(self.gain),
               "n": len(sel)}
        if ry is not None:
            out["y0"], out["vy"] = ry
        if rx is not None:
            out["x0"], out["vx_raw"] = rx
            out["vx"] = out["vx_raw"] * float(self.gain)
        return out

    @staticmethod
    def _apply_line(fit, line):
        """전역 직선을 각 구간 기준(t_off)으로 옮겨 적는다.
        A/B 가 같은 직선을 공유하므로 바운스에서 수평이 꺾이지 않는다."""
        if fit is None or line is None:
            return
        if line["vy"] is not None:
            fit["vy"] = line["vy"]
            fit["y0"] = line["y0"] + line["vy"] * fit["t_off"]
        if line["vx"] is not None:
            fit["vx"] = line["vx"]
            fit["x0"] = line["x0"] + line["vx"] * fit["t_off"]

    # --------------------------------------------------------- 평가 도우미
    @staticmethod
    def _eval(fit, t):
        """t 는 트랙 시작(t0) 기준 시각 -> (x_lat, y_along, z)"""
        u = t - fit["t_off"]
        x = fit["x0"] + fit["vx"] * u
        y = fit["y0"] + fit["vy"] * u
        z = fit["z0"] + fit["vz"] * u - 0.5 * fit["g"] * u * u
        return x, y, z

    @classmethod
    def _curve_velocity(cls, fit, t, dt=1e-3):
        """화면에 그려지는 근사 곡선 자체를 수치미분한 속도."""
        x1, y1, z1 = cls._eval(fit, t - dt)
        x2, y2, z2 = cls._eval(fit, t + dt)
        vx = (x2 - x1) / (2 * dt)
        vy = (y2 - y1) / (2 * dt)
        vz = (z2 - z1) / (2 * dt)
        return vx, vy, vz, math.sqrt(vx * vx + vy * vy + vz * vz)

    @staticmethod
    def _z_zero_time(fit, t_after):
        """Z=0 하강 교차 시각 (t_after 이후). 없으면 None"""
        u0 = t_after - fit["t_off"]
        z_u = fit["z0"] + fit["vz"] * u0 - 0.5 * fit["g"] * u0 * u0
        vz_u = fit["vz"] - fit["g"] * u0
        g = fit["g"]
        if g <= 0:
            return None
        disc = vz_u * vz_u + 2.0 * g * max(z_u, 0.0)
        if disc < 0:
            return None
        du = (vz_u + math.sqrt(disc)) / g
        return None if du < 0 else t_after + du

    @staticmethod
    def _cross_y(fit, target_y, t_after):
        """근사식이 target_y 를 지나는 시각. 없으면 None"""
        if abs(fit["vy"]) < 1e-6:
            return None
        t = fit["t_off"] + (target_y - fit["y0"]) / fit["vy"]
        return t if t >= t_after else None

    def _synth_bounce(self, fitA, tb):
        """바운스 시점 tb 에서 반발/마찰 계수로 뒤 포물선을 생성(예측)."""
        xb, yb, _ = self._eval(fitA, tb)
        u = tb - fitA["t_off"]
        vz_hit = fitA["vz"] - fitA["g"] * u          # 충돌 직전 수직속도(음수)
        f = float(self.cfg["friction"])
        return {"x0": float(xb), "vx": fitA["vx"] * f,
                "y0": float(yb), "vy": fitA["vy"] * f,
                "z0": 0.0, "vz": abs(vz_hit) * float(self.cfg["restitution"]),
                "g": fitA["g"], "t_off": float(tb),
                "rms_y": float("nan"), "rms_z": float("nan"),
                "t_lo": float(tb), "t_hi": float(tb), "n": 0, "pred": True}

    # ----------------------------------------------------------- 바운스 탐색
    def _find_bounce(self, Zs, i_from, n, dbg=None):
        """i_from 이후 첫 번째 '바운스로 인정되는 극소점'. 없으면 None"""
        if n < 3:
            return None
        c = self.cfg
        z_gate = max(c["bounce_z"], min(Zs) + c["bounce_z_slack"])
        w = max(1, int(c["bounce_win"]))
        best = None
        lo = max(i_from, int(c["bounce_min_before"]))
        for i in range(lo, n - int(c["bounce_min_after"])):
            z = Zs[i]
            pre = Zs[max(0, i - w):i + 1]
            post = Zs[i:min(n, i + w + 1)]
            if len(pre) < 2 or len(post) < 2:
                continue
            fall = max(pre) - z
            rise = max(post) - z
            if dbg is not None:
                score = min(fall, rise) - max(0.0, z - z_gate)
                if best is None or score > best[0]:
                    best = (score, i, z, fall, rise, z_gate)
            if z > z_gate:
                continue
            if z > min(pre) or z > min(post):
                continue
            if fall < c["bounce_min_fall"] or rise < c["bounce_min_rise"]:
                continue
            return i
        if dbg is not None and best is not None:
            dbg.update({"i": best[1], "z": best[2], "fall": best[3],
                        "rise": best[4], "gate": best[5]})
        return None

    # ---------------------------------------------------------------- 분석
    def _analyze(self):
        n = len(self.buf)
        if n < 2:
            return None
        c = self.cfg
        t0 = self.buf[0][0]
        t_now = self.buf[-1][0] - t0
        Ys = [p[2] for p in self.buf]
        Zs = [p[3] for p in self.buf]

        k = max(1, min(3, n // 2))
        travel = float(np.mean(Ys[-k:]) - np.mean(Ys[:k]))
        dir_state = 0 if abs(travel) < c["dir_min_travel"] \
            else (1 if travel > 0 else -1)
        if self.direction != 0 and dir_state != 0 \
                and dir_state != self.direction:
            return {"ok": False, "reason": "dir", "n": n, "t0": t0,
                    "t_now": t_now, "travel": travel}
        if n < int(c["min_points"]):
            return {"ok": False, "reason": "few", "n": n, "t0": t0,
                    "t_now": t_now, "travel": travel}

        line = self._fit_line(t0) if c["line_only"] else None
        dbg = {} if c["bounce_debug"] else None
        i_min = self._find_bounce(Zs, 1, n, dbg)

        if i_min is not None:
            j = self._find_bounce(Zs, i_min + 2, n)
            segB = self.buf[i_min: (j + 1 if j is not None else n)]
            tb = self.buf[i_min][0] - t0
            xb, yb, zb = self.buf[i_min][1], self.buf[i_min][2], Zs[i_min]
            fitA = self._fit(self.buf[:i_min + 1], t0)
            fitB = self._fit(segB, t0, t_off=tb)
            if fitB is None and fitA is not None:
                fitB = self._synth_bounce(fitA, tb)
            self._apply_line(fitA, line)
            self._apply_line(fitB, line)
            if line is not None:                    # 마커도 같은 직선 위로
                if line["vy"] is not None:
                    yb = line["y0"] + line["vy"] * tb
                if line["vx"] is not None:
                    xb = line["x0"] + line["vx"] * tb
            return {"ok": True, "n": n, "t0": t0, "t_now": t_now,
                    "travel": travel, "fitA": fitA, "fitB": fitB,
                    "bounced": True, "bounce": (tb, xb, yb, zb),
                    "line": line, "dbg": dbg}

        fitA = self._fit(self.buf, t0)
        if fitA is None:
            return {"ok": False, "reason": "few", "n": n, "t0": t0,
                    "t_now": t_now, "travel": travel}
        self._apply_line(fitA, line)

        tb = self._z_zero_time(fitA, t_now)
        fitB, bounce = None, None
        if tb is not None and tb - t_now <= c["pred_max_t"]:
            fitB = self._synth_bounce(fitA, tb)
            self._apply_line(fitB, line)
            xb, yb, _ = self._eval(fitA, tb)
            bounce = (tb, xb, yb, 0.0)
        return {"ok": True, "n": n, "t0": t0, "t_now": t_now,
                "travel": travel, "fitA": fitA, "fitB": fitB,
                "bounced": False, "bounce": bounce, "line": line, "dbg": dbg}

    # --------------------------------------------------- 타겟 라인 통과 예측
    def compute_arrival(self, after_net_only=True, net_y=None):
        """근사식이 target_y 를 통과하는 지점/시각/속도. 결과는 self.arrival"""
        r = self.res
        self.arrival = None
        if r is None or not r.get("ok"):
            return None
        fA, fB = r["fitA"], r["fitB"]
        tb = r["bounce"][0] if r["bounce"] else None
        t_now = r["t_now"]
        hit = None
        for f, tag in ((fA, "A"), (fB, "B")):
            if f is None:
                continue
            lo = t_now if tag == "A" else max(t_now, f["t_off"])
            tc = self._cross_y(f, self.target_y, lo)
            if tc is None:
                continue
            if tb is not None and tag == "A" and tc > tb:
                continue                       # 바운스 이후는 B 담당
            xc, yc, zc = self._eval(f, tc)
            if zc < -0.05:
                continue
            hit = (tag, f, tc, xc, zc)
            break
        if hit is None:
            return None
        tag, f, tc, xc, zc = hit
        if after_net_only and net_y is not None and self.buf:
            y_now = self.buf[-1][2]
            crossed = (y_now < net_y) if self.direction < 0 else (y_now > net_y)
            if not crossed:
                return None
        vx, vy, vz, v = self._curve_velocity(f, tc)
        self.arrival = {"seg": tag, "t": tc, "eta": tc - t_now,
                        "X": xc, "Y": self.target_y, "Z": zc,
                        "vX": vx, "vY": vy, "vZ": vz, "v": v,
                        "pred": bool(f["pred"])}
        self.last_arrival = dict(self.arrival)
        return self.arrival


# ============================================================================
# 창 (Tk Toplevel + cv2 렌더링)
# ============================================================================

class PredictorWindow(tk.Toplevel):
    """예측 결과 전용 창. 부모 앱과는 push()/render() 로만 연결된다."""

    def __init__(self, master, geo, cfg=None, record_dir=None):
        super().__init__(master)
        self.title("궤적 예측 (2-포물선 · 단일 바운스)")
        self.configure(bg=BG)
        self.geo = geo
        self.alive = True
        self._last_render = 0.0
        self._photo = None
        self.record_dir = record_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "records")

        self.pred = ParabolaPredictor(cfg)

        L = float(geo["table_length"])
        W = float(geo["table_width"])
        self.L, self.W, self.hw = L, W, W / 2.0
        self.S = PX_PER_M
        self.hA = int((Z_MAX_M - Z_MIN_M) * self.S)
        self.hB = int(W * self.S)
        self.yA = MARGIN
        self.yB = self.yA + self.hA + 52
        self.IW = int(L * self.S) + MARGIN * 2
        self.IH = self.yB + self.hB + 150

        self.base = self._make_base()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(root, width=self.IW, height=self.IH, bg="#000",
                                highlightthickness=1,
                                highlightbackground="#444")
        self.canvas.pack()

        bar = tk.Frame(root, bg=PANEL, padx=8, pady=6)
        bar.pack(fill="x", pady=(6, 0))

        tk.Label(bar, text="횡(X) 기울기 배수", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left")
        ttk.Button(bar, text="-", width=3,
                   command=lambda: self._gain(-1)).pack(side="left", padx=1)
        self.lbl_gain = tk.Label(bar, text="%.2f" % self.pred.gain, bg=PANEL,
                                 fg=FG, font=FONT_M, width=5)
        self.lbl_gain.pack(side="left")
        ttk.Button(bar, text="+", width=3,
                   command=lambda: self._gain(+1)).pack(side="left", padx=1)

        ttk.Button(bar, text="방향 반전", width=10,
                   command=self._flip_dir).pack(side="left", padx=(10, 2))
        self.btn_pause = ttk.Button(bar, text="일시정지", width=9,
                                    command=self._pause)
        self.btn_pause.pack(side="left", padx=2)
        ttk.Button(bar, text="지우기", width=8,
                   command=self.pred.clear).pack(side="left", padx=2)
        ttk.Button(bar, text="저장", width=8,
                   command=self.save).pack(side="left", padx=2)

        bar2 = tk.Frame(root, bg=PANEL, padx=8, pady=4)
        bar2.pack(fill="x")
        tk.Label(bar2, text="타겟 Y (m)", bg=PANEL, fg=DIM, font=FONT,
                 width=11, anchor="w").pack(side="left")
        self.var_target = tk.DoubleVar(value=self.pred.target_y)
        tk.Scale(bar2, from_=0.0, to=self.L, resolution=0.05,
                 orient="horizontal", variable=self.var_target, bg=PANEL,
                 fg=FG, font=FONT_M, highlightthickness=0,
                 troughcolor="#15151a", length=260,
                 command=lambda v: self.pred.set_target_y(float(v))
                 ).pack(side="left")
        self.var_net_only = tk.BooleanVar(value=True)
        tk.Checkbutton(bar2, text="네트 통과 후에만 도착 예측",
                       variable=self.var_net_only, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG).pack(side="left", padx=(12, 0))

        self.lbl_status = tk.Label(root, bg=BG, fg=DIM, font=FONT_M,
                                   anchor="w", justify="left")
        self.lbl_status.grid_propagate(False)
        self.lbl_status.pack(fill="x", pady=(6, 0))

        self.bind("<KeyPress-bracketleft>", lambda e: self._gain(-1))
        self.bind("<KeyPress-bracketright>", lambda e: self._gain(+1))
        self.bind("<KeyPress-c>", lambda e: self.pred.clear())
        self.bind("<KeyPress-C>", lambda e: self.pred.clear())

    def _gain(self, sign):
        self.pred.adjust_gain(sign * self.pred.cfg["lat_gain_step"])
        self.lbl_gain.config(text="%.2f" % self.pred.gain)

    def _flip_dir(self):
        self.pred.toggle_direction()

    def _pause(self):
        self.pred.toggle_pause()
        self.btn_pause.config(text="재개" if not self.pred.recording
                              else "일시정지")

    def close(self):
        self.pred.end_track("창 닫힘")
        self.alive = False
        try:
            self.destroy()
        except Exception:
            pass

    # -------------------------------------------------------------- 외부 API
    def push(self, t, X):
        if not self.alive:
            return
        self.pred.push(t, X)

    def render(self, force=False):
        """주기적으로 호출. 내부에서 프레임레이트를 스스로 제한한다."""
        if not self.alive:
            return
        now = time.perf_counter()
        if not force and now - self._last_render < RENDER_MIN_DT:
            return
        self._last_render = now
        self.pred.tick()
        self.pred.compute_arrival(
            after_net_only=bool(self.var_net_only.get()),
            net_y=self.L / 2.0)
        img = self._draw()
        try:
            self._photo = _to_photo(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
            self.lbl_status.config(text=self._status_text())
        except Exception:
            pass

    # ------------------------------------------------------------- 좌표변환
    def _pxA(self, y, z):          # Y-Z 패널 (측면)
        return (int(min(max(MARGIN + y * self.S, -1e4), 1e4)),
                int(min(max(self.yA + (Z_MAX_M - z) * self.S, -1e4), 1e4)))

    def _pxB(self, y, x):          # Y-X 패널 (위)
        return (int(min(max(MARGIN + y * self.S, -1e4), 1e4)),
                int(min(max(self.yB + (x + self.hw) * self.S, -1e4), 1e4)))

    # ----------------------------------------------------------------- 배경
    def _make_base(self):
        g = self.geo
        L, W, hw = self.L, self.W, self.hw
        nh = float(g["net_height"])
        r = g["roi"]
        b = np.full((self.IH, self.IW, 3), BG_IMG, np.uint8)

        # ---- 패널 A : Y-Z (측면) ----
        cv2.rectangle(b, self._pxA(0, Z_MAX_M), self._pxA(L, Z_MIN_M),
                      (50, 41, 33), -1)
        cv2.rectangle(b, self._pxA(r["y_min"], Z_MAX_M),
                      self._pxA(r["y_max"], Z_MIN_M), (62, 51, 41), -1)
        cv2.line(b, self._pxA(0, 0), self._pxA(L, 0), (255, 255, 255), 3)
        cv2.line(b, self._pxA(L / 2, 0), self._pxA(L / 2, nh),
                 (230, 230, 230), 3)
        cv2.putText(b, "NET", (self._pxA(L / 2, nh)[0] - 14,
                               self._pxA(0, nh)[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        for z10 in range(0, int(Z_MAX_M * 10) + 1, 2):
            z = z10 / 10.0
            _dashed(b, self._pxA(0, z), self._pxA(L, z), (105, 78, 52), 1, 4, 8)
            cv2.putText(b, "%.1f" % z, (MARGIN - 40, self._pxA(0, z)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)
        cv2.putText(b, "Y - Z  (side)  [m]", (MARGIN, self.yA - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ---- 패널 B : Y-X (위) ----
        cv2.rectangle(b, self._pxB(0, -hw), self._pxB(L, hw), (96, 48, 22), -1)
        cv2.rectangle(b, self._pxB(0, -hw), self._pxB(L, hw),
                      (255, 255, 255), 2)
        cv2.line(b, self._pxB(0, 0), self._pxB(L, 0), (235, 235, 235), 1)
        cv2.line(b, self._pxB(L / 2, -hw), self._pxB(L / 2, hw),
                 (230, 230, 230), 2)
        cv2.rectangle(b, self._pxB(r["y_min"], r["x_min"]),
                      self._pxB(r["y_max"], r["x_max"]), (0, 255, 255), 1)
        cv2.putText(b, "Y - X  (top)  [m]", (MARGIN, self.yB - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ---- 공통 눈금 ----
        for y10 in range(0, int(L * 10) + 1, 5):
            y = y10 / 10.0
            _dashed(b, self._pxA(y, Z_MAX_M), self._pxA(y, 0),
                    (105, 78, 52), 1, 4, 6)
            cv2.line(b, self._pxB(y, -hw), self._pxB(y, hw), (105, 78, 52), 1)
            cv2.putText(b, "%.1f" % y, (self._pxA(y, 0)[0] - 12, self.yA - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)
        # ROI 경계 (Y 방향)
        for y in (r["y_min"], r["y_max"]):
            _dashed(b, self._pxA(y, Z_MAX_M), self._pxA(y, Z_MIN_M),
                    (0, 200, 200), 1, 6, 6)
        return b

    # ---------------------------------------------------------------- 곡선
    def _curve(self, img, fit, t_from, t_to, color, dashed=False, steps=48):
        if fit is None or t_to <= t_from:
            return
        pa, pb = [], []
        for t in np.linspace(t_from, t_to, steps):
            x, y, z = ParabolaPredictor._eval(fit, t)
            pa.append(self._pxA(y, z))
            pb.append(self._pxB(y, x))
        for pts in (pa, pb):
            for i in range(1, len(pts)):
                if dashed and i % 2 == 0:
                    continue
                cv2.line(img, pts[i - 1], pts[i], color, 2)

    # --------------------------------------------------------------- 렌더링
    def _draw(self):
        p = self.pred
        img = self.base.copy()

        # 타겟 라인
        ty = p.target_y
        cv2.line(img, self._pxA(ty, Z_MAX_M), self._pxA(ty, Z_MIN_M),
                 C_TARGET, 2)
        cv2.line(img, self._pxB(ty, -self.hw), self._pxB(ty, self.hw),
                 C_TARGET, 2)
        cv2.putText(img, "TARGET Y=%.2f" % ty,
                    (self._pxB(ty, 0)[0] - 40, self.yB + self.hB - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TARGET, 1)

        # 진행 방향 화살표
        ay = self.yA - 32
        if p.direction < 0:
            cv2.arrowedLine(img, (self.IW - 110, ay), (self.IW - 230, ay),
                            C_A, 2, tipLength=0.25)
        elif p.direction > 0:
            cv2.arrowedLine(img, (self.IW - 230, ay), (self.IW - 110, ay),
                            C_A, 2, tipLength=0.25)
        cv2.putText(img, "ball dir", (self.IW - 300, ay + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_A, 1)

        # 실측점
        for (t, x, y, z) in p.buf:
            cv2.circle(img, self._pxA(y, z), 2, C_RAW, -1)
            cv2.circle(img, self._pxB(y, x), 2, C_RAW, -1)

        y_hud = self.IH - 128
        r = p.res

        if r is None or not r.get("ok"):
            recent = (time.perf_counter() - p.wrong_dir_t) < 0.4
            if (r is not None and r.get("reason") == "dir") or recent:
                dy = r["travel"] if (r is not None and r.get("reason") == "dir")\
                    else p.wrong_dir_dy
                msg = ("ignored: wrong direction (dY=%+.3f m, need %s)"
                       % (dy, "-Y" if p.direction < 0 else "+Y"))
                col = (0, 165, 255)
            else:
                nn = 0 if r is None else r.get("n", 0)
                msg = "collecting... %d/%d points" % (nn, p.cfg["min_points"])
                col = (0, 200, 255)
            cv2.putText(img, msg, (MARGIN, y_hud), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1)
            self._footer(img)
            return img

        fA, fB = r["fitA"], r["fitB"]
        tb = r["bounce"][0] if r["bounce"] else None
        t_now = r["t_now"]

        # 구간 A
        if fA is not None:
            self._curve(img, fA, fA["t_lo"], min(fA["t_hi"], t_now), C_A)
            a_end = tb if tb is not None else fA["t_hi"] + p.cfg["extrapolate_t"]
            if a_end > fA["t_hi"]:
                self._curve(img, fA, fA["t_hi"], a_end, C_PRED, dashed=True)

        # 바운스 지점
        if r["bounce"] is not None:
            _, xb, yb, zb = r["bounce"]
            bcol = C_B if r["bounced"] else C_PRED
            cv2.circle(img, self._pxA(yb, zb), 7, bcol, 2)
            cv2.drawMarker(img, self._pxA(yb, zb), bcol,
                           cv2.MARKER_TILTED_CROSS, 12, 1)
            cv2.circle(img, self._pxB(yb, xb), 7, bcol, 2)

        # 구간 B
        if fB is not None:
            if fB["pred"]:
                t_end = fB["t_off"] + min(2.0 * fB["vz"] / max(fB["g"], 1e-6),
                                          p.cfg["pred_max_t"])
                self._curve(img, fB, fB["t_off"], t_end, C_PRED, dashed=True)
            else:
                self._curve(img, fB, fB["t_lo"], min(fB["t_hi"], t_now), C_B)
                self._curve(img, fB, fB["t_hi"],
                            fB["t_hi"] + p.cfg["extrapolate_t"],
                            C_PRED, dashed=True)

        # 현재 공
        if p.buf:
            _, xc, yc, zc = p.buf[-1]
            cv2.circle(img, self._pxA(yc, zc), 5, (255, 255, 255), -1)
            cv2.circle(img, self._pxB(yc, xc), 5, (255, 255, 255), -1)

        # 도착 예측
        a = p.arrival
        if a is not None:
            cv2.circle(img, self._pxA(a["Y"], a["Z"]), 7, C_TARGET, 2)
            cv2.circle(img, self._pxB(a["Y"], a["X"]), 7, C_TARGET, 2)
            cv2.putText(img,
                        "-> Y=%.2f in %4.0f ms (seg %s)  X=%+.3f  Z=%.3f m"
                        % (a["Y"], a["eta"] * 1000, a["seg"], a["X"], a["Z"]),
                        (MARGIN, y_hud + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_TARGET, 1)

        # HUD
        if fA is not None:
            gtxt = "g=%.2f%s" % (fA["g"],
                                 "(fix)" if p.cfg["gravity_fixed"] else "(fit)")
            cv2.putText(img, "n=%d  %s  RMS_A  Y=%.1f mm  Z=%.1f mm"
                        % (r["n"], gtxt, fA["rms_y"] * 1000,
                           fA["rms_z"] * 1000),
                        (MARGIN, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_A, 1)
            cv2.putText(img, "A: vY=%+.2f vX=%+.2f vZ0=%+.2f m/s"
                        % (fA["vy"], fA["vx"], fA["vz"]),
                        (MARGIN, y_hud + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, C_A, 1)
        if fB is not None:
            bcol = C_PRED if fB["pred"] else C_B
            cv2.putText(img, "B(%s): vY=%+.2f vX=%+.2f vZ0=%+.2f m/s"
                        % ("pred" if fB["pred"] else "meas",
                           fB["vy"], fB["vx"], fB["vz"]),
                        (MARGIN, y_hud + 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, bcol, 1)

        btxt = "BOUNCED" if r["bounced"] else "predicted"
        bpos = "-" if r["bounce"] is None else ("Y=%.2f X=%+.2f"
                                                % (r["bounce"][2],
                                                   r["bounce"][1]))
        cv2.putText(img, "bounce [%s]  %s" % (btxt, bpos),
                    (MARGIN, y_hud + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    C_B if r["bounced"] else C_PRED, 1)

        line = r.get("line")
        if p.cfg["line_only"]:
            if line is None:
                ltxt = "top line: not ready (need %d pts)" % p.cfg["line_min_points"]
            else:
                vxs = "-" if line["vx"] is None else \
                    "%+.2f(raw %+.2f x%.2f)" % (line["vx"], line["vx_raw"],
                                                line["gain"])
                vys = "-" if line["vy"] is None else "%+.2f" % line["vy"]
                ltxt = ("top line (straight): vY=%s vX=%s m/s  (first %.2fm, n=%d)"
                        % (vys, vxs, p.cfg["line_window_m"], line["n"]))
            cv2.putText(img, ltxt, (MARGIN, y_hud + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 120), 1)

        d = r.get("dbg")
        if (not r["bounced"]) and d:
            cv2.putText(img, "cand: z=%.3f(gate %.3f) fall=%.3f rise=%.3f"
                        % (d["z"], d["gate"], d["fall"], d["rise"]),
                        (MARGIN, y_hud + 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1)

        self._footer(img)
        return img

    def _footer(self, img):
        p = self.pred
        cv2.putText(img, "[FIT] pts=%d  gain=%.2f  %s   [ / ] : gain   c : clear"
                    % (len(p.buf), p.gain,
                       "REC" if p.recording else "PAUSED"),
                    (MARGIN, self.IH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255) if p.recording else (0, 165, 255), 1)

    def _status_text(self):
        p = self.pred
        r = p.res
        if r is None or not r.get("ok"):
            n = 0 if r is None else r.get("n", 0)
            if r is not None and r.get("reason") == "dir":
                return "진행 방향 불일치로 무시 중 (dY=%+.3f m)" % r["travel"]
            return "샘플 수집 중 %d/%d" % (n, p.cfg["min_points"])
        a = p.arrival
        head = ("n=%3d  바운스 %s" %
                (r["n"], "관측" if r["bounced"] else "예측"))
        if a is None:
            return head + "   |  타겟 통과 예측 없음"
        return (head + "   |  Y=%.2f 도착까지 %5.0f ms   X=%+.3f  Z=%.3f m   "
                "v=%.2f m/s" % (a["Y"], a["eta"] * 1000, a["X"], a["Z"],
                                a["v"]))

    # ---------------------------------------------------------------- 저장
    def save(self):
        os.makedirs(self.record_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ip = os.path.join(self.record_dir, "predict_%s.png" % stamp)
        cp = os.path.join(self.record_dir, "predict_%s.csv" % stamp)
        cv2.imwrite(ip, self._draw())
        r = self.pred.res
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["phase", "measured", "n", "t_lo_s", "t_hi_s",
                        "x0_m", "vx_ms", "y0_m", "vy_ms", "z0_m", "vz_ms",
                        "g_ms2", "rms_y_m", "rms_z_m",
                        "bounce_observed", "bounce_y_m", "bounce_x_m",
                        "target_y_m", "lat_gain"])
            if r is not None and r.get("ok"):
                bo = 1 if r["bounced"] else 0
                by = "" if r["bounce"] is None else round(r["bounce"][2], 4)
                bx = "" if r["bounce"] is None else round(r["bounce"][1], 4)
                for tag, ft in (("A", r["fitA"]), ("B", r["fitB"])):
                    if ft is None:
                        continue
                    w.writerow([tag, 0 if ft["pred"] else 1, ft["n"],
                                round(ft["t_lo"], 4), round(ft["t_hi"], 4),
                                round(ft["x0"], 4), round(ft["vx"], 4),
                                round(ft["y0"], 4), round(ft["vy"], 4),
                                round(ft["z0"], 4), round(ft["vz"], 4),
                                round(ft["g"], 3),
                                "" if ft["n"] == 0 else round(ft["rms_y"], 5),
                                "" if ft["n"] == 0 else round(ft["rms_z"], 5),
                                bo, by, bx, round(self.pred.target_y, 3),
                                round(self.pred.gain, 3)])
        print("[예측 저장] %s , %s" % (ip, cp))


# ============================================================================
# 유틸
# ============================================================================

def _dashed(img, p1, p2, color, thickness=1, dash=9, gap=7):
    x1, y1 = p1
    x2, y2 = p2
    d = math.hypot(x2 - x1, y2 - y1)
    if d < 1:
        return
    step = dash + gap
    for i in range(int(d // step) + 1):
        s = min(i * step / d, 1.0)
        e = min((i * step + dash) / d, 1.0)
        cv2.line(img,
                 (int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s)),
                 (int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e)),
                 color, thickness)


_PPM_COUNTER = [0]
_PPM_DIR = [None]


def _to_photo(bgr):
    """numpy BGR -> tk.PhotoImage (Pillow 없으면 임시 PPM 경유)"""
    if _HAS_PIL:
        return ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if _PPM_DIR[0] is None:
        import tempfile
        _PPM_DIR[0] = tempfile.mkdtemp(prefix="ttpredict_")
    _PPM_COUNTER[0] = (_PPM_COUNTER[0] + 1) % 4
    path = os.path.join(_PPM_DIR[0], "pred%d.ppm" % _PPM_COUNTER[0])
    if not cv2.imwrite(path, bgr):
        raise RuntimeError("임시 이미지 기록 실패: %s" % path)
    return tk.PhotoImage(file=path)


# ============================================================================
# 임베드용 렌더러 — 다른 창의 캔버스에 예측 그림을 그린다
# ============================================================================

class PredictorRenderer:
    """PredictorWindow 의 그리기 부분만 떼어낸 것.

    Tk 위젯을 만들지 않으므로, 통합 UI 처럼 이미 있는 캔버스에 그림만
    올리고 싶을 때 쓴다.  그리기 코드는 PredictorWindow 것을 그대로
    재사용하므로 두 벌로 갈라지지 않는다.

        r = PredictorRenderer(geo)
        r.pred.push(t, X)
        img = r.image()          # numpy BGR
    """

    #  PredictorWindow 의 그리기 메서드를 그대로 빌려 쓴다
    _pxA = PredictorWindow._pxA
    _pxB = PredictorWindow._pxB
    _make_base = PredictorWindow._make_base
    _curve = PredictorWindow._curve
    _draw = PredictorWindow._draw
    _footer = PredictorWindow._footer
    _status_text = PredictorWindow._status_text

    def __init__(self, geo, cfg=None, predictor=None):
        self.geo = geo
        self.pred = predictor or ParabolaPredictor(cfg)
        L = float(geo["table_length"])
        W = float(geo["table_width"])
        self.L, self.W, self.hw = L, W, W / 2.0
        self.S = PX_PER_M
        self.hA = int((Z_MAX_M - Z_MIN_M) * self.S)
        self.hB = int(W * self.S)
        self.yA = MARGIN
        self.yB = self.yA + self.hA + 52
        self.IW = int(L * self.S) + MARGIN * 2
        self.IH = self.yB + self.hB + 150
        self.base = self._make_base()

    # ------------------------------------------------------------------
    def push(self, t, X):
        self.pred.push(t, X)

    def image(self, *, after_net_only=True):
        """현재 상태를 그린 numpy BGR 이미지."""
        self.pred.tick()
        self.pred.compute_arrival(after_net_only=after_net_only,
                                  net_y=self.L / 2.0)
        return self._draw()

    def status_text(self) -> str:
        return self._status_text()
