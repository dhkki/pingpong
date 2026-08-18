#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_ball.py
=============
탁구 로봇 프로젝트 - 스테레오 탁구공 트래킹 + 실시간 3D 시각화.

settings/ 의 내부·외부 파라미터를 읽어, 두 카메라 영상에서 주황색 공을
검출하고 삼각측량으로 월드 좌표를 구한 뒤 3D 뷰로 보여준다.
현실과 육안 대조가 가능하도록 탁구대·네트·마스킹 영역·바닥 그림자를 함께 그린다.

필요 파일
---------
  settings/intrinsics_cam0.json, intrinsics_cam1.json
  settings/extrinsics_cam0.json, extrinsics_cam1.json
  settings/table_geometry.json
  settings/detector.json                (없으면 자동 생성)

실행
----
  python track_ball.py
  python track_ball.py --cam0 0 --cam1 1
  python track_ball.py --sim              # 카메라 없이 합성 공으로 파이프라인 점검

조작
----
  3D 뷰 드래그 : 시점 회전       휠 : 확대/축소      우클릭 드래그 : 이동
  SPACE : 일시정지 / 재개        T : 궤적 지우기
  M     : 마스크 보기 토글       R : 기록 시작/정지
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
    PIL_ERROR = None
except Exception as _e:
    HAS_PIL = False
    PIL_ERROR = "%s: %s" % (type(_e).__name__, _e)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_DIR = os.path.join(SCRIPT_DIR, "settings")
RECORD_DIR = os.path.join(SCRIPT_DIR, "records")

BG = "#1e1e22"
FG = "#e8e8ec"
ACC = "#4ea3ff"
OK_C = "#4fd07a"
WARN_C = "#ffc451"
BAD_C = "#ff6b6b"
DIM_C = "#8b8b95"
PANEL = "#26262c"
FONT = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 9, "bold")
FONT_T = ("Malgun Gothic", 11, "bold")
FONT_M = ("Consolas", 9)

BALL_D = 0.040          # 탁구공 지름 (m)
GRAVITY = -9.81

DEFAULT_DETECTOR = {
    "_comment": ("주황색 공 검출 파라미터. H 는 OpenCV 기준 0~179. "
                 "내부는 관대하게, ROI 경계 부근만 엄격하게 거른다."),
    "mode": "color",              # color | bright | both
    "h_lo": 15, "h_hi": 35,
    "s_lo": 50, "s_hi": 255,
    "v_lo": 60, "v_hi": 255,
    "bright_v_lo": 210,
    "bright_s_hi": 90,

    "blur_ksize": 5,              # HSV 변환 전 가우시안 블러
    "open_ksize": 3, "open_iter": 1,
    "close_ksize": 7, "close_iter": 2,

    "min_area": 10.0, "max_area": 50000.0,
    "min_radius_px": 2.0, "max_radius_px": 300.0,
    "min_circularity": 0.0,       # 모션 블러로 늘어난 타원도 통과시킨다
    "min_fill": 0.0,

    # ROI 경계 부근(테이블 흰 테두리 등)에서만 적용하는 엄격한 조건
    "border_band_px": 25.0,
    "border_min_area": 120.0,
    "border_min_radius": 6.0,
    "border_min_circularity": 0.55,

    "max_candidates": 8,
    "use_bgsub": False,
    "max_pair_dt_ms": 6.0,
    "max_reproj_px": 6.0,
    "roi_margin_m": 0.25,
    "use_roi_mask": True,
    "roi_mask_margin_px": 20,
    "compensate_async": True
}


# ============================================================================
# 설정 로드
# ============================================================================

def load_json(path, required=True, default=None):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        if default is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            return dict(default)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Calib:
    """한 카메라의 내부 + 외부 파라미터 묶음."""

    def __init__(self, cam_id):
        ip = os.path.join(SETTINGS_DIR, "intrinsics_cam%d.json" % cam_id)
        ep = os.path.join(SETTINGS_DIR, "extrinsics_cam%d.json" % cam_id)
        for p, what in ((ip, "내부"), (ep, "외부")):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "cam%d 의 %s 파라미터가 없습니다:\n%s" % (cam_id, what, p))
        di = load_json(ip)
        de = load_json(ep)
        self.cam_id = cam_id
        self.K = np.array(di["K"], dtype=np.float64)
        self.dist = np.array(di["dist"], dtype=np.float64).reshape(1, -1)
        self.w, self.h = int(di["img_w"]), int(di["img_h"])
        self.R = np.array(de["R"], dtype=np.float64)
        self.t = np.array(de["tvec"], dtype=np.float64).reshape(3, 1)
        self.rvec, _ = cv2.Rodrigues(self.R)
        self.C = (-self.R.T @ self.t).ravel()
        self.device_index = int(de.get("device_index", cam_id))
        self.rms = float(de.get("reprojection_rms_px", 0.0))
        self.P = self.K @ np.hstack([self.R, self.t])       # 왜곡 포함 전 투영
        self.Pn = np.hstack([self.R, self.t])               # 정규화 좌표용

    def project(self, X):
        """월드 3D (N,3) -> 이미지 (N,2). 왜곡 포함."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        pts, _ = cv2.projectPoints(X, self.rvec, self.t, self.K, self.dist)
        return pts.reshape(-1, 2)

    def normalize(self, uv):
        """이미지 좌표 -> 왜곡 보정된 정규화 좌표 (N,2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)


# ============================================================================
# 카메라
# ============================================================================

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY,
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)}


class CameraThread:
    """타임스탬프가 붙은 최신 프레임 버퍼."""

    def __init__(self, index, w, h, fps=120, fourcc="MJPG", backend="dshow",
                 buf=8):
        self.exposure_req = None      # None = 변경 없음, "auto" = 자동
        self._exposure_applied = "init"
        self.index, self.w, self.h = index, w, h
        self.fps, self.fourcc, self.backend = fps, fourcc, backend
        self.buf = deque(maxlen=buf)
        self.lock = threading.Lock()
        self.running = False
        self.error = None
        self.fps_measured = 0.0
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        cap = cv2.VideoCapture(self.index,
                               BACKENDS.get(self.backend, cv2.CAP_ANY))
        if not cap.isOpened():
            self.error = "카메라 %d 를 열 수 없습니다." % self.index
            self.running = False
            return
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        t0, n = time.time(), 0
        while self.running:
            if self.exposure_req != self._exposure_applied:
                req = self.exposure_req
                try:
                    if req == "auto" or req is None:
                        for v in (0.75, 1, 3):
                            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
                    else:
                        for v in (0.25, 0):
                            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
                        cap.set(cv2.CAP_PROP_EXPOSURE, float(req))
                except Exception:
                    pass
                self._exposure_applied = req
            ok, f = cap.read()
            ts = time.perf_counter()
            if not ok or f is None:
                time.sleep(0.002)
                continue
            with self.lock:
                self.buf.append((ts, f))
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps_measured = n / (time.time() - t0)
                t0, n = time.time(), 0
        cap.release()

    def snapshot(self):
        with self.lock:
            return list(self.buf)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
        self.thread = None


class SimCamera:
    """--sim 용. 저장된 외부 파라미터로 합성 장면을 렌더링한다."""

    def __init__(self, calib, geo, sim_state, time_offset=0.0):
        self.calib, self.geo, self.sim = calib, geo, sim_state
        self.time_offset = float(time_offset)
        self.fps_measured = 0.0
        self.error = None
        self._n, self._t0 = 0, time.time()
        self.bg = self._render_background()
        self._noise = [np.random.randint(-4, 5, (calib.h, calib.w, 3)
                                         ).astype(np.int16) for _ in range(6)]

    def start(self):
        pass

    def stop(self):
        pass

    def _P(self, p):
        return self.calib.project(np.array([p], float))[0]

    def _render_background(self):
        c, g = self.calib, self.geo
        img = np.full((c.h, c.w, 3), 32, np.uint8)
        L, hw = g["table_length"], g["table_width"] / 2
        ny, nh = L / 2, g["net_height"]
        nx = hw + g["net_overhang"]
        r = g["roi"]
        quad = np.array([self._P((-hw, 0, 0)), self._P((hw, 0, 0)),
                         self._P((hw, L, 0)), self._P((-hw, L, 0))], np.int32)
        cv2.fillPoly(img, [quad], (110, 55, 25))
        cv2.polylines(img, [quad], True, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(img, tuple(self._P((0, 0, 0)).astype(int)),
                 tuple(self._P((0, L, 0)).astype(int)),
                 (235, 235, 235), 1, cv2.LINE_AA)
        tape = [((r["x_min"], r["y_min"], 0), (r["x_min"], r["y_max"], 0)),
                ((r["x_max"], r["y_min"], 0), (r["x_max"], r["y_max"], 0)),
                ((r["x_min"], r["y_min"], 0), (r["x_max"], r["y_min"], 0)),
                ((r["x_min"], r["y_max"], 0), (r["x_max"], r["y_max"], 0))]
        for a, b in tape:
            cv2.line(img, tuple(self._P(a).astype(int)),
                     tuple(self._P(b).astype(int)), (70, 200, 235), 3,
                     cv2.LINE_AA)
        netq = np.array([self._P((-nx, ny, 0)), self._P((nx, ny, 0)),
                         self._P((nx, ny, nh)), self._P((-nx, ny, nh))],
                        np.int32)
        cv2.fillPoly(img, [netq], (75, 75, 75))
        cv2.polylines(img, [netq], True, (245, 245, 245), 2, cv2.LINE_AA)
        return img

    def snapshot(self):
        ts, X = self.sim.sample(self.time_offset)
        img = self.bg.copy()
        cam = self.calib.R @ X.reshape(3, 1) + self.calib.t
        if cam[2, 0] > 0.05:
            uv = self.calib.project(X.reshape(1, 3))[0]
            rad = self.calib.K[0, 0] * (BALL_D / 2) / cam[2, 0]
            if 0 <= uv[0] < self.calib.w and 0 <= uv[1] < self.calib.h \
                    and rad >= 1:
                cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))),
                           max(int(round(rad)), 2), (30, 120, 245), -1,
                           cv2.LINE_AA)
                cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))),
                           max(int(round(rad)), 2), (60, 160, 255), 1,
                           cv2.LINE_AA)
        if self._noise:
            k = np.random.randint(0, len(self._noise))
            img = cv2.add(img, self._noise[k], dtype=cv2.CV_8U)
        self._n += 1
        if time.time() - self._t0 >= 1.0:
            self.fps_measured = self._n / (time.time() - self._t0)
            self._n, self._t0 = 0, time.time()
        return [(ts, img)]


class SimBall:
    """포물선 + 바운스로 ROI 안을 왕복하는 가상 공."""

    def __init__(self, geo):
        self.geo = geo
        self.frame_t = time.perf_counter()
        self.reset()

    def sample(self, offset=0.0):
        """두 가상 카메라가 같은 프레임(약간의 시간차 포함)을 보게 한다."""
        now = time.perf_counter()
        if offset <= 0.0:          # cam0 이 프레임 시각을 정하고, cam1 이 따라옴
            self.frame_t = now
        t = self.frame_t + offset
        return t, self.position(t)

    def reset(self):
        r = self.geo["roi"]
        self.t0 = time.perf_counter()
        self.p0 = np.array([np.random.uniform(r["x_min"], r["x_max"]) * 0.7,
                            r["y_max"] + 0.25, 0.55])
        self.v0 = np.array([np.random.uniform(-0.25, 0.25), -3.2,
                            np.random.uniform(0.2, 0.9)])

    def position(self, t_abs=None):
        t = (time.perf_counter() if t_abs is None else t_abs) - self.t0
        p = self.p0 + self.v0 * t + np.array([0, 0, 0.5 * GRAVITY * t * t])
        if p[2] < BALL_D / 2:                       # 바운스
            self.p0 = np.array([p[0], p[1], BALL_D / 2])
            vz = self.v0[2] + GRAVITY * t
            self.v0 = np.array([self.v0[0], self.v0[1], -vz * 0.75])
            self.t0 = (time.perf_counter() if t_abs is None else t_abs)
            p = self.p0.copy()
        if p[1] < self.geo["roi"]["y_min"] - 0.3 or t > 6:
            self.reset()
            return self.p0.copy()
        return p


# ============================================================================
# 탐지 영역 (ROI 를 각 카메라 화면에 투영한 마스크)
# ============================================================================

def roi_image_region(calib, geo, margin_px=20, n=8):
    """ROI 볼륨을 카메라 화면에 투영해 (마스크, 외곽 폴리곤) 반환.

    ROI 밖(나무 바닥 등)에서 생기는 오탐을 원천 차단한다.
    ROI 일부가 카메라 뒤에 있거나 화면을 벗어나면 그만큼만 잘라 쓴다.
    """
    r = geo["roi"]
    xs = np.linspace(r["x_min"], r["x_max"], n)
    ys = np.linspace(r["y_min"], r["y_max"], n)
    zs = np.linspace(r["z_min"], r["z_max"], n)
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                # 표면만 (내부는 볼록껍질에 영향 없음)
                if (x in (xs[0], xs[-1]) or y in (ys[0], ys[-1])
                        or z in (zs[0], zs[-1])):
                    pts.append((x, y, z))
    P = np.array(pts, dtype=np.float64)
    cam = (calib.R @ P.T + calib.t).T
    front = cam[:, 2] > 0.20
    if front.sum() < 3:
        return None, None
    uv = calib.project(P[front])
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) < 3:
        return None, None
    uv[:, 0] = np.clip(uv[:, 0], -4 * calib.w, 5 * calib.w)
    uv[:, 1] = np.clip(uv[:, 1], -4 * calib.h, 5 * calib.h)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    mask = np.zeros((calib.h, calib.w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    if margin_px > 0:
        k = int(margin_px) * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)))
    # 화면에 그릴 폴리곤도 같은 여유만큼 넓힌 외곽선으로
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = max(cs, key=cv2.contourArea).reshape(-1, 2) if cs else None
    return mask, poly


# ============================================================================
# 공 검출
# ============================================================================

class BallDetector:
    def __init__(self, params):
        self.p = dict(params)
        self.bgsub = None
        self.stats = {}

    def set(self, key, value):
        self.p[key] = value

    def _mask(self, bgr):
        p = self.p
        kb = int(p.get("blur_ksize", 5))
        if kb >= 3:
            if kb % 2 == 0:
                kb += 1
            bgr = cv2.GaussianBlur(bgr, (kb, kb), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo_s, hi_s = int(p["s_lo"]), int(p["s_hi"])
        lo_v, hi_v = int(p["v_lo"]), int(p["v_hi"])
        h_lo, h_hi = int(p["h_lo"]), int(p["h_hi"])
        if h_lo <= h_hi:
            m = cv2.inRange(hsv, (h_lo, lo_s, lo_v), (h_hi, hi_s, hi_v))
        else:                                    # 빨강 쪽으로 감기는 범위
            m1 = cv2.inRange(hsv, (h_lo, lo_s, lo_v), (179, hi_s, hi_v))
            m2 = cv2.inRange(hsv, (0, lo_s, lo_v), (h_hi, hi_s, hi_v))
            m = cv2.bitwise_or(m1, m2)

        mode = str(p.get("mode", "color"))
        if mode in ("bright", "both"):
            bright = cv2.inRange(
                hsv, (0, 0, int(p.get("bright_v_lo", 210))),
                (179, int(p.get("bright_s_hi", 90)), 255))
            m = bright if mode == "bright" else cv2.bitwise_or(m, bright)

        ko, ki = int(p.get("open_ksize", 3)), int(p.get("open_iter", 1))
        kc, kci = int(p.get("close_ksize", 7)), int(p.get("close_iter", 2))
        if ko >= 3 and ki > 0:
            m = cv2.morphologyEx(
                m, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)),
                iterations=ki)
        if kc >= 3 and kci > 0:
            m = cv2.morphologyEx(
                m, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)),
                iterations=kci)

        if p.get("use_bgsub"):
            if self.bgsub is None:
                self.bgsub = cv2.createBackgroundSubtractorMOG2(
                    history=300, varThreshold=24, detectShadows=False)
            fg = self.bgsub.apply(bgr)
            fg = cv2.dilate(fg, np.ones((5, 5), np.uint8))
            m = cv2.bitwise_and(m, fg)
        return m

    def detect(self, bgr, return_mask=False, roi_mask=None, roi_poly=None):
        """returns (candidates, mask). candidate = dict(uv, r, score)

        ROI 안쪽은 관대하게 통과시키고(모션 블러 대응),
        ROI 경계 밴드에서만 면적·반지름·원형도를 엄격히 본다.
        """
        p = self.p
        mask = self._mask(bgr)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        band = float(p.get("border_band_px", 25.0))
        poly = None
        if roi_poly is not None and len(roi_poly) >= 3:
            poly = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 1, 2)

        st = dict(blobs=len(cnts), area=0, radius=0, circ=0, border=0,
                  passed=0, mask_px=int((mask > 0).sum()),
                  best_r=0.0, best_circ=0.0, best_area=0.0)
        out = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            perim = float(cv2.arcLength(c, True))
            if perim <= 0:
                st["area"] += 1
                continue
            circ = 4 * math.pi * area / (perim * perim)
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            if area > st["best_area"]:
                st["best_area"], st["best_r"], st["best_circ"] = \
                    area, float(rad), float(circ)
            if area > p["max_area"] or rad > p["max_radius_px"]:
                st["area"] += 1
                continue

            in_border = False
            if poly is not None:
                d = cv2.pointPolygonTest(poly, (float(cx), float(cy)), True)
                in_border = d < band
            if in_border:
                if (area < p["border_min_area"]
                        or rad < p["border_min_radius"]
                        or circ < p["border_min_circularity"]):
                    st["border"] += 1
                    continue
            else:
                if area < p["min_area"]:
                    st["area"] += 1
                    continue
                if rad < p["min_radius_px"]:
                    st["radius"] += 1
                    continue
                if circ < p["min_circularity"]:
                    st["circ"] += 1
                    continue
                if p.get("min_fill", 0.0) > 0:
                    if area / (math.pi * rad * rad) < p["min_fill"]:
                        st["circ"] += 1
                        continue

            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            st["passed"] += 1
            out.append(dict(uv=(float(cx), float(cy)), r=float(rad),
                            score=float(area * max(circ, 1e-3)),
                            area=area, circ=float(circ)))
        out.sort(key=lambda d: -d["score"])
        out = out[:int(p["max_candidates"])]
        self.stats = st
        return (out, mask) if return_mask else (out, None)


# ============================================================================
# 삼각측량 / 매칭
# ============================================================================

def triangulate(c0, c1, uv0, uv1, vel=None, dt=0.0):
    """비동기 보정 포함 선형 삼각측량. 반환 (X(3,), err0, err1)"""
    n0 = c0.normalize([uv0])[0]
    n1 = c1.normalize([uv1])[0]
    P0 = c0.Pn
    shift = np.zeros(3)
    if vel is not None and abs(dt) > 1e-6:
        # cam1 은 dt 뒤(또는 앞)의 공을 봤다고 보고 보정
        shift = np.asarray(vel, dtype=np.float64).ravel() * dt
        t1 = c1.t + (c1.R @ shift.reshape(3, 1))
        P1 = np.hstack([c1.R, t1])
    else:
        P1 = c1.Pn
    X4 = cv2.triangulatePoints(P0, P1,
                               n0.reshape(2, 1), n1.reshape(2, 1))
    if abs(X4[3, 0]) < 1e-12:
        return None, 1e9, 1e9
    X = (X4[:3] / X4[3]).ravel()
    # 카메라 앞쪽인지
    if (c0.R @ X.reshape(3, 1) + c0.t)[2, 0] <= 0.05:
        return None, 1e9, 1e9
    X1 = X + shift                       # cam1 이 실제로 본 위치
    if (c1.R @ X1.reshape(3, 1) + c1.t)[2, 0] <= 0.05:
        return None, 1e9, 1e9
    e0 = float(np.linalg.norm(c0.project(X.reshape(1, 3))[0] - np.array(uv0)))
    e1 = float(np.linalg.norm(c1.project(X1.reshape(1, 3))[0] - np.array(uv1)))
    return X, e0, e1


def expected_radius_px(calib, X):
    z = float((calib.R @ X.reshape(3, 1) + calib.t)[2, 0])
    if z <= 1e-6:
        return 0.0
    return float(calib.K[0, 0] * (BALL_D / 2) / z)


def match_candidates(c0, c1, cands0, cands1, params, geo, vel=None, dt=0.0,
                     predicted=None):
    """가능한 조합 중 최적 1개 선택. 반환 dict 또는 None"""
    r = geo["roi"]
    mg = params["roi_margin_m"]
    best = None
    for a in cands0:
        for b in cands1:
            X, e0, e1 = triangulate(c0, c1, a["uv"], b["uv"], vel, dt)
            if X is None:
                continue
            if max(e0, e1) > params["max_reproj_px"]:
                continue
            if not (r["x_min"] - mg <= X[0] <= r["x_max"] + mg and
                    r["y_min"] - mg <= X[1] <= r["y_max"] + mg and
                    r["z_min"] - mg <= X[2] <= r["z_max"] + mg):
                continue
            # 반지름 정합
            r0e, r1e = expected_radius_px(c0, X), expected_radius_px(c1, X)
            rp = 0.0
            for rr, re in ((a["r"], r0e), (b["r"], r1e)):
                if re > 0.5:
                    rp += abs(rr - re) / re
            cost = (e0 + e1) + rp * 2.0
            if predicted is not None:
                cost += float(np.linalg.norm(X - predicted)) * 8.0
            if best is None or cost < best["cost"]:
                best = dict(X=X, e0=e0, e1=e1, cost=cost, uv0=a["uv"],
                            uv1=b["uv"], r0=a["r"], r1=b["r"],
                            r0e=r0e, r1e=r1e)
    return best


def roi_coverage(calibs, geo, n=(13, 21, 9)):
    """ROI 를 격자로 훑어 두 카메라 동시 가시 비율을 계산."""
    r = geo["roi"]
    xs = np.linspace(r["x_min"], r["x_max"], n[0])
    ys = np.linspace(r["y_min"], r["y_max"], n[1])
    zs = np.linspace(r["z_min"], r["z_max"], n[2])
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    vis = np.ones(len(grid), dtype=bool)
    per_cam = []
    for c in calibs:
        cam = (c.R @ grid.T + c.t).T
        uv = c.project(grid)
        ok = (cam[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < c.w) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < c.h)
        per_cam.append(ok)
        vis &= ok
    return grid, vis, per_cam


# ============================================================================
# 칼만 필터 (등가속 + 중력)
# ============================================================================

class BallKF:
    def __init__(self, pos_noise=0.004, accel_noise=40.00000):
        self.x = None                      # [px,py,pz,vx,vy,vz]
        self.P = None
        self.q = accel_noise
        self.r = pos_noise
        self.miss = 0
        self.t_last = None

    def reset(self):
        self.x, self.P, self.miss, self.t_last = None, None, 0, None

    def predict_to(self, t):
        if self.x is None:
            return None
        dt = t - self.t_last
        if dt <= 0:
            return self.x[:3].copy()
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        x = F @ self.x
        x[2] += 0.5 * GRAVITY * dt * dt
        x[5] += GRAVITY * dt
        return x[:3].copy()

    def update(self, z, t):
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if self.x is None:
            self.x = np.hstack([z, np.zeros(3)])
            self.P = np.diag([0.01] * 3 + [1.0] * 3)
            self.t_last = t
            self.miss = 0
            return True
        dt = max(t - self.t_last, 1e-4)
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        x = F @ self.x
        x[2] += 0.5 * GRAVITY * dt * dt
        x[5] += GRAVITY * dt
        G = np.vstack([np.eye(3) * (dt * dt / 2), np.eye(3) * dt])
        Q = G @ G.T * (self.q ** 2)
        P = F @ self.P @ F.T + Q

        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        R = np.eye(3) * (self.r ** 2)
        y = z - H @ x
        S = H @ P @ H.T + R
        # 게이팅
        try:
            md = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            md = 1e9
        if md > 60.0 or float(np.linalg.norm(y)) > 0.35:   # 이상치
            self.miss += 1
            self.x, self.P, self.t_last = x, P, t
            if self.miss > 4:
                self.reset()
                self.x = np.hstack([z, np.zeros(3)])
                self.P = np.diag([0.01] * 3 + [1.0] * 3)
                self.t_last = t
            return False
        Kg = P @ H.T @ np.linalg.inv(S)
        self.x = x + Kg @ y
        self.P = (np.eye(6) - Kg @ H) @ P
        self.t_last = t
        self.miss = 0
        return True

    @property
    def pos(self):
        return None if self.x is None else self.x[:3].copy()

    @property
    def vel(self):
        return None if self.x is None else self.x[3:].copy()


# ============================================================================
# 3D 뷰어
# ============================================================================

class Viewer3D:
    def __init__(self, w=620, h=560):
        self.w, self.h = w, h
        self.az = -50.0
        self.el = 24.0
        self.dist = 3.6
        self.target = np.array([0.0, 0.8, 0.3])
        self.f = 620.0

    def set_preset(self, name):
        presets = {
            "로봇 시점": (-90.0, 18.0, 3.2, [0.0, 0.8, 0.25]),
            "사람 시점": (90.0, 18.0, 3.2, [0.0, 0.8, 0.25]),
            "측면": (0.0, 12.0, 3.4, [0.0, 0.8, 0.3]),
            "상단": (-90.0, 78.0, 3.4, [0.0, 0.8, 0.2]),
            "비스듬히": (-50.0, 24.0, 3.6, [0.0, 0.8, 0.3]),
        }
        if name in presets:
            a, e, d, t = presets[name]
            self.az, self.el, self.dist = a, e, d
            self.target = np.array(t, dtype=np.float64)

    def orbit(self, dx, dy):
        # 방위각을 키우면 화면 속 물체가 왼쪽으로 이동한다.
        # 드래그 방향을 그대로 따라가려면 부호를 뒤집어야 한다.
        self.az -= dx * 0.4
        self.el = max(-85.0, min(85.0, self.el + dy * 0.4))

    def pan(self, dx, dy):
        a = math.radians(self.az)
        right = np.array([math.sin(a), -math.cos(a), 0.0])
        self.target = self.target - right * dx * 0.004 \
            + np.array([0, 0, 1.0]) * dy * 0.004

    def zoom(self, s):
        self.dist = max(0.8, min(12.0, self.dist * s))

    def _pose(self):
        a, e = math.radians(self.az), math.radians(self.el)
        eye = self.target + self.dist * np.array(
            [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
             math.sin(e)])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= nr
        down = np.cross(fwd, right)
        R = np.vstack([right, down, fwd])
        t = -R @ eye
        return R, t.reshape(3, 1), eye

    def project(self, R, t, X):
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        cam = (R @ X.T + t).T
        z = cam[:, 2]
        uv = np.full((len(X), 2), np.nan)
        ok = z > 0.05
        uv[ok, 0] = self.f * cam[ok, 0] / z[ok] + self.w / 2
        uv[ok, 1] = self.f * cam[ok, 1] / z[ok] + self.h / 2
        return uv, z

    def render(self, geo, ball=None, trail=None, cams=None, raw=None,
               show_cams=True, coverage=None):
        img = np.full((self.h, self.w, 3), 24, np.uint8)
        R, t, eye = self._pose()

        def pr(pts):
            uv, z = self.project(R, t, pts)
            return uv, z

        def line(a, b, col, th=1):
            uv, z = pr([a, b])
            if not np.all(np.isfinite(uv)):
                return
            if np.max(np.abs(uv)) > 1e5:
                return
            cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)),
                     col, th, cv2.LINE_AA)

        L = geo["table_length"]
        hw = geo["table_width"] / 2
        nh, ny = geo["net_height"], L / 2
        nx = hw + geo["net_overhang"]
        r = geo["roi"]

        # 바닥 그리드 (탁구대 아래 0.76 m)
        for gx in np.arange(-2.0, 2.01, 0.5):
            line((gx, -1.0, -0.76), (gx, 3.8, -0.76), (44, 44, 48))
        for gy in np.arange(-1.0, 3.81, 0.5):
            line((-2.0, gy, -0.76), (2.0, gy, -0.76), (44, 44, 48))

        # 탁구대 상판
        quad = np.array([[-hw, 0, 0], [hw, 0, 0], [hw, L, 0], [-hw, L, 0]])
        uv, z = pr(quad)
        if np.all(np.isfinite(uv)) and np.max(np.abs(uv)) < 1e5:
            cv2.fillPoly(img, [uv.astype(np.int32)], (96, 48, 22))
            cv2.polylines(img, [uv.astype(np.int32)], True,
                          (225, 225, 225), 2, cv2.LINE_AA)
        line((0, 0, 0), (0, L, 0), (215, 215, 215), 1)

        # 마스킹 영역 (노랑)
        tape = [((r["x_min"], r["y_min"], 0), (r["x_min"], r["y_max"], 0)),
                ((r["x_max"], r["y_min"], 0), (r["x_max"], r["y_max"], 0)),
                ((r["x_min"], r["y_min"], 0), (r["x_max"], r["y_min"], 0)),
                ((r["x_min"], r["y_max"], 0), (r["x_max"], r["y_max"], 0))]
        for a, b in tape:
            line(a, b, (70, 205, 240), 2)

        # ROI 박스
        x0, x1, y0, y1 = r["x_min"], r["x_max"], r["y_min"], r["y_max"]
        z0, z1 = r["z_min"], r["z_max"]
        bot = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
        top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for i in range(4):
            line(top[i], top[(i + 1) % 4], (90, 90, 210), 1)
            line(bot[i], top[i], (70, 70, 170), 1)

        # 네트
        netq = np.array([[-nx, ny, 0], [nx, ny, 0], [nx, ny, nh],
                         [-nx, ny, nh]])
        uv, z = pr(netq)
        if np.all(np.isfinite(uv)) and np.max(np.abs(uv)) < 1e5:
            ov = img.copy()
            cv2.fillPoly(ov, [uv.astype(np.int32)], (150, 150, 150))
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, [uv.astype(np.int32)], True,
                          (245, 245, 245), 2, cv2.LINE_AA)

        # 월드 축
        line((0, 0, 0), (0.3, 0, 0), (60, 60, 255), 3)
        line((0, 0, 0), (0, 0.3, 0), (60, 220, 60), 3)
        line((0, 0, 0), (0, 0, 0.3), (255, 140, 60), 3)

        # 카메라 위치
        if show_cams and cams:
            for c in cams:
                uv, z = pr([c.C])
                if np.isfinite(uv[0]).all() and np.max(np.abs(uv)) < 1e5:
                    p = tuple(uv[0].astype(int))
                    cv2.drawMarker(img, p, (200, 200, 90),
                                   cv2.MARKER_TRIANGLE_UP, 12, 2)
                    cv2.putText(img, "cam%d" % c.cam_id, (p[0] + 8, p[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (200, 200, 90), 1, cv2.LINE_AA)
                    line(c.C, c.C + c.R.T @ np.array([0, 0, 0.35]),
                         (150, 150, 70), 1)

        # 시야 커버리지 (사각지대 = 빨강)
        if coverage is not None:
            grid, vis = coverage
            uv, zz = self.project(R, t, grid)
            for i in range(0, len(grid), 1):
                if not np.isfinite(uv[i]).all() or abs(uv[i]).max() > 1e5:
                    continue
                col = (70, 160, 70) if vis[i] else (60, 60, 235)
                cv2.circle(img, tuple(uv[i].astype(int)), 1, col, -1)

        # 궤적
        if trail and len(trail) >= 2:
            ts = np.array([t for t, _ in trail])
            pts = np.array([p for _, p in trail])
            uv, z = pr(pts)
            for i in range(1, len(pts)):
                if not (np.isfinite(uv[i]).all() and np.isfinite(uv[i - 1]).all()):
                    continue
                if max(abs(uv[i]).max(), abs(uv[i - 1]).max()) > 1e5:
                    continue
                # 추적이 끊긴 구간은 선을 잇지 않는다
                if (ts[i] - ts[i - 1]) > 1.00000:
                    continue
                if np.linalg.norm(pts[i] - pts[i - 1]) > 5.00000:
                    continue
                a = i / len(pts)
                col = (int(80 + 100 * a), int(140 + 90 * a), int(60 + 60 * a))
                cv2.line(img, tuple(uv[i - 1].astype(int)),
                         tuple(uv[i].astype(int)), col, 2, cv2.LINE_AA)

        # 원시 관측(필터 전)
        if raw is not None:
            uv, z = pr([raw])
            if np.isfinite(uv[0]).all() and np.max(np.abs(uv)) < 1e5:
                cv2.drawMarker(img, tuple(uv[0].astype(int)), (120, 120, 255),
                               cv2.MARKER_TILTED_CROSS, 10, 1)

        # 공 + 바닥 그림자 + 낙하선
        if ball is not None:
            shadow = np.array([ball[0], ball[1], 0.0])
            uvs, _ = pr([shadow])
            uvb, zb = pr([ball])
            if np.isfinite(uvs[0]).all() and np.max(np.abs(uvs)) < 1e5:
                cv2.ellipse(img, tuple(uvs[0].astype(int)),
                            (max(int(self.f * 0.02 / max(self.dist, 0.1)), 4),
                             max(int(self.f * 0.008 / max(self.dist, 0.1)), 2)),
                            0, 0, 360, (18, 18, 18), -1, cv2.LINE_AA)
            if np.isfinite(uvb[0]).all() and np.max(np.abs(uvb)) < 1e5:
                if np.isfinite(uvs[0]).all():
                    cv2.line(img, tuple(uvb[0].astype(int)),
                             tuple(uvs[0].astype(int)), (90, 90, 90), 1)
                rad = max(int(self.f * (BALL_D / 2) / max(zb[0], 0.05)), 4)
                cv2.circle(img, tuple(uvb[0].astype(int)), rad,
                           (40, 140, 255), -1, cv2.LINE_AA)
                cv2.circle(img, tuple(uvb[0].astype(int)), rad,
                           (255, 255, 255), 1, cv2.LINE_AA)
        return img


# ============================================================================
# 메인 앱
# ============================================================================

class TrackerApp(tk.Tk):

    def __init__(self, args):
        super().__init__()
        self.title("탁구 로봇 - 스테레오 공 트래킹")
        self.configure(bg=BG)
        self.args = args
        self._closing = False
        self._after_id = None

        self.geo = load_json(os.path.join(SETTINGS_DIR, "table_geometry.json"))
        self.det_params = load_json(
            os.path.join(SETTINGS_DIR, "detector.json"),
            required=False, default=DEFAULT_DETECTOR)
        for k, v in DEFAULT_DETECTOR.items():
            self.det_params.setdefault(k, v)

        self.calib = [Calib(0), Calib(1)]
        raw = self.det_params
        if "cam0" in raw or "cam1" in raw:      # 새 형식
            base = {k: v for k, v in raw.items()
                    if k not in ("cam0", "cam1")}
            self.params = []
            for i in (0, 1):
                d = dict(DEFAULT_DETECTOR); d.update(base)
                d.update(raw.get("cam%d" % i, {}))
                self.params.append(d)
        else:                                   # 기존 평면 형식
            self.params = [dict(raw), dict(raw)]
        self.det_params = self.params[0]         # 공통 옵션 참조용
        self.detector = [BallDetector(self.params[0]),
                         BallDetector(self.params[1])]
        self.slider_target = 2                   # 0=cam0, 1=cam1, 2=양쪽
        self.kf = BallKF()
        self.viewer = Viewer3D()
        self.trail = deque(maxlen=260)
        self.paused = False
        self.show_mask = False
        self.recording = False
        self.rec_rows = []
        self.last = None
        self.stat_n, self.stat_hit, self.stat_t0 = 0, 0, time.time()
        self.coverage = None
        self.show_coverage = False
        self.roi_mask = [None, None]
        self.roi_poly = [None, None]
        self.pick_mode = False
        self._last_frames = [None, None]
        self.loop_fps = 0.0
        self._photo = [None, None]
        self._photo3d = None
        self.preview_scale = args.preview_scale

        if args.sim:
            self.sim = SimBall(self.geo)
            self.cams = [
                SimCamera(self.calib[0], self.geo, self.sim, 0.0),
                SimCamera(self.calib[1], self.geo, self.sim,
                          args.sim_offset_ms / 1000.0)]
        else:
            self.sim = None
            self.cams = [
                CameraThread(args.cam0 if args.cam0 is not None
                             else self.calib[0].device_index,
                             self.calib[0].w, self.calib[0].h, args.fps,
                             args.fourcc, args.backend),
                CameraThread(args.cam1 if args.cam1 is not None
                             else self.calib[1].device_index,
                             self.calib[1].w, self.calib[1].h, args.fps,
                             args.fourcc, args.backend)]
        for c in self.cams:
            c.start()

        self._rebuild_roi_masks()
        self._build_ui()
        self._bind_events()
        self._tick()

    def _rebuild_roi_masks(self):
        m = int(self.det_params.get("roi_mask_margin_px", 20))
        for i in (0, 1):
            try:
                self.roi_mask[i], self.roi_poly[i] = roi_image_region(
                    self.calib[i], self.geo, m)
            except Exception:
                self.roi_mask[i], self.roi_poly[i] = None, None

    # ------------------------------------------------------------------ UI
    def _section(self, parent, title, side=None):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", pady=(0, 6))
        tk.Label(outer, text=title, bg=BG, fg=ACC, font=FONT_B,
                 anchor="w").pack(fill="x")
        inner = tk.Frame(outer, bg=PANEL, padx=8, pady=6)
        inner.pack(fill="x")
        return inner

    def _build_ui(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TButton", font=FONT, padding=3)

        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        pw = int(self.calib[0].w * self.preview_scale)
        ph = int(self.calib[0].h * self.preview_scale)

        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="n")
        self.cam_canvas = []
        self.cam_label = []
        for i in (0, 1):
            tk.Label(left, text="cam%d" % i, bg=BG, fg=ACC, font=FONT_B,
                     anchor="w").pack(fill="x")
            cv_ = tk.Canvas(left, width=pw, height=ph, bg="#000",
                            highlightthickness=1, highlightbackground="#444")
            cv_.pack()
            cv_.bind("<Button-1>", lambda e, k=i: self.on_pick(k, e))
            self.cam_canvas.append(cv_)
            lb = tk.Label(left, text="", bg=BG, fg=DIM_C, font=FONT_M,
                          anchor="w")
            lb.pack(fill="x", pady=(0, 6))
            self.cam_label.append(lb)

        mid = tk.Frame(root, bg=BG)
        mid.grid(row=0, column=1, sticky="n", padx=(10, 0))
        tk.Label(mid, text="3D 뷰  (드래그 회전 · 휠 확대 · 우클릭 이동)",
                 bg=BG, fg=ACC, font=FONT_B, anchor="w").pack(fill="x")
        self.canvas3d = tk.Canvas(mid, width=self.viewer.w,
                                  height=self.viewer.h, bg="#000",
                                  highlightthickness=1,
                                  highlightbackground="#444")
        self.canvas3d.pack()
        vb = tk.Frame(mid, bg=BG)
        vb.pack(fill="x", pady=(4, 0))
        for name in ("비스듬히", "로봇 시점", "사람 시점", "측면", "상단"):
            ttk.Button(vb, text=name, width=9,
                       command=lambda n=name: self.viewer.set_preset(n)
                       ).pack(side="left", padx=1)

        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=2, sticky="n", padx=(10, 0))

        f_st = self._section(right, "측정값")
        self.lbl_pos = tk.Label(f_st, bg=PANEL, fg=OK_C, font=("Consolas", 12),
                                justify="left", anchor="w")
        self.lbl_pos.pack(fill="x")
        self.lbl_det = tk.Label(f_st, bg=PANEL, fg=FG, font=FONT_M,
                                justify="left", anchor="w")
        self.lbl_det.pack(fill="x", pady=(4, 0))

        f_ct = self._section(right, "조작")
        g = tk.Frame(f_ct, bg=PANEL); g.pack(fill="x")
        self.btn_pause = ttk.Button(g, text="일시정지 (Space)", width=17,
                                    command=self.toggle_pause)
        self.btn_pause.grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(g, text="궤적 지우기 (T)", width=17,
                   command=self.clear_trail).grid(row=0, column=1, padx=2,
                                                  pady=2)
        self.btn_mask = ttk.Button(g, text="마스크 보기 (M)", width=17,
                                   command=self.toggle_mask)
        self.btn_mask.grid(row=1, column=0, padx=2, pady=2)
        self.btn_rec = ttk.Button(g, text="기록 시작 (R)", width=17,
                                  command=self.toggle_record)
        self.btn_rec.grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(g, text="검출값 저장", width=17,
                   command=self.save_detector).grid(row=2, column=0, padx=2,
                                                    pady=2)
        ttk.Button(g, text="기본값 복원", width=17,
                   command=self.reset_detector).grid(row=2, column=1, padx=2,
                                                     pady=2)
        self.btn_pick = ttk.Button(g, text="공 색 찍기 (P)", width=17,
                                   command=self.toggle_pick)
        self.btn_pick.grid(row=4, column=0, padx=2, pady=2)
        ttk.Button(g, text="주황 기본범위 복원", width=17,
                   command=self.reset_hsv).grid(row=4, column=1, padx=2,
                                                pady=2)
        ttk.Button(g, text="검출 진단", width=17,
                   command=self.diagnose_detection).grid(row=5, column=0,
                                                         padx=2, pady=2)
        self.btn_cov = ttk.Button(g, text="시야 커버리지 점검", width=17,
                                  command=self.check_coverage)
        self.btn_cov.grid(row=3, column=0, padx=2, pady=2)
        ttk.Button(g, text="커버리지 표시 끄기", width=17,
                   command=self.hide_coverage).grid(row=3, column=1, padx=2,
                                                    pady=2)

        f_hsv = self._section(right, "주황색 검출 범위")
        tr = tk.Frame(f_hsv, bg=PANEL); tr.pack(fill="x", pady=(0, 3))
        tk.Label(tr, text="적용 대상", bg=PANEL, fg=DIM_C,
                 font=FONT).pack(side="left")
        self.var_target = tk.IntVar(value=2)
        for lab, val in (("양쪽", 2), ("cam0", 0), ("cam1", 1)):
            tk.Radiobutton(tr, text=lab, variable=self.var_target, value=val,
                           command=self.on_target, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left", padx=2)
        mr = tk.Frame(f_hsv, bg=PANEL); mr.pack(fill="x", pady=(0, 3))
        tk.Label(mr, text="검출 방식", bg=PANEL, fg=DIM_C,
                 font=FONT).pack(side="left")
        self.var_mode = tk.StringVar(
            value=str(self.params[0].get("mode", "both")))
        for lab, val in (("색상", "color"), ("밝기", "bright"),
                         ("둘 다", "both")):
            tk.Radiobutton(mr, text=lab, variable=self.var_mode, value=val,
                           command=self.on_mode, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left", padx=2)

        self.sliders = {}
        specs = [("h_lo", "H 하한", 0, 179), ("h_hi", "H 상한", 0, 179),
                 ("s_lo", "S 하한", 0, 255), ("v_lo", "V 하한", 0, 255),
                 ("bright_v_lo", "밝기 V 하한", 100, 255),
                 ("bright_s_hi", "밝기 S 상한", 0, 200),
                 ("min_radius_px", "최소 반지름 px", 1, 30),
                 ("max_radius_px", "최대 반지름 px", 5, 80),
                 ("min_circularity", "최소 원형도", 0.0, 0.95),
                 ("border_band_px", "경계 밴드 px", 0, 80)]
        for key, label, lo, hi in specs:
            row = tk.Frame(f_hsv, bg=PANEL); row.pack(fill="x")
            tk.Label(row, text=label, bg=PANEL, fg=DIM_C, font=FONT,
                     width=14, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=float(self.params[0][key]))
            s = tk.Scale(row, from_=lo, to=hi, orient="horizontal",
                         variable=var, bg=PANEL, fg=FG, font=FONT_M,
                         highlightthickness=0, troughcolor="#15151a",
                         length=170, showvalue=True,
                         resolution=(0.05 if isinstance(lo, float) else 1),
                         command=lambda v, k=key: self.on_slider(k, v))
            s.pack(side="left")
            self.sliders[key] = var

        f_op = self._section(right, "옵션")
        self.var_bg = tk.BooleanVar(value=bool(self.det_params["use_bgsub"]))
        tk.Checkbutton(f_op, text="배경 차분 사용 (정지한 주황 물체 제거)",
                       variable=self.var_bg, command=self.on_opt, bg=PANEL,
                       fg=FG, selectcolor=BG, font=FONT,
                       activebackground=PANEL, activeforeground=FG,
                       anchor="w").pack(fill="x")
        self.var_async = tk.BooleanVar(
            value=bool(self.det_params["compensate_async"]))
        tk.Checkbutton(f_op, text="비동기 셔터 보정 (속도 기반)",
                       variable=self.var_async, command=self.on_opt, bg=PANEL,
                       fg=FG, selectcolor=BG, font=FONT,
                       activebackground=PANEL, activeforeground=FG,
                       anchor="w").pack(fill="x")
        self.var_roimask = tk.BooleanVar(
            value=bool(self.det_params.get("use_roi_mask", True)))
        tk.Checkbutton(f_op, text="탐지 영역을 ROI 안으로 제한 (바닥 오탐 차단)",
                       variable=self.var_roimask, command=self.on_opt,
                       bg=PANEL, fg=FG, selectcolor=BG, font=FONT,
                       activebackground=PANEL, activeforeground=FG,
                       anchor="w").pack(fill="x")
        rr = tk.Frame(f_op, bg=PANEL); rr.pack(fill="x")
        tk.Label(rr, text="탐지 영역 여유 px", bg=PANEL, fg=DIM_C, font=FONT,
                 width=14, anchor="w").pack(side="left")
        self.var_roimargin = tk.DoubleVar(
            value=float(self.det_params.get("roi_mask_margin_px", 20)))
        tk.Scale(rr, from_=0, to=80, orient="horizontal",
                 variable=self.var_roimargin, bg=PANEL, fg=FG, font=FONT_M,
                 highlightthickness=0, troughcolor="#15151a", length=170,
                 command=self.on_roi_margin).pack(side="left")
        self.var_showroi = tk.BooleanVar(value=True)
        tk.Checkbutton(f_op, text="카메라 화면에 탐지 영역 표시",
                       variable=self.var_showroi, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG, anchor="w").pack(fill="x")

        er = tk.Frame(f_op, bg=PANEL); er.pack(fill="x")
        self.var_autoexp = tk.BooleanVar(value=True)
        tk.Checkbutton(er, text="자동 노출", variable=self.var_autoexp,
                       command=self.on_exposure, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG).pack(side="left")
        self.var_exp = tk.DoubleVar(value=-6)
        tk.Scale(er, from_=-13, to=0, orient="horizontal",
                 variable=self.var_exp, bg=PANEL, fg=FG, font=FONT_M,
                 highlightthickness=0, troughcolor="#15151a", length=140,
                 command=lambda v: self.on_exposure()).pack(side="left")
        tk.Label(f_op, text="노출값이 작을수록 어둡고 블러가 줄어듭니다 "
                            "(과노출로 공이 하얗게 뜰 때 낮추세요)",
                 bg=PANEL, fg=DIM_C, font=FONT, wraplength=300,
                 justify="left", anchor="w").pack(fill="x")

        self.var_showcam = tk.BooleanVar(value=True)
        tk.Checkbutton(f_op, text="3D 뷰에 카메라 위치 표시",
                       variable=self.var_showcam, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG, anchor="w").pack(fill="x")

        f_info = self._section(right, "캘리브레이션")
        info = ("cam0  RMS %.3f px   위치 (%+.2f, %+.2f, %+.2f)\n"
                "cam1  RMS %.3f px   위치 (%+.2f, %+.2f, %+.2f)\n"
                "기선 %.3f m" %
                (self.calib[0].rms, *self.calib[0].C,
                 self.calib[1].rms, *self.calib[1].C,
                 float(np.linalg.norm(self.calib[0].C - self.calib[1].C))))
        tk.Label(f_info, text=info, bg=PANEL, fg=DIM_C, font=FONT_M,
                 justify="left", anchor="w").pack(fill="x")

        self.lbl_status = tk.Label(root, bg=BG, fg=DIM_C, font=FONT,
                                   anchor="w")
        self.lbl_status.grid(row=1, column=0, columnspan=3, sticky="ew",
                             pady=(6, 0))
        if self.args.sim:
            self._status("시뮬레이션 모드: 저장된 외부 파라미터로 합성 공을 "
                         "생성해 파이프라인을 점검합니다.")

    def _bind_events(self):
        self.canvas3d.bind("<Button-1>", self._m_down)
        self.canvas3d.bind("<B1-Motion>", self._m_orbit)
        self.canvas3d.bind("<Button-3>", self._m_down)
        self.canvas3d.bind("<B3-Motion>", self._m_pan)
        self.canvas3d.bind("<MouseWheel>",
                           lambda e: self.viewer.zoom(0.9 if e.delta > 0
                                                      else 1.1))
        self.canvas3d.bind("<Button-4>", lambda e: self.viewer.zoom(0.9))
        self.canvas3d.bind("<Button-5>", lambda e: self.viewer.zoom(1.1))
        self.bind_all("<space>", lambda e: self.toggle_pause())
        for k in "tT":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.clear_trail())
        for k in "mM":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.toggle_mask())
        for k in "rR":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.toggle_record())
        for k in "pP":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.toggle_pick())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _m_down(self, ev):
        self._mx, self._my = ev.x, ev.y

    def _m_orbit(self, ev):
        self.viewer.orbit(ev.x - self._mx, ev.y - self._my)
        self._mx, self._my = ev.x, ev.y

    def _m_pan(self, ev):
        self.viewer.pan(ev.x - self._mx, ev.y - self._my)
        self._mx, self._my = ev.x, ev.y

    # --------------------------------------------------------------- 조작
    def _targets(self):
        t = self.slider_target
        return (0, 1) if t == 2 else (t,)

    def on_mode(self):
        m = self.var_mode.get()
        for i in self._targets():
            self.params[i]["mode"] = m
            self.detector[i].set("mode", m)
        self._status("검출 방식: %s"
                     % {"color": "색상", "bright": "밝기",
                        "both": "둘 다"}[m])

    def on_target(self):
        self.slider_target = int(self.var_target.get())
        src = self.params[0 if self.slider_target == 2
                          else self.slider_target]
        for k, var in self.sliders.items():
            var.set(float(src[k]))
        self.var_mode.set(str(src.get("mode", "both")))
        self._status("슬라이더 적용 대상: %s"
                     % ("양쪽" if self.slider_target == 2
                        else "cam%d" % self.slider_target))

    def on_slider(self, key, val):
        v = float(val)
        for i in self._targets():
            self.params[i][key] = v
            self.detector[i].set(key, v)

    def on_opt(self):
        for i in (0, 1):
            self.params[i]["use_bgsub"] = bool(self.var_bg.get())
            self.params[i]["compensate_async"] = bool(self.var_async.get())
            self.params[i]["use_roi_mask"] = bool(self.var_roimask.get())
            self.detector[i].set("use_bgsub", bool(self.var_bg.get()))

    def on_exposure(self):
        req = "auto" if self.var_autoexp.get() else float(self.var_exp.get())
        for c in self.cams:
            if hasattr(c, "exposure_req"):
                c.exposure_req = req
        self._status("노출: %s" % ("자동" if req == "auto"
                                 else "수동 %g" % req))

    def on_roi_margin(self, val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return
        if abs(v - float(self.params[0].get("roi_mask_margin_px", 20))) < 0.5:
            return
        for i in (0, 1):
            self.params[i]["roi_mask_margin_px"] = v
        self._rebuild_roi_masks()

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="재개 (Space)" if self.paused
                              else "일시정지 (Space)")

    def clear_trail(self):
        self.trail.clear()
        self.kf.reset()
        self.stat_n = self.stat_hit = 0

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.btn_mask.config(text="영상 보기 (M)" if self.show_mask
                             else "마스크 보기 (M)")

    def toggle_record(self):
        if not self.recording:
            self.rec_rows = []
            self.recording = True
            self.btn_rec.config(text="기록 정지 (R)")
            self._status("기록 시작.")
        else:
            self.recording = False
            self.btn_rec.config(text="기록 시작 (R)")
            if self.rec_rows:
                os.makedirs(RECORD_DIR, exist_ok=True)
                p = os.path.join(RECORD_DIR, "track_%s.csv"
                                 % datetime.now().strftime("%Y%m%d_%H%M%S"))
                with open(p, "w", newline="", encoding="utf-8") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(["t_s", "x_m", "y_m", "z_m",
                                  "vx", "vy", "vz",
                                  "raw_x", "raw_y", "raw_z",
                                  "reproj0_px", "reproj1_px", "pair_dt_ms"])
                    wtr.writerows(self.rec_rows)
                self._status("기록 저장: %s  (%d 샘플)" % (p, len(self.rec_rows)))
            else:
                self._status("기록된 샘플이 없습니다.")

    def save_detector(self):
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        p = os.path.join(SETTINGS_DIR, "detector.json")
        data = {"_comment": DEFAULT_DETECTOR["_comment"] +
                " cam0/cam1 을 따로 둘 수 있습니다.",
                "cam0": self.params[0], "cam1": self.params[1]}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._status("검출 파라미터 저장 (카메라별): %s" % p)

    def reset_detector(self):
        self.params = [dict(DEFAULT_DETECTOR), dict(DEFAULT_DETECTOR)]
        self.det_params = self.params[0]
        for i in (0, 1):
            self.detector[i].p = dict(DEFAULT_DETECTOR)
        for k, var in self.sliders.items():
            var.set(float(self.det_params[k]))
        self.var_bg.set(bool(self.det_params["use_bgsub"]))
        self.var_async.set(bool(self.det_params["compensate_async"]))
        self.var_roimask.set(bool(self.det_params["use_roi_mask"]))
        self.var_roimargin.set(float(self.det_params["roi_mask_margin_px"]))
        self._rebuild_roi_masks()
        self._status("기본값으로 복원했습니다 (양쪽 카메라).")

    def diagnose_detection(self):
        lines = []
        for i in (0, 1):
            st = self.detector[i].stats
            pm = self.params[i]
            if not st:
                lines.append("cam%d: 아직 프레임을 처리하지 않았습니다." % i)
                continue
            lines.append(
                "cam%d  H %d~%d  S %d+  V %d+  (%s)\n"
                "  마스크 픽셀 %d개 / 덩어리 %d개\n"
                "  탈락  면적 %d · 반지름 %d · 원형도 %d · 경계밴드 %d\n"
                "  통과 %d개\n"
                "  가장 큰 덩어리: 면적 %.0f px², 반지름 %.1f px, 원형도 %.2f"
                % (i, pm["h_lo"], pm["h_hi"], pm["s_lo"], pm["v_lo"],
                   {"color": "색상", "bright": "밝기",
                    "both": "둘 다"}.get(pm.get("mode", "color"), "?"),
                   st["mask_px"], st["blobs"], st["area"], st["radius"],
                   st["circ"], st["border"], st["passed"],
                   st["best_area"], st["best_r"], st["best_circ"]))
        tips = []
        for i in (0, 1):
            st = self.detector[i].stats
            if not st:
                continue
            pm = self.params[i]
            if st["mask_px"] == 0:
                tips.append("cam%d: 색 마스크가 완전히 비었습니다. "
                            "[공 색 찍기] 로 이 카메라의 색을 실측하세요." % i)
            elif st["blobs"] and st["passed"] == 0:
                worst = max(("면적", st["area"]), ("반지름", st["radius"]),
                            ("원형도", st["circ"]), ("경계밴드", st["border"]),
                            key=lambda kv: kv[1])[0]
                if worst == "경계밴드":
                    tips.append("cam%d: 후보가 ROI 경계 밴드 안에 있어 "
                                "엄격한 조건에 걸렸습니다. detector.json 의 "
                                "border_band_px 를 줄이거나 "
                                "border_min_* 를 낮추세요." % i)
                elif worst == "반지름":
                    tips.append("cam%d: 반지름 조건에서 탈락했습니다. "
                                "가장 큰 덩어리가 %.1f px 입니다."
                                % (i, st["best_r"]))
                elif worst == "면적":
                    tips.append("cam%d: 면적 조건에서 탈락했습니다 "
                                "(가장 큰 덩어리 %.0f px²)."
                                % (i, st["best_area"]))
                else:
                    tips.append("cam%d: 원형도에서 탈락했습니다 (%.2f). "
                                "모션 블러라면 최소 원형도를 0 으로 두세요."
                                % (i, st["best_circ"]))
            elif st["mask_px"] > 20000:
                tips.append("cam%d: 색 마스크가 너무 넓습니다(%d px). "
                            "S 하한을 올리세요." % (i, st["mask_px"]))
        # 두 카메라 색이 많이 다르면 알림
        if abs(self.params[0]["h_lo"] - self.params[1]["h_lo"]) < 1 and \
                self.detector[0].stats and self.detector[1].stats:
            a = self.detector[0].stats["mask_px"]
            b = self.detector[1].stats["mask_px"]
            if a > 0 and b == 0:
                tips.append("두 카메라가 같은 색 범위를 쓰는데 cam1 에서만 "
                            "검출이 안 됩니다. 화이트밸런스 차이일 가능성이 "
                            "큽니다. [적용 대상]을 cam1 로 두고 "
                            "[공 색 찍기] 하세요.")
        msg = "\n\n".join(lines)
        if tips:
            msg += "\n\n" + "\n".join("! " + t for t in tips)
        else:
            msg += "\n\n[OK] 뚜렷한 문제는 보이지 않습니다."
        messagebox.showinfo("검출 진단", msg)

    def toggle_pick(self):
        self.pick_mode = not self.pick_mode
        self.btn_pick.config(text="색 찍기 취소" if self.pick_mode
                             else "공 색 찍기 (P)")
        self._status("카메라 화면에서 공 위를 클릭하세요."
                     if self.pick_mode else "색 찍기 취소.")

    def reset_hsv(self):
        for k in ("h_lo", "h_hi", "s_lo", "v_lo"):
            for i in self._targets():
                self.params[i][k] = DEFAULT_DETECTOR[k]
                self.detector[i].set(k, DEFAULT_DETECTOR[k])
            if k in self.sliders:
                self.sliders[k].set(float(DEFAULT_DETECTOR[k]))
        self._status("주황 검출 범위를 기본값으로 되돌렸습니다.")

    def on_pick(self, cam, ev):
        if not self.pick_mode:
            return
        frame = self._last_frames[cam]
        if frame is None:
            return
        x = int(ev.x / self.preview_scale)
        y = int(ev.y / self.preview_scale)
        h, w = frame.shape[:2]
        r = 5
        x0, y0 = max(x - r, 0), max(y - r, 0)
        x1, y1 = min(x + r + 1, w), min(y + r + 1, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        px = patch.reshape(-1, 3).astype(np.int32)
        # 가장 채도 높은 60% 만 사용 (테두리 배경 섞임 방지)
        thr = np.percentile(px[:, 1], 40)
        core = px[px[:, 1] >= thr]
        if len(core) < 5:
            core = px
        hs = core[:, 0]
        # 색상 환형 평균
        ang = hs * (2 * np.pi / 180.0)
        hm = (np.degrees(np.arctan2(np.sin(ang).mean(),
                                    np.cos(ang).mean())) / 2.0) % 180
        spread = np.abs((hs - hm + 90) % 180 - 90)
        hw_ = float(min(max(np.percentile(spread, 90), 5.0) + 3.0, 25.0))
        h_lo = int(round((hm - hw_) % 180))
        h_hi = int(round((hm + hw_) % 180))
        s_lo = int(max(30, np.percentile(core[:, 1], 5) * 0.75))
        v_lo = int(max(30, np.percentile(core[:, 2], 5) * 0.65))
        msg = ("cam%d (%d, %d) 에서 측정한 값\n\n"
               "색상 H  중앙 %.1f   폭 ±%.1f\n"
               "채도 S  중앙 %d\n"
               "명도 V  중앙 %d\n\n"
               "제안 범위\n"
               "  H %d ~ %d\n  S %d 이상\n  V %d 이상\n\n"
               "적용할까요?"
               % (cam, x, y, hm, hw_, int(np.median(core[:, 1])),
                  int(np.median(core[:, 2])), h_lo, h_hi, s_lo, v_lo))
        self.pick_mode = False
        self.btn_pick.config(text="공 색 찍기 (P)")
        s_med = float(np.median(core[:, 1]))
        v_med = float(np.median(core[:, 2]))
        bad = []
        if np.percentile(spread, 90) > 20:
            bad.append("색이 한 가지로 모이지 않습니다 (색상 폭 ±%.0f)"
                       % np.percentile(spread, 90))
        if s_med < 60:
            bad.append("채도가 너무 낮습니다 (%d). 주황 공이라면 보통 "
                       "150 이상입니다" % int(s_med))
        if v_med < 50:
            bad.append("너무 어둡습니다 (명도 %d)" % int(v_med))
        if bad:
            if not messagebox.askyesno(
                    "공 색 찍기 — 확인 필요",
                    "공이 아니라 배경을 클릭했을 가능성이 큽니다.\n\n"
                    + "\n".join("! " + b for b in bad)
                    + "\n\n영상을 [정지] 시키고 공 한가운데를 다시 "
                      "클릭하시길 권합니다.\n\n그래도 적용할까요?\n\n" + msg):
                return
        elif not messagebox.askyesno("공 색 찍기", msg):
            return
        tgt = (0, 1) if self.slider_target == 2 else (cam,)
        for k, v in (("h_lo", h_lo), ("h_hi", h_hi),
                     ("s_lo", s_lo), ("v_lo", v_lo)):
            for i in tgt:
                self.params[i][k] = v
                self.detector[i].set(k, v)
            if k in self.sliders and (self.slider_target == 2
                                      or self.slider_target == cam):
                self.sliders[k].set(float(v))
        self._status("검출 범위 적용(%s): H %d~%d, S %d+, V %d+  "
                     "— [마스크 보기] 로 확인 후 [검출값 저장]"
                     % ("양쪽" if len(tgt) == 2 else "cam%d" % cam,
                        h_lo, h_hi, s_lo, v_lo))

    def check_coverage(self):
        grid, vis, per = roi_coverage(self.calib, self.geo)
        self.coverage = (grid, vis)
        self.show_coverage = True
        tot = len(grid)
        msg = ("ROI 격자 %d점 기준\n\n"
               "두 카메라 동시 가시 : %.1f %%\n"
               "cam0 단독 가시      : %.1f %%\n"
               "cam1 단독 가시      : %.1f %%\n"
               "어느 쪽에도 안 보임 : %.1f %%"
               % (tot, 100.0 * vis.sum() / tot,
                  100.0 * per[0].sum() / tot, 100.0 * per[1].sum() / tot,
                  100.0 * (~(per[0] | per[1])).sum() / tot))
        pct = 100.0 * vis.sum() / tot
        if pct < 90:
            miss = grid[~vis]
            msg += ("\n\n! 사각지대가 %.1f %% 입니다.\n"
                    "  주로 Y %.2f~%.2f m, Z %.2f~%.2f m 구간입니다.\n"
                    "  카메라를 뒤로 물리거나 더 높이면 개선됩니다."
                    % (100 - pct, miss[:, 1].min(), miss[:, 1].max(),
                       miss[:, 2].min(), miss[:, 2].max()))
        else:
            msg += "\n\n[OK] 커버리지 양호."
        messagebox.showinfo("시야 커버리지", msg)
        self._status("3D 뷰에 커버리지 표시: 초록=양쪽 가시, 빨강=사각지대")

    def hide_coverage(self):
        self.show_coverage = False

    def _status(self, m):
        self.lbl_status.config(text=m)

    # --------------------------------------------------------------- 루프
    def _pair_frames(self):
        """두 카메라 버퍼에서 시간차가 가장 작은 조합 선택."""
        b0, b1 = self.cams[0].snapshot(), self.cams[1].snapshot()
        if not b0 or not b1:
            return None
        best = None
        for t0, f0 in b0[-4:]:
            for t1, f1 in b1[-4:]:
                d = abs(t0 - t1)
                if best is None or d < best[0]:
                    best = (d, t0, f0, t1, f1)
        return best

    def _tick(self):
        if self._closing:
            return
        t_start = time.perf_counter()
        try:
            self._process()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status("오류: %s" % e)
        dt = time.perf_counter() - t_start
        delay = max(5, int(20 - dt * 1000))
        self._after_id = self.after(delay, self._tick)

    def _process(self):
        for i, c in enumerate(self.cams):
            if getattr(c, "error", None):
                self._status(c.error)
                return
        pair = self._pair_frames()
        if pair is None:
            return
        pdt, t0, f0, t1, f1 = pair
        if self.paused:
            self._draw(f0, f1, None, None, None, pdt)
            return

        frames = (f0, f1)
        cands, masks = [], []
        use_roi = bool(self.params[0].get("use_roi_mask", True))
        for i in (0, 1):
            c, m = self.detector[i].detect(
                frames[i], return_mask=self.show_mask,
                roi_mask=(self.roi_mask[i] if use_roi else None),
                roi_poly=(self.roi_poly[i] if use_roi else None))
            cands.append(c)
            masks.append(m)

        self.stat_n += 1
        result = None
        if cands[0] and cands[1] and pdt * 1000 <= self.det_params["max_pair_dt_ms"]:
            vel = self.kf.vel if self.det_params["compensate_async"] else None
            pred = self.kf.predict_to(t0) if self.kf.x is not None else None
            result = match_candidates(self.calib[0], self.calib[1],
                                      cands[0], cands[1], self.det_params,
                                      self.geo, vel, (t1 - t0), pred)
        raw = None
        tracked = False
        if result is not None:
            raw = result["X"]
            tracked = self.kf.update(raw, t0)
            self.stat_hit += 1
            self.last = dict(t=t0, raw=raw, e0=result["e0"], e1=result["e1"],
                             pdt=pdt, uv0=result["uv0"], uv1=result["uv1"],
                             r0=result["r0"], r1=result["r1"],
                             r0e=result["r0e"], r1e=result["r1e"])
        else:
            self.kf.miss += 1
            if self.kf.miss > 25:
                self.kf.reset()

        pos = self.kf.pos if tracked else None
        # ROI 를 크게 벗어난 추정치는 발산으로 보고 초기화
        if pos is not None:
            r = self.geo["roi"]
            if not (r["x_min"] - 0.6 <= pos[0] <= r["x_max"] + 0.6 and
                    r["y_min"] - 0.6 <= pos[1] <= r["y_max"] + 0.6 and
                    r["z_min"] - 0.6 <= pos[2] <= r["z_max"] + 0.6):
                self.kf.reset()
                pos = None
        if pos is not None and result is not None:
            self.trail.append((t0, pos.copy()))
            if self.recording:
                v = self.kf.vel
                self.rec_rows.append([
                    round(t0, 6), *np.round(pos, 5), *np.round(v, 4),
                    *np.round(raw, 5), round(result["e0"], 3),
                    round(result["e1"], 3), round(pdt * 1000, 3)])

        self._draw(f0, f1, masks, result, pos, pdt)

    # --------------------------------------------------------------- 그리기
    def _draw(self, f0, f1, masks, result, pos, pdt):
        frames = (f0, f1)
        self._last_frames = [f0, f1]
        for i in (0, 1):
            if self.show_mask and masks and masks[i] is not None:
                vis = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)
            else:
                vis = frames[i].copy()
            if self.var_showroi.get() and self.roi_poly[i] is not None:
                poly = self.roi_poly[i].reshape(-1, 1, 2).astype(np.int32)
                ov = vis.copy()
                cv2.fillPoly(ov, [poly], (60, 45, 20))
                cv2.addWeighted(ov, 0.28, vis, 0.72, 0, vis)
                cv2.polylines(vis, [poly], True, (255, 190, 80), 2,
                              cv2.LINE_AA)
                cv2.putText(vis, "detect zone", (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 80), 1,
                            cv2.LINE_AA)
            if result is not None:
                uv = result["uv0"] if i == 0 else result["uv1"]
                rr = result["r0"] if i == 0 else result["r1"]
                p = (int(round(uv[0])), int(round(uv[1])))
                cv2.circle(vis, p, max(int(rr) + 3, 5), (60, 230, 120), 1,
                           cv2.LINE_AA)
                cv2.drawMarker(vis, p, (60, 230, 120), cv2.MARKER_CROSS, 12, 1)
            if pos is not None:
                rp = self.calib[i].project(pos.reshape(1, 3))[0]
                if np.all(np.isfinite(rp)) and abs(rp).max() < 1e4:
                    q = (int(round(rp[0])), int(round(rp[1])))
                    cv2.drawMarker(vis, q, (60, 200, 255),
                                   cv2.MARKER_TILTED_CROSS, 14, 1)
            disp = cv2.resize(vis, (int(self.calib[i].w * self.preview_scale),
                                    int(self.calib[i].h * self.preview_scale)),
                              interpolation=cv2.INTER_AREA)
            self._photo[i] = to_photo(disp)
            self.cam_canvas[i].delete("all")
            self.cam_canvas[i].create_image(0, 0, image=self._photo[i],
                                            anchor="nw")
            if self.pick_mode:
                cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1,
                                            vis.shape[0] - 1),
                              (60, 220, 255), 3)
                cv2.putText(vis, "click the ball", (6, vis.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 255), 1,
                            cv2.LINE_AA)
            self.cam_label[i].config(
                text="%.0f fps   검출 %s"
                     % (getattr(self.cams[i], "fps_measured", 0.0),
                        "O" if result is not None else "-"))

        img3d = self.viewer.render(
            self.geo, ball=pos, trail=list(self.trail), cams=self.calib,
            raw=(result["X"] if result is not None else None),
            show_cams=self.var_showcam.get(),
            coverage=(self.coverage if self.show_coverage else None))
        self._photo3d = to_photo(img3d)
        self.canvas3d.delete("all")
        self.canvas3d.create_image(0, 0, image=self._photo3d, anchor="nw")

        if pos is not None:
            v = self.kf.vel
            sp = float(np.linalg.norm(v))
            r = self.geo["roi"]
            inside = (r["x_min"] <= pos[0] <= r["x_max"] and
                      r["y_min"] <= pos[1] <= r["y_max"] and
                      r["z_min"] <= pos[2] <= r["z_max"])
            self.lbl_pos.config(
                text="X %+7.3f m\nY %+7.3f m\nZ %+7.3f m\n속도 %5.2f m/s"
                     % (pos[0], pos[1], pos[2], sp),
                fg=OK_C if inside else WARN_C)
        else:
            self.lbl_pos.config(text="공 미검출", fg=DIM_C)

        rate = 100.0 * self.stat_hit / max(self.stat_n, 1)
        if result is not None:
            self.lbl_det.config(
                text=("재투영 오차  cam0 %.2f px  cam1 %.2f px\n"
                      "반지름 실측/예상  %.1f/%.1f  %.1f/%.1f px\n"
                      "프레임 시간차 %.2f ms\n검출률 %.1f %%  (%d/%d)"
                      % (result["e0"], result["e1"],
                         result["r0"], result["r0e"],
                         result["r1"], result["r1e"],
                         pdt * 1000, rate, self.stat_hit, self.stat_n)),
                fg=FG)
        else:
            self.lbl_det.config(
                text="프레임 시간차 %.2f ms\n검출률 %.1f %%  (%d/%d)"
                     % (pdt * 1000, rate, self.stat_hit, self.stat_n),
                fg=DIM_C)

    def on_close(self):
        self._closing = True
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        for c in self.cams:
            try:
                c.stop()
            except Exception:
                pass
        self.destroy()


def to_photo(bgr):
    """numpy BGR -> tk.PhotoImage.

    Pillow 가 있으면 그것을 쓰고, 없으면 임시 PPM 파일을 거친다.
    (tk.PhotoImage 는 base64 PPM 데이터를 인식하지 못한다 ->
     'couldn't recognize image data' 오류의 원인)
    """
    if HAS_PIL:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))
    path = _ppm_path()
    if not cv2.imwrite(path, bgr):
        raise RuntimeError("임시 이미지 파일 기록 실패: %s" % path)
    return tk.PhotoImage(file=path)


_PPM_COUNTER = [0]
_PPM_DIR = None


def _ppm_path():
    global _PPM_DIR
    if _PPM_DIR is None:
        import tempfile
        _PPM_DIR = tempfile.mkdtemp(prefix="ttrobot_")
    _PPM_COUNTER[0] = (_PPM_COUNTER[0] + 1) % 4
    return os.path.join(_PPM_DIR, "feed%d.ppm" % _PPM_COUNTER[0])
# ============================================================================
# 엔트리 포인트
# ============================================================================

def print_env_banner():
    """실행 환경 진단 정보. Pillow 문제 추적용."""
    print("-" * 60)
    print(" Python      : %s" % sys.version.split()[0])
    print(" 실행 파일   : %s" % sys.executable)
    print(" OpenCV      : %s" % cv2.__version__)
    print(" Tk          : %s" % tk.TkVersion)
    if HAS_PIL:
        try:
            import PIL
            print(" Pillow      : %s (OK)" % PIL.__version__)
        except Exception:
            print(" Pillow      : OK")
    else:
        print(" Pillow      : 사용 불가 -> %s" % PIL_ERROR)
        print("   같은 인터프리터에 설치하려면:")
        print("     \"%s\" -m pip install pillow" % sys.executable)
        print("   (임시 파일 방식으로 계속 진행합니다. 동작에는 문제 없음)")
    print("-" * 60)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="스테레오 탁구공 트래킹 + 3D 시각화")
    p.add_argument("--cam0", type=int, default=None,
                   help="cam0 장치 index (기본: extrinsics 에 기록된 값)")
    p.add_argument("--cam1", type=int, default=None)
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--backend", default="dshow",
                   choices=list(BACKENDS.keys()))
    p.add_argument("--preview-scale", type=float, default=0.7)
    p.add_argument("--sim", action="store_true",
                   help="카메라 없이 합성 공으로 파이프라인 점검")
    p.add_argument("--sim-offset-ms", type=float, default=3.0,
                   help="시뮬레이션에서 두 카메라의 셔터 시간차 (ms)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print_env_banner()
    try:
        app = TrackerApp(args)
    except FileNotFoundError as e:
        print("[!] 설정 파일이 없습니다:\n%s" % e)
        try:
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("설정 없음",
                                 "%s\n\nsetup_extrinsics.py 를 먼저 실행하세요."
                                 % e)
            r.destroy()
        except Exception:
            pass
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())