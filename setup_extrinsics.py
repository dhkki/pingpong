#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_extrinsics.py
===================
탁구 로봇 프로젝트 - 카메라 외부 파라미터(extrinsics) 설정 GUI 도구.

카메라 위치가 고정되어 있지 않은 환경을 전제로, 화면에서 탁구대의 기지점을
직접 클릭해 각 카메라의 월드 좌표계 자세(R, t)를 즉시 산출한다.
스테레오 캘리브레이션 없이 두 카메라가 동일한 월드 좌표계를 공유하게 된다.

폴더 구조
---------
  <프로젝트>/
    setup_extrinsics.py
    settings/
      intrinsics_cam0.json      (입력, 필수)
      intrinsics_cam1.json      (입력, 필수)
      table_geometry.json       (없으면 자동 생성)
      extrinsics_cam0.json      (출력)
      extrinsics_cam1.json      (출력)

월드 좌표계
-----------
  원점 O : 탁구대 상면, 로봇측 엔드라인의 중앙
  +X     : 로봇에서 사람을 바라볼 때 오른쪽
  +Y     : 사람 방향
  +Z     : 위쪽
  (오른손 좌표계, 단위 m)

조작
----
  마우스 클릭 : 임시점 지정 (지정 즉시 영상이 자동 정지됨)
  방향키      : 임시점 1 px 이동   /  Shift+방향키 : 0.2 px 미세 이동
  Z / X       : 돋보기 확대 / 축소
  Enter, Space: 현재 임시점을 확정하고 다음 지점으로
  Backspace   : 이전 지점으로 되돌아가기
  F           : 영상 정지 / 재생
  C           : 자세 계산      S : 저장      O : 오버레이 토글
  모든 기능은 우측 버튼으로도 동일하게 수행 가능

실행
----
  python setup_extrinsics.py
  python setup_extrinsics.py --cam0 0 --cam1 1 --scale 1.5
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
    PIL_ERROR = None
except Exception as _e:
    HAS_PIL = False
    PIL_ERROR = "%s: %s" % (type(_e).__name__, _e)


# ============================================================================
# 경로 / 기본 설정
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_DIR = os.path.join(SCRIPT_DIR, "settings")

DEFAULT_GEOMETRY = {
    "_comment": "단위 m. 월드 원점 = 탁구대 상면, 로봇측 엔드라인 중앙. "
                "+X 오른쪽(로봇→사람 기준), +Y 사람 방향, +Z 위",
    "table_length": 2.74,
    "table_width": 1.525,
    "net_height": 0.1525,
    "net_overhang": 0.1525,
    "roi": {
        "x_min": -0.38125, "x_max": 0.38125,
        "y_min": 0.0, "y_max": 1.60,
        "z_min": 0.0, "z_max": 1.00
    }
}

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


# ============================================================================
# 기하 / 기지점 정의
# ============================================================================

def load_geometry():
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    path = os.path.join(SETTINGS_DIR, "table_geometry.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)
        for k, v in DEFAULT_GEOMETRY.items():
            g.setdefault(k, v)
        return g, path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_GEOMETRY, f, indent=2, ensure_ascii=False)
    return dict(DEFAULT_GEOMETRY), path


def build_presets(geo):
    """기하 정보로부터 기지점 프리셋을 생성한다."""
    L = float(geo["table_length"])
    hw = float(geo["table_width"]) / 2.0
    nh = float(geo["net_height"])
    no = float(geo["net_overhang"])
    ny = L / 2.0
    nx = hw + no
    r = geo["roi"]
    rx0, rx1 = float(r["x_min"]), float(r["x_max"])
    ry0, ry1 = float(r["y_min"]), float(r["y_max"])

    def P(key, name, hint, world, required=False):
        return {"key": key, "name": name, "hint": hint,
                "world": [float(v) for v in world], "required": bool(required)}

    roi_pts = [
        P("R_NEAR_L", "마스킹 · 근거리 좌측 코너",
          "로봇측 엔드라인 위, 테이프 왼쪽 끝", (rx0, ry0, 0.0)),
        P("R_NEAR_R", "마스킹 · 근거리 우측 코너",
          "로봇측 엔드라인 위, 테이프 오른쪽 끝", (rx1, ry0, 0.0)),
        P("R_NEAR_C", "마스킹 근거리변 × 센터라인",
          "★ 월드 좌표계 원점", (0.0, ry0, 0.0)),
        P("R_NET_L", "마스킹 좌측 라인 × 네트",
          "왼쪽 테이프가 네트 밑동과 만나는 점", (rx0, ny, 0.0)),
        P("R_NET_R", "마스킹 우측 라인 × 네트",
          "오른쪽 테이프가 네트 밑동과 만나는 점", (rx1, ny, 0.0)),
        P("R_NET_C", "센터라인 × 네트 밑동",
          "네트 아래 중앙", (0.0, ny, 0.0)),
        P("NETTOP_ML", "네트 상단 × 좌측 테이프 연장선",
          "★ 높이 기준점", (rx0, ny, nh), True),
        P("NETTOP_MR", "네트 상단 × 우측 테이프 연장선",
          "★ 높이 기준점", (rx1, ny, nh), True),
        P("NETTOP_C", "네트 상단 중앙",
          "★ 높이 기준점 (센터라인 바로 위)", (0.0, ny, nh), True),
        P("R_FAR_L", "마스킹 · 원거리 좌측 코너",
          "네트 너머 0.23 m, 보이면 클릭", (rx0, ry1, 0.0)),
        P("R_FAR_R", "마스킹 · 원거리 우측 코너",
          "네트 너머 0.23 m, 보이면 클릭", (rx1, ry1, 0.0)),
        P("R_FAR_C", "마스킹 원거리변 × 센터라인",
          "네트 너머 0.23 m, 보이면 클릭", (0.0, ry1, 0.0)),
        P("NETTOP_L", "네트 상단 · 좌측 끝",
          "네트 기둥. 화각에 들어오면 클릭", (-nx, ny, nh)),
        P("NETTOP_R", "네트 상단 · 우측 끝",
          "네트 기둥. 화각에 들어오면 클릭", (nx, ny, nh)),
    ]

    table_pts = [
        P("ROBOT_L", "로봇측 엔드라인 · 좌측 코너",
          "로봇 뒤에서 볼 때 왼쪽 끝", (-hw, 0.0, 0.0)),
        P("ROBOT_R", "로봇측 엔드라인 · 우측 코너",
          "로봇 뒤에서 볼 때 오른쪽 끝", (hw, 0.0, 0.0)),
        P("ROBOT_C", "로봇측 엔드라인 × 센터라인",
          "★ 월드 좌표계 원점", (0.0, 0.0, 0.0)),
        P("NET_L", "좌측 사이드라인 × 네트",
          "왼쪽 사이드라인이 네트와 만나는 점", (-hw, ny, 0.0)),
        P("NET_R", "우측 사이드라인 × 네트",
          "오른쪽 사이드라인이 네트와 만나는 점", (hw, ny, 0.0)),
        P("NETTOP_L", "네트 상단 · 좌측 끝",
          "★ 높이 기준점", (-nx, ny, nh), True),
        P("NETTOP_R", "네트 상단 · 우측 끝",
          "★ 높이 기준점", (nx, ny, nh), True),
        P("HUMAN_L", "사람측 엔드라인 · 좌측 코너",
          "먼 쪽 왼쪽 끝", (-hw, L, 0.0)),
        P("HUMAN_R", "사람측 엔드라인 · 우측 코너",
          "먼 쪽 오른쪽 끝", (hw, L, 0.0)),
        P("HUMAN_C", "사람측 엔드라인 × 센터라인",
          "먼 쪽 중앙", (0.0, L, 0.0)),
    ]

    wide_pts = list(roi_pts)
    extra = [
        P("ROBOT_L", "탁구대 로봇측 · 좌측 코너",
          "★ 넓은 기준. 화면에 보이면 꼭 클릭", (-hw, 0.0, 0.0)),
        P("ROBOT_R", "탁구대 로봇측 · 우측 코너",
          "★ 넓은 기준. 화면에 보이면 꼭 클릭", (hw, 0.0, 0.0)),
        P("NET_L", "좌측 사이드라인 × 네트",
          "탁구대 왼쪽 끝이 네트와 만나는 점", (-hw, ny, 0.0)),
        P("NET_R", "우측 사이드라인 × 네트",
          "탁구대 오른쪽 끝이 네트와 만나는 점", (hw, ny, 0.0)),
    ]
    # 로봇측 코너는 앞쪽(3번 뒤)에, 네트측은 네트 그룹 뒤에 끼워 넣는다
    wide_pts = roi_pts[:3] + extra[:2] + roi_pts[3:6] + extra[2:] + roi_pts[6:]

    return {
        "_comment": "클릭 기지점 프리셋. world 는 [X, Y, Z] (m). "
                    "직접 편집 가능하며, table_geometry.json 을 고친 뒤에는 "
                    "GUI의 [프리셋 재생성] 을 누르면 이 파일이 갱신됩니다.",
        "default": "roi",
        "presets": {
            "roi": {"label": "마스킹 영역 기준 (권장)", "points": roi_pts},
            "roi_wide": {"label": "마스킹 + 탁구대 코너 (정밀)",
                         "points": wide_pts},
            "table": {"label": "탁구대 전체 기준", "points": table_pts},
        },
    }


def load_reference_points(geo, regenerate=False):
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    path = os.path.join(SETTINGS_DIR, "reference_points.json")
    if regenerate or not os.path.exists(path):
        data = build_presets(geo)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data, path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "presets" not in data or not data["presets"]:
        data = build_presets(geo)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data, path
    # 프로그램이 업데이트되어 새 프리셋이 생겼으면 자동으로 채워 넣는다.
    # (이미 있는 프리셋은 사용자가 편집했을 수 있으므로 건드리지 않는다)
    fresh = build_presets(geo)
    added = [k for k in fresh["presets"] if k not in data["presets"]]
    if added:
        for k in added:
            data["presets"][k] = fresh["presets"][k]
        data.setdefault("default", fresh.get("default", "roi"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return data, path


def preset_tuples(preset):
    """JSON 프리셋 -> [(key, name, hint, (x,y,z), required), ...]"""
    out = []
    for d in preset["points"]:
        w = tuple(float(v) for v in d["world"])
        out.append((d["key"], d.get("name", d["key"]), d.get("hint", ""),
                    w, bool(d.get("required", False))))
    return out


def roi_wireframe(geo):
    """기준점이 있는 영역만. 검증은 이걸로 하는 것이 정확하다."""
    L = float(geo["table_length"])
    hw = float(geo["table_width"]) / 2
    nh = float(geo["net_height"])
    nx = hw + float(geo["net_overhang"])
    ny = L / 2
    r = geo["roi"]
    C_TAPE = (70, 205, 240)
    C_NET = (255, 255, 255)
    C_ROI = (90, 90, 255)
    seg = []
    x0, x1 = r["x_min"], r["x_max"]
    y0, y1 = r["y_min"], r["y_max"]
    z1 = r["z_max"]
    tape = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
    for i in range(4):
        seg.append((tape[i], tape[(i + 1) % 4], C_TAPE, 2))
    top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for i in range(4):
        seg.append((top[i], top[(i + 1) % 4], C_ROI, 1))
        seg.append((tape[i], top[i], C_ROI, 1))
    seg.append(((-nx, ny, 0), (nx, ny, 0), C_NET, 1))
    seg.append(((-nx, ny, nh), (nx, ny, nh), C_NET, 2))
    seg.append(((-nx, ny, 0), (-nx, ny, nh), C_NET, 2))
    seg.append(((nx, ny, 0), (nx, ny, nh), C_NET, 2))
    return seg


def table_wireframe(geo):
    """검증 오버레이용 3D 선분 목록. [(P0, P1, color, thickness), ...]"""
    L = float(geo["table_length"])
    hw = float(geo["table_width"]) / 2.0
    nh = float(geo["net_height"])
    nx = hw + float(geo["net_overhang"])
    ny = L / 2.0
    r = geo["roi"]
    seg = []
    C_TABLE = (255, 200, 90)
    C_NET = (255, 255, 255)
    C_ROI = (90, 90, 255)

    # 탁구대 외곽
    corners = [(-hw, 0, 0), (hw, 0, 0), (hw, L, 0), (-hw, L, 0)]
    for i in range(4):
        seg.append((corners[i], corners[(i + 1) % 4], C_TABLE, 2))
    # 센터라인
    seg.append(((0, 0, 0), (0, L, 0), C_TABLE, 1))
    # 네트
    seg.append(((-nx, ny, 0), (nx, ny, 0), C_NET, 1))
    seg.append(((-nx, ny, nh), (nx, ny, nh), C_NET, 2))
    seg.append(((-nx, ny, 0), (-nx, ny, nh), C_NET, 2))
    seg.append(((nx, ny, 0), (nx, ny, nh), C_NET, 2))
    # ROI 박스
    x0, x1 = r["x_min"], r["x_max"]
    y0, y1 = r["y_min"], r["y_max"]
    z0, z1 = r["z_min"], r["z_max"]
    bot = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for i in range(4):
        seg.append((bot[i], bot[(i + 1) % 4], C_ROI, 1))
        seg.append((top[i], top[(i + 1) % 4], C_ROI, 1))
        seg.append((bot[i], top[i], C_ROI, 1))
    return seg


def axis_triad(length=0.30):
    return [((0, 0, 0), (length, 0, 0), (60, 60, 255), 3),      # X 빨강
            ((0, 0, 0), (0, length, 0), (60, 220, 60), 3),      # Y 초록
            ((0, 0, 0), (0, 0, length), (255, 120, 60), 3)]     # Z 파랑


# ============================================================================
# 내부 파라미터 로드
# ============================================================================

def load_intrinsics(cam_index):
    path = os.path.join(SETTINGS_DIR, "intrinsics_cam%d.json" % cam_index)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "내부 파라미터 파일이 없습니다:\n%s\n\n"
            "calibrate_intrinsics.py 로 먼저 캘리브레이션하고 결과를 "
            "settings/ 폴더에 넣어 주세요." % path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float64)
    dist = np.array(d["dist"], dtype=np.float64).reshape(1, -1)
    return dict(K=K, dist=dist, w=int(d["img_w"]), h=int(d["img_h"]),
                path=path, raw=d,
                provisional=bool(d.get("provisional", False)))


# ============================================================================
# 카메라 (스레드 캡처)
# ============================================================================

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY,
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)}


class CameraWorker:
    def __init__(self, index, w, h, fps=120, fourcc="MJPG", backend="dshow"):
        self.index = index
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.error = None
        self.fps_measured = 0.0
        self._req = (w, h, fps, fourcc, backend)
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        w, h, fps, fourcc, backend = self._req
        cap = cv2.VideoCapture(self.index, BACKENDS.get(backend, cv2.CAP_ANY))
        if not cap.isOpened():
            self.error = "카메라 %d 를 열 수 없습니다." % self.index
            self.running = False
            return
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        t0, n = time.time(), 0
        while self.running:
            ok, f = cap.read()
            if not ok or f is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f
            n += 1
            dt = time.time() - t0
            if dt >= 1.0:
                self.fps_measured = n / dt
                t0, n = time.time(), 0
        cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        self.thread = None
        self.frame = None


# ============================================================================
# 자세 추정
# ============================================================================

def estimate_pose(obj_pts, img_pts, K, dist):
    """returns dict(rvec, tvec, R, C, rms, per_point) or raises ValueError"""
    obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 4:
        raise ValueError("최소 4개 지점이 필요합니다. (권장 6개 이상)")

    flags = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP 실패. 클릭한 지점의 대응이 잘못되었을 수 "
                         "있습니다. 좌/우가 뒤바뀌지 않았는지 확인하세요.")

    # LM 정밀화
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    except cv2.error:
        pass

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img
    per_point = np.sqrt((d ** 2).sum(axis=1))
    rms = float(np.sqrt((d ** 2).sum() / len(d)))

    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).ravel()          # 월드 기준 카메라 위치
    return dict(rvec=rvec, tvec=tvec, R=R, C=C, rms=rms,
                per_point=per_point)


def project_segments(segments, rvec, tvec, K, dist, w, h, samples=32,
                     min_z=0.60, margin=0.25):
    """3D 선분을 이미지에 투영. 카메라 뒤쪽과 왜곡 모델이 발산하는 구간은 잘라낸다.

    선분 끝점만 투영하면, 한쪽 끝이 카메라 근처/뒤에 있을 때 왜곡 다항식이
    폭주해 화면 밖 수만 픽셀로 튀는 직선이 그려진다. 잘게 나눠 유효한
    구간만 이어 그린다.
    """
    R, _ = cv2.Rodrigues(rvec)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    lim_x = (-margin * w, (1.0 + margin) * w)
    lim_y = (-margin * h, (1.0 + margin) * h)
    out = []
    ts = np.linspace(0.0, 1.0, samples)
    for p0, p1, col, th in segments:
        a = np.asarray(p0, dtype=np.float64)
        b = np.asarray(p1, dtype=np.float64)
        pts3 = a[None, :] + (b - a)[None, :] * ts[:, None]
        cam = (R @ pts3.T + tv).T
        good = cam[:, 2] > min_z
        if not good.any():
            continue
        uv = np.full((samples, 2), np.nan)
        idx = np.where(good)[0]
        proj, _ = cv2.projectPoints(pts3[idx], rvec, tv, K, dist)
        uv[idx] = proj.reshape(-1, 2)
        ok = good & np.isfinite(uv).all(axis=1)
        ok &= (uv[:, 0] > lim_x[0]) & (uv[:, 0] < lim_x[1])
        ok &= (uv[:, 1] > lim_y[0]) & (uv[:, 1] < lim_y[1])
        for i in range(samples - 1):
            if ok[i] and ok[i + 1]:
                out.append((tuple(np.round(uv[i]).astype(int)),
                            tuple(np.round(uv[i + 1]).astype(int)), col, th))
    return out


# ============================================================================
# 모식도 위젯
# ============================================================================

class Schematic(tk.Canvas):
    """탁구대 상면도. 현재 클릭 대상 지점을 강조 표시."""

    W, H = 300, 300
    PAD = 24

    def __init__(self, master, geo, points, **kw):
        super().__init__(master, width=self.W, height=self.H,
                         bg=PANEL, highlightthickness=0, **kw)
        self.geo = geo
        self.points = points
        self.emphasize_roi = True
        L = float(geo["table_length"])
        nx = float(geo["table_width"]) / 2 + float(geo["net_overhang"])
        self.sx = (self.W - 2 * self.PAD) / (2 * nx)
        self.sy = (self.H - 2 * self.PAD - 16) / L
        self.s = min(self.sx, self.sy)
        self.cx = self.W / 2
        self.y0 = self.H - self.PAD          # Y=0 (로봇측)

    def w2c(self, x, y):
        return self.cx + x * self.s, self.y0 - y * self.s

    def redraw(self, current_idx, status, points=None):
        """status: dict key -> 'done' | 'skip' | None"""
        if points is not None:
            self.points = points
        self.delete("all")
        geo = self.geo
        L = float(geo["table_length"])
        hw = float(geo["table_width"]) / 2
        nx = hw + float(geo["net_overhang"])
        ny = L / 2

        a = self.w2c(-hw, 0); b = self.w2c(hw, L)
        self.create_rectangle(a[0], a[1], b[0], b[1],
                              outline="#5b8dd6", fill="#1b2740", width=2)
        c0 = self.w2c(0, 0); c1 = self.w2c(0, L)
        self.create_line(c0[0], c0[1], c1[0], c1[1], fill="#5b8dd6")
        n0 = self.w2c(-nx, ny); n1 = self.w2c(nx, ny)
        self.create_line(n0[0], n0[1], n1[0], n1[1], fill=FG, width=3)

        r = geo["roi"]
        r0 = self.w2c(r["x_min"], r["y_min"])
        r1 = self.w2c(r["x_max"], r["y_max"])
        if self.emphasize_roi:
            self.create_rectangle(r0[0], r0[1], r1[0], r1[1],
                                  outline="#ff7070", width=2,
                                  fill="#3a2030", stipple="gray25")
        self.create_rectangle(r0[0], r0[1], r1[0], r1[1],
                              outline="#ff7070", dash=(3, 3))

        self.create_text(self.cx, 12, text="사람", fill=DIM_C, font=FONT)
        self.create_text(self.cx, self.H - 8, text="로봇", fill=DIM_C,
                         font=FONT)

        for i, (key, name, hint, wp, req) in enumerate(self.points):
            px, py = self.w2c(wp[0], wp[1])
            st = status.get(key)
            if i == current_idx:
                self.create_oval(px - 9, py - 9, px + 9, py + 9,
                                 outline=ACC, width=2)
                col, rr = ACC, 5
            elif st == "done":
                col, rr = OK_C, 4
            elif st == "skip":
                col, rr = DIM_C, 3
            else:
                col, rr = "#6a6a75", 3
            self.create_oval(px - rr, py - rr, px + rr, py + rr,
                             fill=col, outline="")
            if wp[2] > 0:      # 네트 상단(높이 기준점) 표시
                self.create_text(px, py - 13, text="▲", fill=col,
                                 font=("Malgun Gothic", 7))
            self.create_text(px + (11 if wp[0] >= 0 else -11), py,
                             text=str(i + 1), fill=col, font=FONT_B,
                             anchor="w" if wp[0] >= 0 else "e")


# ============================================================================
# 메인 애플리케이션
# ============================================================================

class ExtrinsicsApp(tk.Tk):

    def __init__(self, args):
        super().__init__()
        self.title("탁구 로봇 - 카메라 외부 파라미터 설정")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.args = args
        self.geo, self.geo_path = load_geometry()
        self.refdata, self.rp_path = load_reference_points(self.geo)
        self.preset_name = self.refdata.get("default", "roi")
        if self.preset_name not in self.refdata["presets"]:
            self.preset_name = sorted(self.refdata["presets"])[0]
        self.points = preset_tuples(
            self.refdata["presets"][self.preset_name])
        self.segments = table_wireframe(self.geo) + axis_triad()
        self.roi_segments = roi_wireframe(self.geo) + axis_triad()

        self.cam_indices = {0: args.cam0, 1: args.cam1}
        self.cam_id = 0
        self.cam = None
        self.intr = None

        self.scale = float(args.scale)
        self.frozen_frame = None
        self.live = True
        self.disp_w = self.disp_h = 0

        # 클릭 상태
        self.clicked = {}        # key -> (x, y) 확정
        self.skipped = set()
        self.cur = 0
        self.provisional = None  # (x, y)
        self._prov_source = "user"
        self.pose = None
        self.overlay_mode = 1        # 0=끔, 1=기준 영역만, 2=탁구대 전체
        self.loupe_zoom = 8
        self.cursor_img = None
        self._photo = None
        self._loupe_photo = None

        self._closing = False
        self._after_id = None
        self._build_ui()
        self._bind_keys()
        self._open_camera(0, initial=True)
        self._tick()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=FONT, padding=3)
        style.configure("Acc.TButton", font=FONT_B, padding=4)

        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- 좌측: 영상 + 돋보기 ----
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        self.lbl_banner = tk.Label(left, bg="#101014", fg=ACC, font=FONT_T,
                                   anchor="w", padx=8, pady=4)
        self.lbl_banner.pack(fill="x")

        self.canvas = tk.Canvas(left, bg="#000", highlightthickness=1,
                                highlightbackground="#444", cursor="crosshair",
                                takefocus=1)
        self.canvas.pack()
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        bar = tk.Frame(left, bg=BG)
        bar.pack(fill="x", pady=(6, 0))
        self.lbl_cam = tk.Label(bar, text="", bg=BG, fg=DIM_C, font=FONT_M)
        self.lbl_cam.pack(side="left")
        self.lbl_coord = tk.Label(bar, text="", bg=BG, fg=ACC, font=FONT_M)
        self.lbl_coord.pack(side="right")

        # ---- 우측 패널 ----
        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))
        right2 = tk.Frame(root, bg=BG)
        right2.grid(row=0, column=2, sticky="n", padx=(10, 0))

        # 카메라 선택
        f_cam = self._section(right, "카메라")
        row = tk.Frame(f_cam, bg=PANEL); row.pack(fill="x")
        self.var_cam = tk.IntVar(value=0)
        for i in (0, 1):
            tk.Radiobutton(row, text="cam%d" % i, variable=self.var_cam,
                           value=i, command=self.on_switch_cam, bg=PANEL,
                           fg=FG, selectcolor=BG, font=FONT,
                           activebackground=PANEL, activeforeground=FG
                           ).pack(side="left", padx=(0, 8))
        tk.Label(row, text="장치 index", bg=PANEL, fg=DIM_C,
                 font=FONT).pack(side="left", padx=(6, 3))
        self.var_devidx = tk.StringVar(value=str(self.cam_indices[0]))
        tk.Spinbox(row, from_=0, to=9, width=3, textvariable=self.var_devidx,
                   font=FONT, command=self.on_devidx_change
                   ).pack(side="left")
        ttk.Button(row, text="재연결", width=7,
                   command=lambda: self._open_camera(self.cam_id)
                   ).pack(side="left", padx=4)

        # 진행 상황
        f_pt = self._section(right, "클릭 순서")
        prow = tk.Frame(f_pt, bg=PANEL); prow.pack(fill="x", pady=(0, 4))
        tk.Label(prow, text="기준", bg=PANEL, fg=DIM_C,
                 font=FONT).pack(side="left")
        names = sorted(self.refdata["presets"].keys())
        self.preset_labels = {
            self.refdata["presets"][n].get("label", n): n for n in names}
        self.var_preset = tk.StringVar(
            value=self.refdata["presets"][self.preset_name].get(
                "label", self.preset_name))
        self.cb_preset = ttk.Combobox(prow, textvariable=self.var_preset,
                                      width=22, state="readonly", takefocus=0,
                                      values=list(self.preset_labels.keys()))
        cb = self.cb_preset
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", self.on_preset_change)
        for seq in ("<Up>", "<Down>", "<Left>", "<Right>"):
            cb.bind(seq, lambda e: "break")
        ttk.Button(prow, text="재생성", width=7,
                   command=self.regenerate_presets).pack(side="left")

        hint = tk.Label(f_pt, bg=PANEL, fg=DIM_C, font=FONT, justify="left",
                        text="좌/우 기준 = 로봇 뒤에서 사람을 바라보는 방향\n"
                             "보이지 않는 지점은 [건너뛰기] 하세요")
        hint.pack(anchor="w", pady=(0, 4))

        self.listbox = tk.Listbox(f_pt, height=14, font=FONT, bg="#1a1a1f",
                                  fg=FG, selectbackground=ACC,
                                  highlightthickness=0, activestyle="none",
                                  exportselection=False, takefocus=0)
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        for seq in ("<Up>", "<Down>", "<Left>", "<Right>", "<Prior>",
                    "<Next>", "<Key>"):
            self.listbox.bind(seq, lambda e: "break")

        self.lbl_target = tk.Label(f_pt, bg=PANEL, fg=ACC, font=FONT_T,
                                   wraplength=300, justify="left")
        self.lbl_target.pack(anchor="w", pady=(6, 0))
        self.lbl_hint = tk.Label(f_pt, bg=PANEL, fg=WARN_C, font=FONT,
                                 wraplength=300, justify="left")
        self.lbl_hint.pack(anchor="w")
        self.lbl_world = tk.Label(f_pt, bg=PANEL, fg=DIM_C, font=FONT_M)
        self.lbl_world.pack(anchor="w")

        er = tk.Frame(f_pt, bg=PANEL); er.pack(fill="x", pady=(4, 0))
        tk.Label(er, text="화면 좌표 직접 입력  x", bg=PANEL, fg=DIM_C,
                 font=FONT).pack(side="left")
        self.var_px = tk.StringVar()
        self.var_py = tk.StringVar()
        tk.Entry(er, textvariable=self.var_px, width=8, font=FONT_M,
                 bg="#1a1a1f", fg=FG, insertbackground=FG).pack(side="left",
                                                                padx=2)
        tk.Label(er, text="y", bg=PANEL, fg=DIM_C, font=FONT).pack(side="left")
        tk.Entry(er, textvariable=self.var_py, width=8, font=FONT_M,
                 bg="#1a1a1f", fg=FG, insertbackground=FG).pack(side="left",
                                                                padx=2)
        ttk.Button(er, text="이동", width=6,
                   command=self.goto_typed).pack(side="left", padx=2)

        tk.Label(f_pt, bg=PANEL, fg=WARN_C, font=FONT, justify="left",
                 anchor="w", wraplength=310,
                 text=("키보드만으로 작업 가능:  K 임시점 생성 · "
                       "방향키 1px · Shift+방향키 0.1px · Ctrl+방향키 10px · "
                       "G 예상 위치로 · Enter 적용 · PgUp/PgDn 지점 이동 · "
                       "Esc 취소")).pack(fill="x", pady=(4, 0))

        # 모식도
        f_sch = self._section(right2, "탁구대 모식도")
        self.schematic = Schematic(f_sch, self.geo, self.points)
        self.schematic.pack()

        # 조작 버튼
        f_act = self._section(right, "조작")
        g = tk.Frame(f_act, bg=PANEL); g.pack(fill="x")
        self.btn_apply = ttk.Button(g, text="적용 (Enter)", width=15,
                                    style="Acc.TButton", command=self.apply_point)
        self.btn_apply.grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(g, text="이전 (Backspace)", width=15,
                   command=self.prev_point).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(g, text="건너뛰기", width=15,
                   command=self.skip_point).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(g, text="이 지점 지우기", width=15,
                   command=self.clear_point).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(g, text="정지/재생 (F)", width=15,
                   command=self.toggle_freeze).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(g, text="확대 + (Z)", width=15,
                   command=lambda: self.change_zoom(+1)).grid(row=2, column=1,
                                                              padx=2, pady=2)
        ttk.Button(g, text="축소 − (X)", width=15,
                   command=lambda: self.change_zoom(-1)).grid(row=3, column=0,
                                                              padx=2, pady=2)
        ttk.Button(g, text="오버레이 전환 (O)", width=15,
                   command=self.toggle_overlay).grid(row=3, column=1,
                                                     padx=2, pady=2)
        ttk.Button(g, text="임시점 생성 (K)", width=15,
                   command=lambda: self.seed_provisional()).grid(
            row=4, column=0, padx=2, pady=2)
        ttk.Button(g, text="예상 위치로 (G)", width=15,
                   command=self.goto_predicted).grid(row=4, column=1, padx=2,
                                                     pady=2)

        # 돋보기
        f_lp = self._section(right2, "돋보기")
        self.LW, self.LH = 300, 160
        self.loupe = tk.Canvas(f_lp, width=self.LW, height=self.LH, bg="#000",
                               highlightthickness=1, highlightbackground="#444",
                               cursor="tcross")
        self.loupe.pack()
        self.loupe.bind("<Button-1>", self.on_loupe_click)
        tk.Label(f_lp, text="돋보기 안을 클릭하면 1/배율 픽셀 단위로 지정됩니다",
                 bg=PANEL, fg=WARN_C, font=FONT, anchor="w").pack(fill="x")
        self.lbl_zoom = tk.Label(f_lp, text="", bg=PANEL, fg=DIM_C,
                                 font=FONT_M, justify="left", anchor="w",
                                 wraplength=300)
        self.lbl_zoom.pack(fill="x", pady=(3, 0))

        # 결과
        f_res = self._section(right2, "자세 계산 / 저장")
        g2 = tk.Frame(f_res, bg=PANEL); g2.pack(fill="x")
        ttk.Button(g2, text="자세 계산 (C)", width=15, style="Acc.TButton",
                   command=self.compute).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(g2, text="저장 (S)", width=15,
                   command=self.save).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(g2, text="불러오기", width=15,
                   command=self.load_saved).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(g2, text="이 카메라 재설정", width=15,
                   command=self.reset_camera).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(g2, text="스테레오 점검", width=15,
                   command=self.stereo_check).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(g2, text="설정 폴더 열기", width=15,
                   command=self.open_settings_dir).grid(row=2, column=1,
                                                        padx=2, pady=2)
        ttk.Button(g2, text="네트 높이 추정", width=15,
                   command=self.fit_net_height).grid(row=3, column=0, padx=2,
                                                     pady=2)
        ttk.Button(g2, text="기하 오차 진단", width=15,
                   command=self.diagnose_geometry).grid(row=3, column=1,
                                                        padx=2, pady=2)

        self.lbl_result = tk.Label(f_res, bg=PANEL, fg=FG, font=FONT_M,
                                   justify="left", anchor="w", wraplength=300)
        self.lbl_result.pack(fill="x", pady=(6, 0))

        self.lbl_status = tk.Label(root, bg=BG, fg=DIM_C, font=FONT,
                                   anchor="w")
        self.lbl_status.grid(row=1, column=0, columnspan=3, sticky="ew",
                             pady=(6, 0))

        self._refresh_list()
        self.after(120, self._focus_canvas)

    def _focus_canvas(self):
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    def _section(self, parent, title):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", pady=(0, 8))
        tk.Label(outer, text=title, bg=BG, fg=ACC, font=FONT_B,
                 anchor="w").pack(fill="x")
        inner = tk.Frame(outer, bg=PANEL, padx=8, pady=6)
        inner.pack(fill="x")
        return inner

    def _typing(self):
        """텍스트를 실제로 입력 중일 때만 True.

        주의: ttk.Combobox 는 ttk.Entry 의 하위 클래스라, 단순 isinstance 로
        판정하면 드롭다운에 포커스가 갔을 때 모든 단축키가 죽는다.
        readonly 상태면 입력 위젯으로 보지 않는다.
        """
        try:
            w = self.focus_get()
        except Exception:
            return False
        if w is None:
            return False
        if isinstance(w, ttk.Combobox):
            return False
        if isinstance(w, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Spinbox)):
            try:
                if str(w.cget("state")) == "readonly":
                    return False
            except Exception:
                pass
            return True
        return False

    def _key(self, fn):
        def h(ev=None):
            if self._typing():
                return
            fn()
            return "break"
        return h

    def _key2(self, fn, *args):
        def h(ev=None):
            if self._typing():
                return
            fn(*args)
            return "break"
        return h

    def _bind_keys(self):
        self.bind_all("<Return>", lambda e: self.apply_point())
        self.bind_all("<space>", self._key(self.apply_point))
        self.bind_all("<BackSpace>", lambda e: self.prev_point())
        for k in ("z", "Z"):
            self.bind_all("<KeyPress-%s>" % k, self._key(lambda: self.change_zoom(+1)))
        for k in ("x", "X"):
            self.bind_all("<KeyPress-%s>" % k, self._key(lambda: self.change_zoom(-1)))
        for k in ("f", "F"):
            self.bind_all("<KeyPress-%s>" % k, self._key(self.toggle_freeze))
        for k in ("c", "C"):
            self.bind_all("<KeyPress-%s>" % k, self._key(self.compute))
        for k in ("s", "S"):
            self.bind_all("<KeyPress-%s>" % k, self._key(self.save))
        for k in ("o", "O"):
            self.bind_all("<KeyPress-%s>" % k, self._key(self.toggle_overlay))
        arrows = (("Left", -1, 0), ("Right", 1, 0),
                  ("Up", 0, -1), ("Down", 0, 1))
        for mod, step in (("", 1.0), ("Shift-", 0.1), ("Control-", 10.0)):
            for name, dx, dy in arrows:
                self.bind_all("<%s%s>" % (mod, name),
                              self._key2(lambda a, b, st: self.nudge(a, b, st),
                                         dx, dy, step))
        self.bind_all("<Escape>", self._key(self.cancel_provisional))
        self.bind_all("<Prior>", self._key(lambda: self.goto_point(-1)))
        self.bind_all("<Next>", self._key(lambda: self.goto_point(1)))
        self.bind_all("<Tab>", self._key(lambda: self.goto_point(1)))
        for k in ("g", "G"):
            self.bind_all("<KeyPress-%s>" % k, self._key(self.goto_predicted))
        for k in ("k", "K"):
            self.bind_all("<KeyPress-%s>" % k,
                          self._key(lambda: self.seed_provisional()))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ 카메라
    def _open_camera(self, cam_id, initial=False):
        try:
            intr = load_intrinsics(cam_id)
        except FileNotFoundError as e:
            messagebox.showerror("내부 파라미터 없음", str(e))
            if initial:
                self.after(100, self.destroy)
            return
        if self.cam is not None:
            self.cam.stop()
            self.cam = None

        self.cam_id = cam_id
        self.intr = intr
        self.disp_w = int(intr["w"] * self.scale)
        self.disp_h = int(intr["h"] * self.scale)
        self.canvas.config(width=self.disp_w, height=self.disp_h)

        try:
            idx = int(self.var_devidx.get())
        except ValueError:
            idx = cam_id
        self.cam_indices[cam_id] = idx

        self.cam = CameraWorker(idx, intr["w"], intr["h"], self.args.fps,
                                self.args.fourcc, self.args.backend)
        self.cam.start()

        self.frozen_frame = None
        self.live = True
        self._load_state(cam_id)
        self._status("cam%d (장치 %d) 연결 중... 내부 파라미터: %s"
                     % (cam_id, idx, os.path.basename(intr["path"])))
        if intr["provisional"]:
            self._status("주의: cam%d 내부 파라미터가 provisional(복사본)입니다."
                         % cam_id)

    def on_switch_cam(self):
        new_id = self.var_cam.get()
        if new_id == self.cam_id:
            return
        self.var_devidx.set(str(self.cam_indices[new_id]))
        self._open_camera(new_id)

    def on_devidx_change(self):
        pass

    def _apply_preset(self, name, keep_clicks=False):
        old_clicked = dict(self.clicked)
        old_skipped = set(self.skipped)
        self.preset_name = name
        self.points = preset_tuples(self.refdata["presets"][name])
        self.clicked, self.skipped = {}, set()
        self.cur, self.provisional, self.pose = 0, None, None
        if keep_clicks:
            valid = set(p[0] for p in self.points)
            self.clicked = {k: v for k, v in old_clicked.items() if k in valid}
            self.skipped = set(k for k in old_skipped if k in valid)
            self._recompute_silent()
            self.cur = self._first_unset()
            self._refresh_list()
            if self.pose is not None:
                self.lbl_result.config(
                    text="기하 변경 반영 후 재계산\n%s" % self._pose_text(),
                    fg=OK_C)
            return
        self._load_state(self.cam_id, allow_preset_switch=False)

    def on_preset_change(self, ev=None):
        name = self.preset_labels.get(self.var_preset.get())
        if name is None or name == self.preset_name:
            return
        new_keys = set(d["key"] for d in
                       self.refdata["presets"][name]["points"])
        kept = [k for k in self.clicked if k in new_keys]
        lost = [k for k in self.clicked if k not in new_keys]
        if lost and not messagebox.askyesno(
                "기준 변경",
                "클릭한 지점 %d개 중 %d개는 새 기준에 없어 사라집니다.\n"
                "(%d개는 그대로 유지됩니다)\n\n계속할까요?"
                % (len(self.clicked), len(lost), len(kept))):
            self.var_preset.set(
                self.refdata["presets"][self.preset_name].get(
                    "label", self.preset_name))
            return
        self._apply_preset(name, keep_clicks=True)
        self._focus_canvas()
        n_new = len([p for p in self.points if p[0] not in self.clicked])
        self._status("기준 변경: %s — 기존 %d개 유지, %d개 추가로 찍으면 됩니다."
                     % (self.var_preset.get(), len(kept), n_new))

    def _refresh_preset_list(self):
        names = sorted(self.refdata["presets"].keys())
        self.preset_labels = {
            self.refdata["presets"][n].get("label", n): n for n in names}
        self.cb_preset.config(values=list(self.preset_labels.keys()))
        self.var_preset.set(
            self.refdata["presets"][self.preset_name].get(
                "label", self.preset_name))

    def regenerate_presets(self):
        if not messagebox.askyesno(
                "프리셋 재생성",
                "table_geometry.json 을 기준으로 reference_points.json 을\n"
                "새로 만듭니다. 직접 편집한 내용은 사라집니다.\n\n계속할까요?"):
            return
        self.geo, self.geo_path = load_geometry()
        self.refdata, self.rp_path = load_reference_points(
            self.geo, regenerate=True)
        self.segments = table_wireframe(self.geo) + axis_triad()
        self.roi_segments = roi_wireframe(self.geo) + axis_triad()
        if self.preset_name not in self.refdata["presets"]:
            self.preset_name = self.refdata.get("default", "roi")
        self._refresh_preset_list()
        self._apply_preset(self.preset_name, keep_clicks=True)
        self._status("프리셋 재생성 완료 (클릭 지점 유지): %s" % self.rp_path)

    # -------------------------------------------------------------- 상태
    def _state_path(self, cam_id):
        return os.path.join(SETTINGS_DIR, "extrinsics_cam%d.json" % cam_id)

    def _load_state(self, cam_id, allow_preset_switch=True):
        self.clicked, self.skipped, self.cur = {}, set(), 0
        self.provisional, self.pose = None, None
        path = self._state_path(cam_id)
        if not os.path.exists(path):
            self._refresh_list()
            self.lbl_result.config(
                text="저장된 설정 없음.\n1번 지점부터 클릭하세요.", fg=WARN_C)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            pn = d.get("reference_preset")
            if pn and pn != self.preset_name and not allow_preset_switch:
                self._refresh_list()
                self.lbl_result.config(
                    text="저장된 설정은 다른 기준(%s)으로 만들어졌습니다.\n"
                         "1번 지점부터 새로 클릭하세요." % pn, fg=WARN_C)
                return
            if pn and pn in self.refdata["presets"] and pn != self.preset_name:
                self.preset_name = pn
                self.points = preset_tuples(self.refdata["presets"][pn])
                try:
                    self.var_preset.set(
                        self.refdata["presets"][pn].get("label", pn))
                except Exception:
                    pass
            valid = set(p[0] for p in self.points)
            for p in [q for q in d.get("points", []) if q["key"] in valid]:
                self.clicked[p["key"]] = (float(p["image"][0]),
                                          float(p["image"][1]))
            self.skipped = set(k for k in d.get("skipped", []) if k in valid)
            self._recompute_silent()
            self.cur = self._first_unset()
            self._refresh_list()
            want = os.path.basename(self.intr["path"])
            got = d.get("intrinsics_file", want)
            if got != want:
                self.lbl_result.config(
                    text=("경고: 이 파일은 %s 로 계산되었습니다.\n"
                          "현재 카메라는 %s 를 씁니다. 다시 계산·저장하세요.\n%s"
                          % (got, want, self._pose_text())), fg=BAD_C)
            else:
                self.lbl_result.config(
                    text="저장된 설정을 불러왔습니다.\n%s\n%s"
                         % (os.path.basename(path), self._pose_text()),
                    fg=OK_C)
        except Exception as e:
            self.lbl_result.config(text="설정 로드 실패: %s" % e, fg=BAD_C)
            self._refresh_list()

    def _first_unset(self):
        for i, (key, *_rest) in enumerate(self.points):
            if key not in self.clicked and key not in self.skipped:
                return i
        return 0

    # ------------------------------------------------------------ 클릭
    def on_motion(self, ev):
        x = ev.x / self.scale
        y = ev.y / self.scale
        self.cursor_img = (x, y)
        self.lbl_coord.config(text="cursor  x %7.2f   y %7.2f" % (x, y))

    def on_click(self, ev):
        x = ev.x / self.scale
        y = ev.y / self.scale
        w, h = self.intr["w"], self.intr["h"]
        x = min(max(x, 0.0), w - 1.0)
        y = min(max(y, 0.0), h - 1.0)
        self.provisional = (x, y)
        self._prov_source = "user"
        if self.live:
            self.toggle_freeze(force_freeze=True)
        self._status("임시점 지정. 방향키로 미세조정 후 [적용] 하세요.")

    def on_loupe_click(self, ev):
        """돋보기 화면 클릭 -> 원본 이미지 좌표로 역산 (서브픽셀)."""
        center = self.provisional or self.cursor_img
        if center is None:
            center = (self.intr["w"] / 2, self.intr["h"] / 2)
        z = float(self.loupe_zoom)
        x0 = center[0] - (self.LW / z) / 2.0
        y0 = center[1] - (self.LH / z) / 2.0
        x = x0 + ev.x / z
        y = y0 + ev.y / z
        w, h = self.intr["w"], self.intr["h"]
        self.provisional = (min(max(x, 0.0), w - 1.0),
                            min(max(y, 0.0), h - 1.0))
        self._prov_source = "user"
        if self.live:
            self.toggle_freeze(force_freeze=True)
        self._status("돋보기에서 지정 (%.2f, %.2f) — %.2f px 정밀도"
                     % (self.provisional[0], self.provisional[1], 1.0 / z))

    def _predicted_uv(self, idx=None):
        """현재 자세로 이 지점이 있어야 할 화면 위치 (없으면 None)."""
        if self.pose is None:
            return None
        i = self.cur if idx is None else idx
        wp = np.array([self.points[i][3]], dtype=np.float64)
        try:
            uv = cv2.projectPoints(wp, self.pose["rvec"], self.pose["tvec"],
                                   self.intr["K"], self.intr["dist"]
                                   )[0].reshape(2)
        except cv2.error:
            return None
        if not np.all(np.isfinite(uv)):
            return None
        if not (0 <= uv[0] < self.intr["w"] and 0 <= uv[1] < self.intr["h"]):
            return None
        return (float(uv[0]), float(uv[1]))

    def seed_provisional(self, announce=True):
        """임시점이 없으면 만들어 준다. 클릭한 값 > 예상 위치 > 화면 중앙 순."""
        if self.provisional is not None:
            return False
        key = self.points[self.cur][0]
        src = "화면 중앙"
        if key in self.clicked:
            self.provisional = self.clicked[key]
            self._prov_source = "user"
            src = "기존 클릭 위치"
        else:
            self._prov_source = "seed"
            uv = self._predicted_uv()
            if uv is not None:
                self.provisional = uv
                src = "현재 자세로 예상한 위치"
            else:
                self.provisional = (self.intr["w"] / 2.0,
                                    self.intr["h"] / 2.0)
        if self.live:
            self.toggle_freeze(force_freeze=True)
        if announce:
            self._status("임시점 생성 (%s). 방향키로 이동하세요." % src)
        return True

    def goto_predicted(self):
        if self.pose is None:
            self._status("아직 자세가 없습니다. 지점을 4개 이상 찍고 "
                         "[자세 계산] 을 먼저 하세요.")
            return
        uv = self._predicted_uv()
        if uv is None:
            self._status("이 지점의 예상 위치가 화면 밖입니다. "
                         "보이지 않으면 [건너뛰기] 하세요.")
            return
        self.provisional = uv
        self._prov_source = "seed"
        if self.live:
            self.toggle_freeze(force_freeze=True)
        self._status("예상 위치로 이동 (%.2f, %.2f). 방향키로 미세조정하세요."
                     % uv)

    def goto_typed(self):
        try:
            x = float(self.var_px.get())
            y = float(self.var_py.get())
        except ValueError:
            self._status("숫자를 입력하세요. 예: 452.75 / 85.20")
            return
        w, h = self.intr["w"], self.intr["h"]
        self.provisional = (min(max(x, 0.0), w - 1.0),
                            min(max(y, 0.0), h - 1.0))
        self._prov_source = "user"
        if self.live:
            self.toggle_freeze(force_freeze=True)
        self._status("좌표 입력으로 이동 (%.2f, %.2f)" % self.provisional)

    def cancel_provisional(self):
        self.provisional = None
        self._status("임시점 취소.")

    def nudge(self, dx, dy, step):
        self.seed_provisional(announce=False)
        x, y = self.provisional
        w, h = self.intr["w"], self.intr["h"]
        self.provisional = (min(max(x + dx * step, 0.0), w - 1.0),
                            min(max(y + dy * step, 0.0), h - 1.0))
        self._prov_source = "user"
        self._status("임시점 (%.2f, %.2f)   이동량 %.2f px"
                     % (self.provisional[0], self.provisional[1], step))

    def goto_point(self, delta):
        self.cur = min(max(self.cur + delta, 0), len(self.points) - 1)
        self.provisional = None
        self._refresh_list()
        self.seed_provisional()

    def change_zoom(self, d):
        self._focus_canvas()
        levels = [2, 3, 4, 6, 8, 12, 16, 24, 32]
        try:
            i = levels.index(self.loupe_zoom)
        except ValueError:
            i = 4
        i = min(max(i + d, 0), len(levels) - 1)
        self.loupe_zoom = levels[i]

    def toggle_freeze(self, force_freeze=False):
        if self.live or force_freeze:
            f = self.cam.read() if self.cam else None
            if f is not None:
                self.frozen_frame = f
                self.live = False
        else:
            self.live = True
            self.frozen_frame = None

    OVERLAY_NAMES = ("끔", "기준 영역만", "탁구대 전체")

    def toggle_overlay(self):
        self.overlay_mode = (self.overlay_mode + 1) % 3
        self._status("오버레이: %s" % self.OVERLAY_NAMES[self.overlay_mode]
                     + ("   (탁구대 전체는 카메라 뒤쪽까지 포함되어 "
                        "왜곡되어 보일 수 있습니다)"
                        if self.overlay_mode == 2 else ""))

    def apply_point(self):
        if self.provisional is None:
            self._status("먼저 영상에서 지점을 클릭하세요.")
            return
        if self._prov_source == "seed":
            if not messagebox.askyesno(
                    "확인",
                    "이 임시점은 현재 자세로 계산한 '예상 위치'이며,\n"
                    "아직 실제 영상에서 확인되지 않았습니다.\n\n"
                    "그대로 적용하면 오차가 0으로 보여 품질 지표가\n"
                    "왜곡됩니다. 영상에서 실제 위치를 확인하고\n"
                    "방향키로 맞춘 뒤 적용하세요.\n\n그래도 적용할까요?"):
                return
        idx = self.cur
        key = self.points[idx][0]
        self.clicked[key] = self.provisional
        self.skipped.discard(key)
        self.provisional = None
        self.cur = min(idx + 1, len(self.points) - 1)
        self._recompute_silent()
        self._refresh_list()
        uv = self._predicted_uv()
        if uv is not None:
            self.provisional = uv
            self._prov_source = "seed"
            self._status("%d번 확정. 다음 지점의 예상 위치를 잡아뒀습니다 "
                         "— 방향키로 미세조정 후 Enter." % (idx + 1))
        else:
            self._status("%d번 지점 확정." % (idx + 1))

    def skip_point(self):
        key = self.points[self.cur][0]
        if self.points[self.cur][4]:
            if not messagebox.askyesno(
                    "필수 지점", "이 지점은 높이 기준점(필수)입니다.\n"
                    "건너뛰면 자세 추정이 매우 불안정해집니다.\n\n"
                    "그래도 건너뛸까요?"):
                return
        self.skipped.add(key)
        self.clicked.pop(key, None)
        self.provisional = None
        self.cur = min(self.cur + 1, len(self.points) - 1)
        self._recompute_silent()
        self._refresh_list()

    def prev_point(self):
        self.cur = max(self.cur - 1, 0)
        self.provisional = self.clicked.get(self.points[self.cur][0])
        self._refresh_list()

    def clear_point(self):
        key = self.points[self.cur][0]
        self.clicked.pop(key, None)
        self.skipped.discard(key)
        self.provisional = None
        self._recompute_silent()
        self._refresh_list()

    def on_list_select(self, ev):
        sel = self.listbox.curselection()
        if sel:
            self.cur = sel[0]
            self.provisional = self.clicked.get(self.points[self.cur][0])
            self._refresh_list()
            self._focus_canvas()

    def reset_camera(self):
        if not messagebox.askyesno(
                "재설정", "cam%d 의 클릭 지점과 계산 결과를 모두 지웁니다.\n"
                "저장된 파일은 [저장]을 다시 누를 때까지 유지됩니다.\n\n계속할까요?"
                % self.cam_id):
            return
        self.clicked, self.skipped = {}, set()
        self.cur, self.provisional, self.pose = 0, None, None
        self._refresh_list()
        self.lbl_result.config(text="재설정됨. 1번 지점부터 클릭하세요.",
                               fg=WARN_C)

    # -------------------------------------------------------------- 계산
    def _gather(self):
        obj, img, keys = [], [], []
        for key, name, hint, wp, req in self.points:
            if key in self.clicked:
                obj.append(wp)
                img.append(self.clicked[key])
                keys.append(key)
        return obj, img, keys

    def _recompute_silent(self):
        obj, img, keys = self._gather()
        if len(obj) < 4:
            self.pose = None
            return
        try:
            self.pose = estimate_pose(obj, img, self.intr["K"],
                                      self.intr["dist"])
            self.pose["keys"] = keys
        except Exception:
            self.pose = None

    def compute(self):
        obj, img, keys = self._gather()
        if len(obj) < 4:
            messagebox.showwarning("지점 부족",
                                   "최소 4개(권장 6개 이상)를 클릭해야 합니다.\n"
                                   "현재 %d개." % len(obj))
            return
        try:
            pose = estimate_pose(obj, img, self.intr["K"], self.intr["dist"])
        except Exception as e:
            messagebox.showerror("계산 실패", str(e))
            return
        pose["keys"] = keys
        self.pose = pose

        n_h = sum(1 for k in keys
                  if dict((p[0], p[3]) for p in self.points)[k][2] > 1e-6)
        warn = []
        if len(obj) < 6:
            warn.append("지점이 %d개뿐입니다. 6개 이상 권장." % len(obj))
        if n_h == 0:
            warn.append("높이 기준점(네트 상단)이 없습니다. "
                        "모든 점이 한 평면에 있어 자세가 매우 불안정합니다.")
        elif n_h == 1:
            warn.append("높이 기준점이 1개뿐입니다. 2개 모두 클릭을 권장합니다.")
        if pose["rms"] > 2.0:
            warn.append("재투영 오차가 큽니다(%.2f px). 오차가 큰 지점을 "
                        "다시 클릭하세요." % pose["rms"])
        if pose["C"][2] < 0:
            warn.append("카메라가 탁구대 상면보다 아래에 있다고 계산되었습니다. "
                        "좌/우 지점이 뒤바뀌지 않았는지 확인하세요.")

        worst = int(np.argmax(pose["per_point"]))
        txt = self._pose_text()
        txt += "\n최대 오차 지점: %s (%.2f px)" % (
            self._name_of(keys[worst]), pose["per_point"][worst])
        if warn:
            txt += "\n\n" + "\n".join("! " + w for w in warn)
            self.lbl_result.config(text=txt, fg=WARN_C)
        else:
            txt += "\n\n[OK] 품질 양호. [저장] 하세요."
            self.lbl_result.config(text=txt, fg=OK_C)
        self._refresh_list()

    # ------------------------------------------------------- 기하 자동 진단
    def _rms_with(self, net_h=None, drop_height=False):
        """네트 높이를 바꿔가며 PnP RMS 를 계산 (외부 라이브러리 불필요)."""
        obj, img = [], []
        for key, name, hint, wp, req in self.points:
            if key not in self.clicked:
                continue
            w = list(wp)
            if w[2] > 1e-6:
                if drop_height:
                    continue
                if net_h is not None:
                    w[2] = net_h
            obj.append(w)
            img.append(self.clicked[key])
        if len(obj) < 4:
            return None
        try:
            return estimate_pose(obj, img, self.intr["K"],
                                 self.intr["dist"])["rms"]
        except Exception:
            return None

    def fit_net_height(self):
        if len([k for k in self.clicked]) < 6:
            messagebox.showwarning("네트 높이 추정",
                                   "지점을 6개 이상 클릭한 뒤 실행하세요.")
            return
        n_h = sum(1 for key, _n, _h, wp, _r in self.points
                  if key in self.clicked and wp[2] > 1e-6)
        if n_h < 1:
            messagebox.showwarning("네트 높이 추정",
                                   "네트 상단 지점을 최소 1개 클릭해야 합니다.")
            return
        cur = float(self.geo["net_height"])
        base = self._rms_with()
        best_h, best_r = None, None
        for h in np.arange(0.080, 0.2001, 0.0005):
            r = self._rms_with(net_h=float(h))
            if r is not None and (best_r is None or r < best_r):
                best_r, best_h = r, float(h)
        if best_h is None:
            messagebox.showerror("네트 높이 추정", "계산에 실패했습니다.")
            return
        msg = ("현재 설정값 : %.4f m  ->  RMS %.3f px\n"
               "추정 최적값 : %.4f m  ->  RMS %.3f px\n"
               "차이        : %+.1f mm\n\n"
               "네트 상단 지점 %d개 기준입니다."
               % (cur, base, best_h, best_r, (best_h - cur) * 1000, n_h))
        if abs(best_h - cur) < 0.003:
            messagebox.showinfo("네트 높이 추정",
                                msg + "\n\n[OK] 현재 값이 적절합니다.")
            return
        msg += ("\n\n! 실제 네트를 자로 재서 확인하세요.\n"
                "  (테이프 라인 위치에서 탁구대 상면 ~ 흰 띠 윗변)\n\n"
                "table_geometry.json 의 net_height 를 %.4f 로 바꾸고\n"
                "바로 다시 계산할까요?" % best_h)
        if not messagebox.askyesno("네트 높이 추정", msg):
            return
        self.geo["net_height"] = float(best_h)
        with open(self.geo_path, "w", encoding="utf-8") as f:
            json.dump(self.geo, f, indent=2, ensure_ascii=False)
        self.refdata, self.rp_path = load_reference_points(
            self.geo, regenerate=True)
        self.segments = table_wireframe(self.geo) + axis_triad()
        self.roi_segments = roi_wireframe(self.geo) + axis_triad()
        self.schematic.geo = self.geo
        self._refresh_preset_list()
        self._apply_preset(self.preset_name, keep_clicks=True)
        self._status("net_height = %.4f m 적용 및 재계산 완료" % best_h)

    def diagnose_geometry(self):
        if len(self.clicked) < 6:
            messagebox.showwarning("기하 오차 진단",
                                   "지점을 6개 이상 클릭한 뒤 실행하세요.")
            return
        base = self._rms_with()
        no_h = self._rms_with(drop_height=True)
        lines = ["전체 지점 RMS            : %.3f px" % base]
        if no_h is not None:
            lines.append("네트 상단 제외 RMS       : %.3f px" % no_h)
        best = None
        for h in np.arange(0.080, 0.2001, 0.0005):
            r = self._rms_with(net_h=float(h))
            if r is not None and (best is None or r < best[1]):
                best = (float(h), r)
        if best:
            lines.append("네트 높이 최적화 RMS     : %.3f px (h=%.4f m)"
                         % (best[1], best[0]))
        tips = []
        if no_h is not None and no_h < base * 0.7:
            tips.append("탁구대 평면 지점은 정확한데 네트 상단만 어긋납니다.\n"
                        "  -> [네트 높이 추정] 을 실행하세요.")
        if no_h is not None and no_h > 1.5:
            tips.append("탁구대 평면 지점 자체의 오차가 큽니다.\n"
                        "  -> 테이프 치수(table_geometry.json 의 roi)를 실측과\n"
                        "     대조하고, 오차 큰 지점을 돋보기로 다시 찍으세요.")
        if self.pose is not None:
            e = np.asarray(self.pose["per_point"])
            if len(e) and e.max() > 3 * max(np.median(e), 0.2):
                worst = self.pose["keys"][int(np.argmax(e))]
                tips.append("특정 지점만 튑니다: %s (%.2f px)\n"
                            "  -> 이 지점만 다시 클릭해 보세요."
                            % (self._name_of(worst), e.max()))
        msg = "\n".join(lines)
        if tips:
            msg += "\n\n" + "\n".join("! " + t for t in tips)
        else:
            msg += "\n\n[OK] 뚜렷한 계통 오차는 보이지 않습니다."
        messagebox.showinfo("기하 오차 진단", msg)

    def _name_of(self, key):
        for i, (k, name, *_r) in enumerate(self.points):
            if k == key:
                return "%d.%s" % (i + 1, name)
        return key

    def _pose_text(self):
        if self.pose is None:
            return "자세 미계산"
        C = self.pose["C"]
        R = self.pose["R"]
        fwd = R[2, :]                     # 월드에서의 카메라 광축 방향
        pitch = np.degrees(np.arcsin(-max(min(fwd[2], 1.0), -1.0)))
        yaw = np.degrees(np.arctan2(fwd[0], fwd[1]))
        return ("사용 지점 %d개   재투영 RMS %.3f px\n"
                "카메라 위치  X %+.3f  Y %+.3f  Z %+.3f (m)\n"
                "광축         하향 %.1f°   좌우 %+.1f°"
                % (len(self.pose["keys"]), self.pose["rms"],
                   C[0], C[1], C[2], pitch, yaw))

    # -------------------------------------------------------------- 저장
    def save(self):
        if self.pose is None:
            messagebox.showwarning("저장 불가",
                                   "먼저 [자세 계산]을 실행하세요.")
            return
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        path = self._state_path(self.cam_id)
        pose = self.pose
        R = pose["R"]
        wmap = dict((p[0], p[3]) for p in self.points)
        data = {
            "camera_index": self.cam_id,
            "device_index": self.cam_indices[self.cam_id],
            "reference_preset": self.preset_name,
            "img_w": self.intr["w"], "img_h": self.intr["h"],
            "intrinsics_file": os.path.basename(self.intr["path"]),
            "world_frame": {
                "origin": "탁구대 상면, 로봇측 엔드라인 중앙",
                "x": "로봇→사람 기준 오른쪽", "y": "사람 방향", "z": "위",
                "units": "m"
            },
            "rvec": [float(v) for v in pose["rvec"].ravel()],
            "tvec": [float(v) for v in pose["tvec"].ravel()],
            "R": [[float(v) for v in row] for row in R],
            "camera_position_world": [float(v) for v in pose["C"]],
            "reprojection_rms_px": float(pose["rms"]),
            "n_points": len(pose["keys"]),
            "points": [
                {"key": k,
                 "world": [float(v) for v in wmap[k]],
                 "image": [float(self.clicked[k][0]),
                           float(self.clicked[k][1])],
                 "error_px": float(pose["per_point"][i])}
                for i, k in enumerate(pose["keys"])
            ],
            "skipped": sorted(self.skipped),
            "table_geometry_file": "table_geometry.json",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._status("저장 완료: %s" % path)
        messagebox.showinfo("저장 완료",
                            "%s\n\nRMS %.3f px, 지점 %d개"
                            % (path, pose["rms"], len(pose["keys"])))

    def load_saved(self):
        self._load_state(self.cam_id)

    def open_settings_dir(self):
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(SETTINGS_DIR)
            elif sys.platform == "darwin":
                os.system('open "%s"' % SETTINGS_DIR)
            else:
                os.system('xdg-open "%s"' % SETTINGS_DIR)
        except Exception as e:
            messagebox.showinfo("설정 폴더", SETTINGS_DIR + "\n\n%s" % e)

    def stereo_check(self):
        paths = [self._state_path(i) for i in (0, 1)]
        if not all(os.path.exists(p) for p in paths):
            messagebox.showwarning(
                "스테레오 점검",
                "두 카메라 모두 저장되어야 합니다.\n"
                "현재: cam0 %s / cam1 %s"
                % ("O" if os.path.exists(paths[0]) else "X",
                   "O" if os.path.exists(paths[1]) else "X"))
            return
        ds = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                ds.append(json.load(f))
        C0 = np.array(ds[0]["camera_position_world"])
        C1 = np.array(ds[1]["camera_position_world"])
        B = float(np.linalg.norm(C0 - C1))
        R0 = np.array(ds[0]["R"]); R1 = np.array(ds[1]["R"])
        a0, a1 = R0[2, :], R1[2, :]
        cosang = float(np.clip(np.dot(a0, a1), -1, 1))
        conv = np.degrees(np.arccos(cosang))

        roi = self.geo["roi"]
        ctr = np.array([(roi["x_min"] + roi["x_max"]) / 2,
                        (roi["y_min"] + roi["y_max"]) / 2,
                        (roi["z_min"] + roi["z_max"]) / 2])
        Z = float(np.linalg.norm((C0 + C1) / 2 - ctr))
        f = float(np.array(self.intr["K"])[0, 0])
        sigma = 0.5
        depth_err = (Z ** 2) * sigma / (f * B) * 1000 if B > 1e-6 else 9999

        msg = ("기선(baseline)      : %.3f m\n"
               "광축 교차각        : %.1f°\n"
               "ROI 중심까지 거리  : %.2f m\n"
               "예상 깊이 오차     : %.1f mm  (0.5 px 가정)\n\n"
               "cam0 RMS %.3f px / cam1 RMS %.3f px"
               % (B, conv, Z, depth_err,
                  ds[0]["reprojection_rms_px"], ds[1]["reprojection_rms_px"]))
        tips = []
        if B < 0.35:
            tips.append("기선이 너무 짧습니다(<0.35 m). 0.6 m 이상 권장.")
        if conv < 5:
            tips.append("광축이 거의 평행합니다. 약간 안쪽으로 틀어(toe-in) 주세요.")
        if conv > 75:
            tips.append("교차각이 큽니다. 정밀도는 좋지만 가림/매칭이 어려워집니다.")
        if depth_err > 10:
            tips.append("예상 깊이 오차가 10 mm를 넘습니다. 기선을 늘리거나 "
                        "카메라를 ROI에 가깝게 배치하세요.")
        if tips:
            msg += "\n\n" + "\n".join("! " + t for t in tips)
        else:
            msg += "\n\n[OK] 배치 양호."
        messagebox.showinfo("스테레오 점검", msg)

    # -------------------------------------------------------- 리스트/표시
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        errmap = {}
        if self.pose is not None:
            errmap = dict(zip(self.pose["keys"], self.pose["per_point"]))
        for i, (key, name, hint, wp, req) in enumerate(self.points):
            if key in self.clicked:
                e = errmap.get(key)
                mark = "[v]" if e is None else "[v %.2fpx]" % e
            elif key in self.skipped:
                mark = "[--]"
            else:
                mark = "[  ]"
            star = "*" if req else " "
            self.listbox.insert(tk.END, "%s%2d. %-26s %s"
                                % (star, i + 1, name[:26], mark))
            if key in self.clicked:
                self.listbox.itemconfig(i, fg=OK_C)
            elif key in self.skipped:
                self.listbox.itemconfig(i, fg=DIM_C)
            else:
                self.listbox.itemconfig(i, fg=FG)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.cur)
        self.listbox.see(self.cur)

        key, name, hint, wp, req = self.points[self.cur]
        self.lbl_target.config(text="%d. %s" % (self.cur + 1, name))
        self.lbl_hint.config(text=hint)
        self.lbl_world.config(text="월드 좌표  X %+.4f  Y %+.4f  Z %+.4f"
                                   % wp)
        status = {}
        for k in self.clicked:
            status[k] = "done"
        for k in self.skipped:
            status[k] = "skip"
        self.schematic.emphasize_roi = (self.preset_name == "roi")
        self.schematic.redraw(self.cur, status, self.points)

    def _status(self, msg):
        self.lbl_status.config(text=msg)

    # ------------------------------------------------------------- 렌더
    def _tick(self):
        if self._closing:
            return
        try:
            self._render()
        except Exception as e:
            try:
                self._status("렌더 오류: %s" % e)
            except Exception:
                pass
        self._after_id = self.after(33, self._tick)

    def _render(self):
        if self.cam is None:
            return
        if self.cam.error:
            self._status(self.cam.error)
            return
        frame = self.frozen_frame if not self.live else self.cam.read()
        if frame is None:
            return
        vis = frame.copy()

        # 오버레이: 확정 지점 + 재투영 + 와이어프레임
        if self.overlay_mode:
            segs = (self.roi_segments if self.overlay_mode == 1
                    else self.segments)
            if self.pose is not None:
                for p0, p1, col, th in project_segments(
                        segs, self.pose["rvec"], self.pose["tvec"],
                        self.intr["K"], self.intr["dist"],
                        self.intr["w"], self.intr["h"]):
                    cv2.line(vis, p0, p1, col, th, cv2.LINE_AA)
            errmap = {}
            if self.pose is not None:
                errmap = dict(zip(self.pose["keys"], self.pose["per_point"]))
            for i, (key, name, hint, wp, req) in enumerate(self.points):
                if key not in self.clicked:
                    continue
                x, y = self.clicked[key]
                e = errmap.get(key, 0.0)
                col = (90, 220, 120) if e < 2.0 else (70, 70, 235)
                p = (int(round(x)), int(round(y)))
                cv2.drawMarker(vis, p, col, cv2.MARKER_CROSS, 12, 1,
                               cv2.LINE_AA)
                cv2.circle(vis, p, 7, col, 1, cv2.LINE_AA)
                cv2.putText(vis, str(i + 1), (p[0] + 9, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(vis, str(i + 1), (p[0] + 9, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                            cv2.LINE_AA)

        if self.provisional is not None:
            x, y = self.provisional
            if self.focus_get() not in (None,) and not self._typing():
                self.var_px.set("%.2f" % x)
                self.var_py.set("%.2f" % y)
            p = (int(round(x)), int(round(y)))
            cv2.line(vis, (p[0] - 14, p[1]), (p[0] + 14, p[1]),
                     (60, 220, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (p[0], p[1] - 14), (p[0], p[1] + 14),
                     (60, 220, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, p, 10, (60, 220, 255), 1, cv2.LINE_AA)

        self.lbl_banner.config(
            text="%s   |   %d/%d   %s   |   오버레이: %s"
                 % ("● 실시간" if self.live else "■ 정지",
                    self.cur + 1, len(self.points),
                    self.points[self.cur][1],
                    self.OVERLAY_NAMES[self.overlay_mode]),
            fg=WARN_C if self.live else ACC)

        disp = cv2.resize(vis, (self.disp_w, self.disp_h),
                          interpolation=cv2.INTER_NEAREST)
        self._photo = to_photo(disp)
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw",
                                 tags="img")
        self.canvas.tag_lower("img")

        self._render_loupe(frame)
        self.lbl_cam.config(
            text="cam%d  %dx%d  %.0f fps  %s"
                 % (self.cam_id, self.intr["w"], self.intr["h"],
                    self.cam.fps_measured,
                    "정지" if not self.live else "실시간"))
        self.lbl_zoom.config(text="배율 %dx    Z 확대 / X 축소\n"
                                  "방향키 1 px  ·  Shift 0.1 px  ·  Ctrl 10 px"
                                  % self.loupe_zoom)

    def _render_loupe(self, frame):
        LW, LH = self.LW, self.LH
        center = self.provisional or self.cursor_img
        if center is None:
            center = (self.intr["w"] / 2, self.intr["h"] / 2)
        z = self.loupe_zoom
        rw, rh = LW / z, LH / z
        cx, cy = center
        x0, y0 = cx - rw / 2, cy - rh / 2

        canvas_img = np.zeros((LH, LW, 3), np.uint8)
        ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
        ix1, iy1 = ix0 + int(np.ceil(rw)) + 1, iy0 + int(np.ceil(rh)) + 1
        sx0, sy0 = max(ix0, 0), max(iy0, 0)
        sx1, sy1 = min(ix1, frame.shape[1]), min(iy1, frame.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            patch = frame[sy0:sy1, sx0:sx1]
            big = cv2.resize(patch, ((sx1 - sx0) * z, (sy1 - sy0) * z),
                             interpolation=cv2.INTER_NEAREST)
            ox = int(round((sx0 - x0) * z))
            oy = int(round((sy0 - y0) * z))
            bx0, by0 = max(ox, 0), max(oy, 0)
            px0, py0 = max(-ox, 0), max(-oy, 0)
            cw = min(big.shape[1] - px0, LW - bx0)
            chh = min(big.shape[0] - py0, LH - by0)
            if cw > 0 and chh > 0:
                canvas_img[by0:by0 + chh, bx0:bx0 + cw] = \
                    big[py0:py0 + chh, px0:px0 + cw]

        # 픽셀 격자 (배율이 충분할 때만)
        if z >= 8:
            for gx in range(0, LW, z):
                off = int(round((np.ceil(x0) - x0) * z))
                cv2.line(canvas_img, (gx + off, 0), (gx + off, LH),
                         (55, 55, 55), 1)
            for gy in range(0, LH, z):
                off = int(round((np.ceil(y0) - y0) * z))
                cv2.line(canvas_img, (0, gy + off), (LW, gy + off),
                         (55, 55, 55), 1)

        col = (60, 220, 255) if self.provisional is not None else (150, 150, 150)
        cv2.line(canvas_img, (LW // 2 - 18, LH // 2), (LW // 2 - 4, LH // 2),
                 col, 1)
        cv2.line(canvas_img, (LW // 2 + 4, LH // 2), (LW // 2 + 18, LH // 2),
                 col, 1)
        cv2.line(canvas_img, (LW // 2, LH // 2 - 18), (LW // 2, LH // 2 - 4),
                 col, 1)
        cv2.line(canvas_img, (LW // 2, LH // 2 + 4), (LW // 2, LH // 2 + 18),
                 col, 1)
        cv2.circle(canvas_img, (LW // 2, LH // 2), 2, col, -1)

        self._loupe_photo = to_photo(canvas_img)
        self.loupe.delete("all")
        self.loupe.create_image(0, 0, image=self._loupe_photo, anchor="nw")

    # -------------------------------------------------------------- 종료
    def on_close(self):
        self._closing = True
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
        self.destroy()


# ============================================================================
# 이미지 -> Tk 변환 (Pillow 없어도 동작)
# ============================================================================

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
    p = argparse.ArgumentParser(
        description="탁구 로봇 - 카메라 외부 파라미터 설정 GUI")
    p.add_argument("--cam0", type=int, default=0, help="cam0 의 장치 index")
    p.add_argument("--cam1", type=int, default=1, help="cam1 의 장치 index")
    p.add_argument("--scale", type=float, default=1.25,
                   help="영상 표시 배율")
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--backend", default="dshow",
                   choices=list(BACKENDS.keys()))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    print_env_banner()
    app = ExtrinsicsApp(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())