#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_intrinsics.py
=======================
ChArUco 보드를 이용한 단일 카메라 내부 파라미터(intrinsics) 캘리브레이션 도구.

탁구 로봇 프로젝트 1단계 - 스테레오 트래킹의 선행 작업.
Arducam OV9782 (640x480 @ 120fps, MJPG) 기준으로 기본값이 맞춰져 있음.

주요 기능
---------
  * 실시간 프리뷰 + ChArUco 자동 검출
  * 품질 게이트: 코너 수 / 블러 / 자세 다양성(novelty) / 화면 커버리지
  * 커버리지 히트맵 오버레이 -> 어느 구역을 더 찍어야 하는지 시각적으로 안내
  * 자동 캡처 모드 (쿨다운 + 품질 통과 시에만 저장)
  * 캘리브레이션 후 뷰별 재투영 오차 표시 + 불량 뷰 자동 프루닝 + 재계산
  * 왜곡 보정 결과 실시간 비교 프리뷰
  * intrinsics_camN.json 저장 (기존 스키마와 호환)
  * 딕셔너리 자동 탐색(--auto-dict), 보드 이미지 생성(--generate-board)
  * 저장된 이미지 폴더로부터 오프라인 캘리브레이션(--from-dir)
  * OpenCV 4.6 이하 / 4.7 이상 ArUco API 양쪽 모두 지원

사용 예
-------
  # 0) 보드 스펙을 모를 때: 딕셔너리 자동 탐색
  python calibrate_intrinsics.py --cam 0 --auto-dict

  # 1) cam0 캘리브레이션
  python calibrate_intrinsics.py --cam 0 --squares-x 5 --squares-y 7 \
      --square-mm 33.0 --marker-mm 25.0 --dict DICT_4X4_50

  # 2) 저장된 이미지로 재계산
  python calibrate_intrinsics.py --from-dir captures/cam0 --squares-x 5 --squares-y 7

  # 3) 인쇄용 보드 이미지 생성
  python calibrate_intrinsics.py --generate-board board.png --squares-x 5 --squares-y 7

조작 키
-------
  SPACE : 현재 프레임 캡처
  A     : 자동 캡처 on/off
  C     : 캘리브레이션 실행
  P     : 불량 뷰 프루닝 후 재캘리브레이션 (C 실행 후 가능)
  U     : 왜곡 보정 프리뷰 on/off (C 실행 후 가능)
  M     : 커버리지 히트맵 on/off
  D     : 마지막 캡처 삭제
  R     : 전체 캡처 초기화
  S     : 결과 JSON 저장
  H     : 도움말 콘솔 출력
  Q/ESC : 종료
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ============================================================================
# 상수 / 기본값
# ============================================================================

NEW_API = hasattr(cv2.aruco, "CharucoDetector")

DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]

COV_GRID_X, COV_GRID_Y = 8, 6      # 커버리지 그리드 해상도
MIN_VIEWS_FOR_CALIB = 8            # 캘리브레이션 최소 뷰 수
MIN_CORNERS_PER_VIEW = 6           # calibrateCamera 요구 최소 대응점

# 색상 (BGR)
C_OK = (90, 220, 90)
C_WARN = (60, 200, 255)
C_BAD = (70, 70, 235)
C_TXT = (245, 245, 245)
C_DIM = (150, 150, 150)


# ============================================================================
# ChArUco 래퍼 (구/신 API 흡수)
# ============================================================================

class Charuco:
    """OpenCV 버전에 무관하게 동일한 인터페이스를 제공하는 ChArUco 래퍼."""

    def __init__(self, squares_x, squares_y, square_m, marker_m,
                 dict_name="DICT_4X4_50", legacy=False, refine=True):
        if marker_m >= square_m:
            raise ValueError("marker_mm 는 square_mm 보다 작아야 합니다.")
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_m = float(square_m)
        self.marker_m = float(marker_m)
        self.dict_name = dict_name
        self.legacy = bool(legacy)

        if not hasattr(cv2.aruco, dict_name):
            raise ValueError("알 수 없는 딕셔너리: %s" % dict_name)
        dict_id = getattr(cv2.aruco, dict_name)

        if NEW_API:
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            self.board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_m, self.marker_m, self.dictionary)
            # 4.6 이전 OpenCV로 만든 보드는 마커 배치 규칙이 다르므로 legacy 필요
            if self.legacy and hasattr(self.board, "setLegacyPattern"):
                self.board.setLegacyPattern(True)

            det_p = cv2.aruco.DetectorParameters()
            det_p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            det_p.adaptiveThreshWinSizeMin = 3
            det_p.adaptiveThreshWinSizeMax = 33
            det_p.adaptiveThreshWinSizeStep = 5

            ch_p = cv2.aruco.CharucoParameters()
            ch_p.tryRefineMarkers = bool(refine)
            ch_p.minMarkers = 2
            self.detector = cv2.aruco.CharucoDetector(
                self.board, ch_p, det_p, cv2.aruco.RefineParameters())
        else:
            self.dictionary = cv2.aruco.Dictionary_get(dict_id)
            self.board = cv2.aruco.CharucoBoard_create(
                self.squares_x, self.squares_y,
                self.square_m, self.marker_m, self.dictionary)
            self.det_params = cv2.aruco.DetectorParameters_create()
            self.det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = None

    # 보드 전체의 체스보드 코너 개수 (내부 코너)
    @property
    def n_chessboard_corners(self):
        return (self.squares_x - 1) * (self.squares_y - 1)

    def detect(self, gray):
        """returns (ch_corners Nx1x2 float32|None, ch_ids Nx1 int|None,
                    m_corners list|None, m_ids Nx1|None)"""
        if NEW_API:
            ch_c, ch_i, m_c, m_i = self.detector.detectBoard(gray)
            return ch_c, ch_i, m_c, m_i
        m_c, m_i, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.det_params)
        if m_i is None or len(m_i) == 0:
            return None, None, m_c, m_i
        try:
            _, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
                m_c, m_i, gray, self.board)
        except cv2.error:
            return None, None, m_c, m_i
        return ch_c, ch_i, m_c, m_i

    def match_points(self, ch_c, ch_i):
        """ChArUco 코너 -> (objectPoints Nx1x3, imagePoints Nx1x2)"""
        if ch_c is None or ch_i is None or len(ch_i) < MIN_CORNERS_PER_VIEW:
            return None, None
        if NEW_API:
            try:
                obj, img = self.board.matchImagePoints(ch_c, ch_i)
            except cv2.error:
                return None, None
            if obj is None or img is None or len(obj) < MIN_CORNERS_PER_VIEW:
                return None, None
            return obj.astype(np.float32), img.astype(np.float32)
        all_obj = np.asarray(self.board.chessboardCorners, dtype=np.float32)
        idx = np.asarray(ch_i).flatten().astype(int)
        obj = all_obj[idx].reshape(-1, 1, 3).astype(np.float32)
        img = np.asarray(ch_c, dtype=np.float32).reshape(-1, 1, 2)
        return obj, img

    def generate_image(self, out_px=1600, margin_px=40, border_bits=1):
        ratio = self.squares_y / float(self.squares_x)
        w = int(out_px)
        h = int(round(out_px * ratio))
        if NEW_API:
            return self.board.generateImage((w, h), marginSize=margin_px,
                                            borderBits=border_bits)
        return self.board.draw((w, h), marginSize=margin_px,
                               borderBits=border_bits)


# ============================================================================
# 카메라 스트림
# ============================================================================

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
}


class CameraStream:
    def __init__(self, index, width, height, fps, fourcc="MJPG",
                 backend="dshow", exposure=None):
        self.index = index
        self.req = (width, height, fps)
        tried = [backend] + [b for b in ("msmf", "dshow", "any")
                             if b != backend]
        self.cap = None
        self.backend_used = None
        for bname in tried:
            cap = cv2.VideoCapture(index, BACKENDS.get(bname, cv2.CAP_ANY))
            if cap.isOpened():
                self.cap = cap
                self.backend_used = bname
                if bname != backend:
                    print("[*] backend=%s 실패 -> %s 로 자동 전환했습니다."
                          % (backend, bname))
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(
                "카메라 %d 를 열 수 없습니다. (시도한 backend: %s)\n"
                "  - 다른 프로그램이 점유 중인지 확인 "
                "(이 스크립트의 다른 창 포함)\n"
                "  - python calibrate_intrinsics.py --list-cameras 로 "
                "사용 가능한 index/backend 확인\n"
                "  - Windows 설정 > 개인 정보 및 보안 > 카메라 에서 "
                "'데스크톱 앱이 카메라에 액세스' 허용\n"
                "  - opencv-python-headless 가 설치되어 있으면 제거"
                % (index, ", ".join(tried)))

        # FOURCC 를 먼저 설정해야 고해상도/고프레임 모드가 열립니다.
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC,
                         cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if exposure is not None:
            # 백엔드마다 auto-exposure 표현이 달라 두 방식 모두 시도
            for auto_val in (0.25, 1):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_val)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError("카메라 %d 에서 프레임을 읽지 못했습니다." % index)

        self.h, self.w = frame.shape[:2]
        self.fps_reported = self.cap.get(cv2.CAP_PROP_FPS)
        self._t0 = time.time()
        self._n = 0
        self.fps_measured = 0.0

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._n += 1
            dt = time.time() - self._t0
            if dt >= 1.0:
                self.fps_measured = self._n / dt
                self._t0 = time.time()
                self._n = 0
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ============================================================================
# 품질 평가
# ============================================================================

def blur_score(gray, ch_c):
    """보드 영역에서의 Laplacian 분산. 값이 클수록 선명."""
    if ch_c is None or len(ch_c) < 4:
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    pts = ch_c.reshape(-1, 2)
    x0 = max(int(pts[:, 0].min()) - 5, 0)
    y0 = max(int(pts[:, 1].min()) - 5, 0)
    x1 = min(int(pts[:, 0].max()) + 5, gray.shape[1])
    y1 = min(int(pts[:, 1].max()) + 5, gray.shape[0])
    if x1 - x0 < 20 or y1 - y0 < 20:
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    return cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()


def pose_feature(ch_c, w, h):
    """자세 다양성 판별용 특징 벡터.
    중심 위치 + 크기 + 종횡비 + 주축 방향 (전부 0~1 정규화)"""
    pts = ch_c.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
    scale = np.sqrt(max(area, 1.0)) / float(w)

    centered = pts - np.array([cx, cy])
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-9)
    v = evecs[:, order[0]]
    aspect = float(np.sqrt(evals[1] / evals[0]))       # 0=심한 기울기, 1=정면
    ang = np.arctan2(v[1], v[0]) * 2.0                 # 축 방향(180도 주기)
    return np.array([cx / w, cy / h, scale, aspect,
                     np.cos(ang) * 0.5, np.sin(ang) * 0.5], dtype=np.float64)


def coverage_cells(ch_c, w, h):
    """코너들이 점유한 커버리지 셀 인덱스 집합."""
    cells = set()
    for x, y in ch_c.reshape(-1, 2):
        gx = min(int(x / w * COV_GRID_X), COV_GRID_X - 1)
        gy = min(int(y / h * COV_GRID_Y), COV_GRID_Y - 1)
        if gx >= 0 and gy >= 0:
            cells.add((gx, gy))
    return cells


# ============================================================================
# 캡처 저장소
# ============================================================================

class ViewStore:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.obj = []          # list of Nx1x3
        self.img = []          # list of Nx1x2
        self.feats = []        # list of feature vector
        self.frames = []       # list of BGR frame (저장용)
        self.n_corners = []
        self.coverage = np.zeros((COV_GRID_Y, COV_GRID_X), dtype=np.int32)

    def __len__(self):
        return len(self.obj)

    def add(self, obj, img, feat, frame, cells):
        self.obj.append(obj)
        self.img.append(img)
        self.feats.append(feat)
        self.frames.append(frame.copy())
        self.n_corners.append(len(obj))
        for gx, gy in cells:
            self.coverage[gy, gx] += 1

    def pop_last(self):
        if not self.obj:
            return
        self.obj.pop(); self.img.pop(); self.feats.pop()
        self.frames.pop(); self.n_corners.pop()
        self._rebuild_coverage()

    def keep(self, indices):
        keep = sorted(indices)
        self.obj = [self.obj[i] for i in keep]
        self.img = [self.img[i] for i in keep]
        self.feats = [self.feats[i] for i in keep]
        self.frames = [self.frames[i] for i in keep]
        self.n_corners = [self.n_corners[i] for i in keep]
        self._rebuild_coverage()

    def clear(self):
        self.__init__(self.w, self.h)

    def _rebuild_coverage(self):
        self.coverage[:] = 0
        for img in self.img:
            for gx, gy in coverage_cells(img, self.w, self.h):
                self.coverage[gy, gx] += 1

    def novelty(self, feat):
        """기존 뷰들과의 최소 거리. 클수록 새로운 자세."""
        if not self.feats:
            return 1e9
        d = np.linalg.norm(np.asarray(self.feats) - feat, axis=1)
        return float(d.min())

    def coverage_ratio(self):
        return float((self.coverage > 0).sum()) / (COV_GRID_X * COV_GRID_Y)


# ============================================================================
# 캘리브레이션
# ============================================================================

def build_flags(args):
    flags = 0
    names = []
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3; names.append("FIX_K3")
    if args.zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST; names.append("ZERO_TANGENT_DIST")
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL; names.append("RATIONAL_MODEL")
    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO; names.append("FIX_ASPECT_RATIO")
    return flags, names


def run_calibration(store, image_size, flags):
    """returns dict(K, dist, rms, per_view, rvecs, tvecs) 또는 None"""
    if len(store) < MIN_VIEWS_FOR_CALIB:
        print("[!] 뷰가 부족합니다. 최소 %d개 필요 (현재 %d개)"
              % (MIN_VIEWS_FOR_CALIB, len(store)))
        return None

    init_K = None
    init_d = None
    if flags & cv2.CALIB_FIX_ASPECT_RATIO:
        init_K = np.eye(3, dtype=np.float64)
        init_K[0, 0] = init_K[1, 1] = float(max(image_size))
        init_K[0, 2] = image_size[0] / 2.0
        init_K[1, 2] = image_size[1] / 2.0

    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            store.obj, store.img, image_size, init_K, init_d, flags=flags)
    except cv2.error as e:
        print("[!] calibrateCamera 실패: %s" % e)
        return None

    per_view = []
    for i in range(len(store)):
        proj, _ = cv2.projectPoints(store.obj[i], rvecs[i], tvecs[i], K, dist)
        diff = proj.reshape(-1, 2) - store.img[i].reshape(-1, 2)
        per_view.append(float(np.sqrt((diff ** 2).sum() / len(diff))))

    return dict(K=K, dist=dist.ravel(), rms=float(rms),
                per_view=per_view, rvecs=rvecs, tvecs=tvecs)


def prune(store, result, sigma=2.0):
    """재투영 오차가 큰 뷰를 제거. 제거된 개수 반환."""
    e = np.asarray(result["per_view"])
    thr = e.mean() + sigma * e.std()
    keep = [i for i in range(len(e)) if e[i] <= thr]
    removed = len(e) - len(keep)
    if removed == 0 or len(keep) < MIN_VIEWS_FOR_CALIB:
        return 0
    store.keep(keep)
    return removed


def fov_deg(K, w, h):
    fx, fy = K[0, 0], K[1, 1]
    return (float(np.degrees(2 * np.arctan(w / (2 * fx)))),
            float(np.degrees(2 * np.arctan(h / (2 * fy)))))


def save_json(path, result, store, image_size, charuco, meta):
    K = result["K"]
    dist = result["dist"]
    fh, fv = fov_deg(K, image_size[0], image_size[1])
    data = {
        "img_w": int(image_size[0]),
        "img_h": int(image_size[1]),
        "K": [[float(v) for v in row] for row in K],
        "dist": [float(v) for v in dist],
        "rms_px": float(result["rms"]),
        "n_images": int(len(store)),
        "square_mm": float(charuco.square_m * 1000.0),
        "note": "ChArUco intrinsics calibration (with distortion).",
        "marker_mm": float(charuco.marker_m * 1000.0),
        "board": {
            "squares_x": charuco.squares_x,
            "squares_y": charuco.squares_y,
            "dictionary": charuco.dict_name,
            "legacy_pattern": charuco.legacy,
        },
        "fov_deg": {"horizontal": fh, "vertical": fv},
        "per_view_rms_px": [float(v) for v in result["per_view"]],
        "mean_corners_per_view": float(np.mean(store.n_corners)),
        "coverage_ratio": float(store.coverage_ratio()),
        "calib_flags": meta.get("flag_names", []),
        "capture": meta.get("capture", {}),
        "opencv_version": cv2.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def print_report(result, store, image_size, path=None):
    K, dist = result["K"], result["dist"]
    e = np.asarray(result["per_view"])
    fh, fv = fov_deg(K, image_size[0], image_size[1])
    print("\n" + "=" * 62)
    print(" 캘리브레이션 결과   (%d x %d, %d views)"
          % (image_size[0], image_size[1], len(store)))
    print("=" * 62)
    print(" 전체 RMS      : %.5f px" % result["rms"])
    print(" 뷰별 오차     : min %.4f / mean %.4f / max %.4f px"
          % (e.min(), e.mean(), e.max()))
    print(" fx, fy        : %.4f, %.4f" % (K[0, 0], K[1, 1]))
    print(" cx, cy        : %.4f, %.4f" % (K[0, 2], K[1, 2]))
    print(" 화각          : H %.2f deg / V %.2f deg" % (fh, fv))
    print(" 왜곡계수      : %s"
          % ", ".join("%.6f" % v for v in dist))
    print(" 커버리지      : %.1f %%" % (store.coverage_ratio() * 100))
    print(" 평균 코너수   : %.1f" % float(np.mean(store.n_corners)))

    # 간단한 품질 판정
    verdict = []
    if result["rms"] > 0.6:
        verdict.append("RMS 가 높습니다(>0.6px). 블러/오검출 뷰를 P 로 프루닝하세요.")
    if store.coverage_ratio() < 0.75:
        verdict.append("화면 커버리지 부족(<75%). 가장자리·모서리를 더 촬영하세요.")
    if abs(K[0, 2] - image_size[0] / 2) > image_size[0] * 0.15 or \
       abs(K[1, 2] - image_size[1] / 2) > image_size[1] * 0.15:
        verdict.append("주점(cx,cy)이 중심에서 15%% 이상 벗어났습니다. 자세 다양성을 늘리세요.")
    if len(store) < 20:
        verdict.append("뷰 수가 적습니다(<20). 30~60장 권장.")
    if verdict:
        print("-" * 62)
        for v in verdict:
            print(" [!] " + v)
    else:
        print("-" * 62)
        print(" [OK] 품질 지표 양호.")
    if path:
        print(" 저장됨        : %s" % path)
    print("=" * 62 + "\n")


# ============================================================================
# 화면 오버레이
# ============================================================================

def draw_coverage(canvas, coverage):
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    mx = max(int(coverage.max()), 1)
    for gy in range(COV_GRID_Y):
        for gx in range(COV_GRID_X):
            x0 = int(gx * w / COV_GRID_X); x1 = int((gx + 1) * w / COV_GRID_X)
            y0 = int(gy * h / COV_GRID_Y); y1 = int((gy + 1) * h / COV_GRID_Y)
            c = coverage[gy, gx]
            if c == 0:
                col = (40, 40, 40)
            else:
                t = c / float(mx)
                col = (int(60 + 40 * t), int(80 + 150 * t), int(60 + 40 * t))
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), col, -1)
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (25, 25, 25), 1)
    cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0, canvas)
    return canvas


def put(canvas, text, x, y, color=C_TXT, scale=0.46, thick=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def draw_hud(canvas, state):
    h, w = canvas.shape[:2]
    bar = canvas[:74, :].copy()
    cv2.rectangle(bar, (0, 0), (w, 74), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.62, canvas[:74, :], 0.38, 0, canvas[:74, :])

    put(canvas, "cam %d  %dx%d  %.0f fps" %
        (state["cam"], w, h, state["fps"]), 8, 18)
    put(canvas, "views %d   coverage %.0f%%" %
        (state["n_views"], state["coverage"] * 100), 8, 36,
        C_OK if state["n_views"] >= MIN_VIEWS_FOR_CALIB else C_WARN)

    if state["detected"]:
        col = C_OK if state["gate_ok"] else C_WARN
        put(canvas, "corners %d/%d  blur %.0f  novelty %.3f" %
            (state["n_corners"], state["max_corners"],
             state["blur"], state["novelty"]), 8, 54, col)
        put(canvas, state["reason"], 8, 70,
            C_OK if state["gate_ok"] else C_WARN, scale=0.42)
    else:
        put(canvas, "board not detected", 8, 54, C_BAD)

    right = "AUTO" if state["auto"] else "MANUAL"
    put(canvas, right, w - 92, 18, C_OK if state["auto"] else C_DIM)
    if state["rms"] is not None:
        put(canvas, "RMS %.4f px" % state["rms"], w - 156, 36, C_OK)
    if state["undistort"]:
        put(canvas, "UNDISTORTED", w - 156, 54, C_WARN)
    put(canvas, "SPACE cap | A auto | C calib | S save | Q quit",
        8, h - 10, C_DIM, scale=0.42)

    if state["flash"] > 0:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), C_OK, 6)
    return canvas


# ============================================================================
# 부가 모드
# ============================================================================

def mode_list_cameras(args):
    """장치 index x 백엔드 조합을 훑어 실제로 열리는 것을 찾는다."""
    print("[*] 카메라 탐색 중... (몇 초 걸립니다)\n")
    order = ["msmf", "dshow", "any"]
    found = []
    print("  idx  backend  결과")
    print("  " + "-" * 46)
    for idx in range(int(args.list_max) + 1):
        for bname in order:
            api = BACKENDS.get(bname, cv2.CAP_ANY)
            cap = None
            try:
                cap = cv2.VideoCapture(idx, api)
                if not cap.isOpened():
                    print("  %3d  %-7s  열리지 않음" % (idx, bname))
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("  %3d  %-7s  열리나 프레임 없음" % (idx, bname))
                    continue
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print("  %3d  %-7s  OK  %dx%d  %.0f fps"
                      % (idx, bname, w, h, fps))
                found.append((idx, bname, w, h))
            except Exception as e:
                print("  %3d  %-7s  예외: %s" % (idx, bname, e))
            finally:
                if cap is not None:
                    cap.release()
    print()
    if not found:
        print("  사용 가능한 카메라를 찾지 못했습니다. 아래를 확인하세요:")
        print("   1. 다른 프로그램(카메라 앱, 화상회의, 이 스크립트의 다른 창)이")
        print("      점유 중인지 -> 모두 종료 후 재시도")
        print("   2. Windows 설정 > 개인 정보 및 보안 > 카메라")
        print("      -> '앱이 카메라에 액세스하도록 허용' 및")
        print("         '데스크톱 앱이 카메라에 액세스하도록 허용' 켜기")
        print("   3. USB 케이블/포트 교체 (USB3 포트 권장)")
        print("   4. pip uninstall opencv-python-headless")
        print("      (headless 판이 설치되어 있으면 캡처가 동작하지 않습니다)")
        return 1
    print("  사용 예:")
    for idx, bname, w, h in found[:4]:
        print("    python calibrate_intrinsics.py --cam %d --backend %s"
              % (idx, bname))
    return 0


def mode_generate_board(args):
    ch = Charuco(args.squares_x, args.squares_y,
                 args.square_mm / 1000.0, args.marker_mm / 1000.0,
                 args.dict, args.legacy)
    img = ch.generate_image(args.board_px, args.board_margin_px)
    cv2.imwrite(args.generate_board, img)
    bw = args.squares_x * args.square_mm
    bh = args.squares_y * args.square_mm
    print("[OK] 보드 이미지 저장: %s" % args.generate_board)
    print("     %d x %d squares, square %.1f mm, marker %.1f mm, %s"
          % (args.squares_x, args.squares_y, args.square_mm,
             args.marker_mm, args.dict))
    print("     실제 인쇄 크기: %.1f x %.1f mm (여백 제외)" % (bw, bh))
    print("     ! 인쇄 후 반드시 자로 사각형 한 변을 재서 square_mm 을 실측값으로 보정하세요.")
    print("     ! 반드시 '실제 크기(100%)' 로 인쇄하고, 평평한 판에 부착하세요.")
    return 0


def mode_auto_dict(args):
    """여러 딕셔너리를 순회하며 마커가 가장 많이 검출되는 것을 찾음."""
    print("[*] 딕셔너리 자동 탐색 시작. 보드를 카메라에 크게 비춰 주세요.")
    print("    (10초간 스캔합니다)")
    cam = CameraStream(args.cam, args.width, args.height, args.fps,
                       args.fourcc, args.backend, args.exposure)
    scores = {name: 0 for name in DICT_NAMES}
    ids_seen = {name: set() for name in DICT_NAMES}
    dicts = {}
    for name in DICT_NAMES:
        did = getattr(cv2.aruco, name, None)
        if did is None:
            continue
        dicts[name] = (cv2.aruco.getPredefinedDictionary(did) if NEW_API
                       else cv2.aruco.Dictionary_get(did))
    params = (cv2.aruco.DetectorParameters() if NEW_API
              else cv2.aruco.DetectorParameters_create())
    detectors = ({name: cv2.aruco.ArucoDetector(d, params)
                  for name, d in dicts.items()} if NEW_API else {})

    t_end = time.time() + 10.0
    try:
        while time.time() < t_end:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for name, d in dicts.items():
                if NEW_API:
                    corners, ids, _ = detectors[name].detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, d, parameters=params)
                if ids is not None and len(ids) > 0:
                    scores[name] += len(ids)
                    ids_seen[name].update(int(i) for i in ids.flatten())
            vis = frame.copy()
            best = max(scores, key=scores.get)
            put(vis, "scanning... best: %s (%d)" % (best, scores[best]),
                8, 20, C_OK)
            put(vis, "%.1f s left" % (t_end - time.time()), 8, 40)
            cv2.imshow("auto-dict", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 56)
    print(" 딕셔너리 탐색 결과 (검출 누적 마커 수)")
    print("=" * 56)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    for name, sc in ranked[:6]:
        if sc == 0:
            continue
        ids = sorted(ids_seen[name])
        print("  %-22s %6d   고유 id %d개, 범위 %s~%s"
              % (name, sc, len(ids),
                 ids[0] if ids else "-", ids[-1] if ids else "-"))
    if ranked[0][1] == 0:
        print("  검출 실패. 조명/초점/보드 인쇄 상태를 확인하세요.")
        return 1
    top = ranked[0][0]
    n_ids = len(ids_seen[top])
    print("-" * 56)
    print("  추천: --dict %s" % top)
    print("  고유 마커 %d개 -> 보드 사각형 수는 대략 %d개"
          % (n_ids, n_ids * 2))
    print("  (ChArUco 는 전체 칸의 절반에만 마커가 들어갑니다.")
    print("   예: 5x7=35칸 -> 마커 17개, 6x9=54칸 -> 마커 27개)")
    print("=" * 56)
    return 0


def mode_from_dir(args):
    files = sorted([os.path.join(args.from_dir, f)
                    for f in os.listdir(args.from_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    if not files:
        print("[!] 이미지가 없습니다: %s" % args.from_dir)
        return 1
    ch = Charuco(args.squares_x, args.squares_y,
                 args.square_mm / 1000.0, args.marker_mm / 1000.0,
                 args.dict, args.legacy)
    first = cv2.imread(files[0])
    if first is None:
        print("[!] 이미지를 읽을 수 없습니다: %s" % files[0])
        return 1
    h, w = first.shape[:2]
    store = ViewStore(w, h)

    print("[*] %d개 이미지 처리 중..." % len(files))
    for f in files:
        im = cv2.imread(f)
        if im is None or im.shape[0] != h or im.shape[1] != w:
            print("    skip (크기 불일치): %s" % os.path.basename(f))
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ch_c, ch_i, _, _ = ch.detect(gray)
        obj, img = ch.match_points(ch_c, ch_i)
        if obj is None:
            print("    skip (검출 실패): %s" % os.path.basename(f))
            continue
        store.add(obj, img, pose_feature(img, w, h), im,
                  coverage_cells(img, w, h))
    print("[*] 유효 뷰 %d개" % len(store))

    flags, flag_names = build_flags(args)
    result = run_calibration(store, (w, h), flags)
    if result is None:
        return 1
    if args.auto_prune:
        n = prune(store, result, args.prune_sigma)
        if n:
            print("[*] 불량 뷰 %d개 제거 후 재계산" % n)
            r2 = run_calibration(store, (w, h), flags)
            if r2 is not None:
                result = r2
    out = args.out or os.path.join(args.from_dir, "intrinsics.json")
    save_json(out, result, store, (w, h), ch,
              dict(flag_names=flag_names, capture={"source": args.from_dir}))
    print_report(result, store, (w, h), out)
    return 0


# ============================================================================
# 메인 대화형 캘리브레이션
# ============================================================================

def mode_live(args):
    ch = Charuco(args.squares_x, args.squares_y,
                 args.square_mm / 1000.0, args.marker_mm / 1000.0,
                 args.dict, args.legacy)
    max_corners = ch.n_chessboard_corners
    min_corners = max(args.min_corners, MIN_CORNERS_PER_VIEW)
    if min_corners > max_corners:
        min_corners = max(MIN_CORNERS_PER_VIEW, max_corners // 2)

    cam = CameraStream(args.cam, args.width, args.height, args.fps,
                       args.fourcc, args.backend, args.exposure)
    w, h = cam.w, cam.h
    print("[*] 카메라 %d 열림: %dx%d, 요청 %d fps, 보고 %.1f fps"
          % (args.cam, w, h, args.fps, cam.fps_reported))
    if (w, h) != (args.width, args.height):
        print("[!] 요청 해상도(%dx%d)와 실제(%dx%d)가 다릅니다. 실제값으로 진행합니다."
              % (args.width, args.height, w, h))

    save_dir = args.save_images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    store = ViewStore(w, h)
    flags, flag_names = build_flags(args)

    auto = args.auto
    show_cov = True
    show_undist = False
    result = None
    flash = 0
    last_auto_t = 0.0
    win = "intrinsics calib - cam %d" % args.cam
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w * 2, 1280), min(h * 2, 960))

    map1 = map2 = None
    print_help()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ch_c, ch_i, m_c, m_i = ch.detect(gray)
            obj, img = ch.match_points(ch_c, ch_i)

            detected = obj is not None
            n_c = len(obj) if detected else 0
            blur = blur_score(gray, ch_c) if detected else 0.0
            feat = pose_feature(img, w, h) if detected else None
            nov = store.novelty(feat) if detected else 0.0

            # ---- 품질 게이트 ----
            gate_ok, reason = False, ""
            if not detected:
                reason = "-"
            elif n_c < min_corners:
                reason = "corners too few (need >= %d)" % min_corners
            elif blur < args.min_blur:
                reason = "too blurry (need >= %.0f)" % args.min_blur
            elif nov < args.min_novelty:
                reason = "pose too similar (need >= %.3f)" % args.min_novelty
            else:
                gate_ok, reason = True, "READY"

            # ---- 표시 프레임 ----
            if show_undist and map1 is not None:
                canvas = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            else:
                canvas = frame.copy()
                if show_cov:
                    draw_coverage(canvas, store.coverage)
                if m_i is not None and len(m_i) > 0:
                    cv2.aruco.drawDetectedMarkers(canvas, m_c, m_i)
                if detected:
                    col = C_OK if gate_ok else C_WARN
                    for p in img.reshape(-1, 2):
                        cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))),
                                   3, col, -1, cv2.LINE_AA)
                    hull = cv2.convexHull(img.reshape(-1, 2).astype(np.int32))
                    cv2.polylines(canvas, [hull], True, col, 1, cv2.LINE_AA)

            state = dict(cam=args.cam, fps=cam.fps_measured,
                         n_views=len(store), coverage=store.coverage_ratio(),
                         detected=detected, n_corners=n_c,
                         max_corners=max_corners, blur=blur, novelty=nov,
                         gate_ok=gate_ok, reason=reason, auto=auto,
                         rms=(result["rms"] if result else None),
                         undistort=show_undist, flash=flash)
            draw_hud(canvas, state)
            cv2.imshow(win, canvas)
            flash = max(flash - 1, 0)

            # ---- 자동 캡처 ----
            now = time.time()
            do_capture = False
            if auto and gate_ok and (now - last_auto_t) >= args.auto_interval:
                do_capture = True
                last_auto_t = now

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if detected and n_c >= min_corners:
                    do_capture = True
                else:
                    print("[!] 캡처 불가: %s" % reason)
            elif key in (ord('q'), 27):
                break
            elif key == ord('a'):
                auto = not auto
                print("[*] 자동 캡처 %s" % ("ON" if auto else "OFF"))
            elif key == ord('m'):
                show_cov = not show_cov
            elif key == ord('d'):
                store.pop_last()
                print("[*] 마지막 캡처 삭제 (남은 %d개)" % len(store))
            elif key == ord('r'):
                store.clear(); result = None; map1 = map2 = None
                show_undist = False
                print("[*] 전체 초기화")
            elif key == ord('h'):
                print_help()
            elif key == ord('c'):
                result = run_calibration(store, (w, h), flags)
                if result:
                    print_report(result, store, (w, h))
                    map1, map2 = cv2.initUndistortRectifyMap(
                        result["K"], result["dist"], None, result["K"],
                        (w, h), cv2.CV_16SC2)
            elif key == ord('p'):
                if result is None:
                    print("[!] 먼저 C 로 캘리브레이션을 실행하세요.")
                else:
                    n = prune(store, result, args.prune_sigma)
                    if n == 0:
                        print("[*] 제거할 불량 뷰가 없습니다.")
                    else:
                        print("[*] 불량 뷰 %d개 제거, 재계산 중..." % n)
                        r2 = run_calibration(store, (w, h), flags)
                        if r2:
                            result = r2
                            print_report(result, store, (w, h))
                            map1, map2 = cv2.initUndistortRectifyMap(
                                result["K"], result["dist"], None,
                                result["K"], (w, h), cv2.CV_16SC2)
            elif key == ord('u'):
                if map1 is None:
                    print("[!] 먼저 C 로 캘리브레이션을 실행하세요.")
                else:
                    show_undist = not show_undist
            elif key == ord('s'):
                if result is None:
                    print("[!] 먼저 C 로 캘리브레이션을 실행하세요.")
                else:
                    out = args.out or "intrinsics_cam%d.json" % args.cam
                    meta = dict(flag_names=flag_names, capture=dict(
                        requested=[args.width, args.height, args.fps],
                        actual=[w, h],
                        fps_measured=round(cam.fps_measured, 2),
                        fourcc=args.fourcc, backend=args.backend))
                    save_json(out, result, store, (w, h), ch, meta)
                    print("[OK] 저장됨: %s" % out)
                    if save_dir:
                        for i, fr in enumerate(store.frames):
                            cv2.imwrite(os.path.join(
                                save_dir, "view_%03d.png" % i), fr)
                        print("[OK] 이미지 %d장 저장: %s"
                              % (len(store.frames), save_dir))

            if do_capture and detected:
                store.add(obj, img, feat, frame, coverage_cells(img, w, h))
                flash = 4
                print("[+] capture #%d  corners=%d blur=%.0f novelty=%.3f "
                      "coverage=%.0f%%"
                      % (len(store), n_c, blur, nov,
                         store.coverage_ratio() * 100))
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if result is not None and not args.no_autosave:
        out = args.out or "intrinsics_cam%d.json" % args.cam
        if not os.path.exists(out):
            meta = dict(flag_names=flag_names, capture=dict(
                requested=[args.width, args.height, args.fps],
                actual=[w, h], fourcc=args.fourcc, backend=args.backend))
            save_json(out, result, store, (w, h), ch, meta)
            print("[OK] 종료 시 자동 저장: %s" % out)
    return 0


def print_help():
    print("""
--------------------------------------------------------------
  SPACE 캡처 | A 자동캡처 | C 캘리브레이션 | P 불량뷰 프루닝
  U 왜곡보정 프리뷰 | M 히트맵 | D 마지막삭제 | R 초기화
  S 저장 | H 도움말 | Q/ESC 종료

  촬영 요령
   1. 보드가 화면의 1/3 ~ 2/3 를 채우도록
   2. 화면의 네 모서리와 가장자리를 반드시 포함 (왜곡계수 정확도 직결)
   3. 좌/우/상/하로 30~45도 기울인 자세를 골고루
   4. 보드 전체가 초점 안에 들어오도록, 흔들리면 잠시 정지
   5. 30~60장 권장. 히트맵이 전부 초록이 되면 충분
--------------------------------------------------------------""")


# ============================================================================
# 엔트리 포인트
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ChArUco 내부 파라미터 캘리브레이션 (탁구 로봇 프로젝트)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("보드 스펙 (반드시 실제 인쇄물과 일치시킬 것)")
    g.add_argument("--squares-x", type=int, default=5, help="가로 사각형 개수")
    g.add_argument("--squares-y", type=int, default=7, help="세로 사각형 개수")
    g.add_argument("--square-mm", type=float, default=33.0, help="사각형 한 변 (mm)")
    g.add_argument("--marker-mm", type=float, default=25.0, help="ArUco 마커 한 변 (mm)")
    g.add_argument("--dict", default="DICT_4X4_50", choices=DICT_NAMES,
                   help="ArUco 딕셔너리")
    g.add_argument("--legacy", action="store_true",
                   help="OpenCV 4.6 이전에 생성한 보드일 경우 지정")

    g = p.add_argument_group("카메라")
    g.add_argument("--cam", type=int, default=0, help="카메라 인덱스")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--fps", type=int, default=120)
    g.add_argument("--fourcc", default="MJPG", help="빈 문자열이면 설정 생략")
    g.add_argument("--backend", default="dshow",
                   choices=list(BACKENDS.keys()))
    g.add_argument("--exposure", type=float, default=None,
                   help="수동 노출값 (백엔드 의존, 미지정시 자동)")

    g = p.add_argument_group("캡처 품질 게이트")
    g.add_argument("--min-corners", type=int, default=12)
    g.add_argument("--min-blur", type=float, default=60.0,
                   help="Laplacian 분산 하한. 조명이 어두우면 낮추세요")
    g.add_argument("--min-novelty", type=float, default=0.10,
                   help="기존 뷰와의 최소 자세 거리")
    g.add_argument("--auto", action="store_true", help="자동 캡처로 시작")
    g.add_argument("--auto-interval", type=float, default=0.7,
                   help="자동 캡처 최소 간격 (초)")

    g = p.add_argument_group("캘리브레이션 옵션")
    g.add_argument("--fix-k3", action="store_true")
    g.add_argument("--zero-tangent", action="store_true")
    g.add_argument("--rational", action="store_true",
                   help="8계수 rational 모델 사용")
    g.add_argument("--fix-aspect", action="store_true", help="fx=fy 강제")
    g.add_argument("--auto-prune", action="store_true",
                   help="--from-dir 모드에서 자동 프루닝")
    g.add_argument("--prune-sigma", type=float, default=2.0)

    g = p.add_argument_group("입출력 / 모드")
    g.add_argument("--out", default=None, help="출력 JSON 경로")
    g.add_argument("--save-images", default=None,
                   help="S 저장 시 캡처 이미지도 저장할 폴더")
    g.add_argument("--from-dir", default=None,
                   help="라이브 대신 이미지 폴더로 캘리브레이션")
    g.add_argument("--generate-board", default=None,
                   help="인쇄용 보드 이미지 경로 (예: board.png)")
    g.add_argument("--board-px", type=int, default=1600)
    g.add_argument("--board-margin-px", type=int, default=40)
    g.add_argument("--auto-dict", action="store_true",
                   help="딕셔너리 자동 탐색 후 종료")
    g.add_argument("--list-cameras", action="store_true",
                   help="연결된 카메라와 사용 가능한 백엔드를 탐색 후 종료")
    g.add_argument("--list-max", type=int, default=4,
                   help="--list-cameras 로 탐색할 최대 index")
    g.add_argument("--no-autosave", action="store_true",
                   help="종료 시 자동 저장 하지 않음")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("[*] OpenCV %s (ArUco %s API)"
          % (cv2.__version__, "new" if NEW_API else "legacy"))
    try:
        if args.list_cameras:
            return mode_list_cameras(args)
        if args.generate_board:
            return mode_generate_board(args)
        if args.auto_dict:
            return mode_auto_dict(args)
        if args.from_dir:
            return mode_from_dir(args)
        return mode_live(args)
    except KeyboardInterrupt:
        print("\n[*] 중단됨.")
        return 130
    except Exception as e:
        print("[!] 오류: %s" % e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())