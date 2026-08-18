#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_ball_robot.py
===================
탁구 로봇 통합 실행기 — 비전 · 궤적 예측 · 로봇 모션을 **창 하나**에 담는다.

화면 (1920 x 1080 기준)
-----------------------
    왼쪽    cam0 / cam1 미리보기, 그 아래 콘솔 로그
    가운데  3D 뷰 (960 x 880) / 궤적 예측 — 탭으로 전환
    오른쪽  측정값 · 예측 · 로봇 상태
    아래    조작 바 (보기 / 검출 / 예측 / 로봇 / 보정)

track_ball.py 는 수정하지 않는다.  TrackerApp 을 상속해 화면 구성(_build_ui)
만 새로 짜고, 검출·삼각측량·칼만·3D 렌더링은 원본 것을 그대로 쓴다.

실행
----
    python track_ball_robot.py            # 카메라 사용
    python track_ball_robot.py --sim      # 카메라 없이 합성 공으로 점검

키
--
    a  무장/해제     e  비상정지     space 일시정지
    t  궤적 지우기   m  마스크       r  기록      p  공 색 찍기
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import track_ball as tb
from trajectory_predictor import (PredictorRenderer, Z_MAX_M as Z_TOP_M,
                                  Z_MIN_M as Z_BOT_M, _to_photo)

import app_profile
import robot_config as cfg
import robot_tuning as tuning
from turn_state import HUMAN, ROBOT, TurnState
from linear_link import find_linear_files
from motion_control import MotionWorker, SimBackend
from robot_kinematics import ArmKinematics, swing_lead_sec
from shot_planner import ShotPlanner


BG = "#1e1e22"
FG = "#e8e8ec"
ACC = "#4ea3ff"
OK_C = "#4fd07a"
WARN_C = "#ffc451"
BAD_C = "#ff6b6b"
DIM = "#8b8b95"
PANEL = "#26262c"
CONSOLE_BG = "#141418"
FONT = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 9, "bold")
FONT_M = ("Consolas", 9)
FONT_BIG = ("Consolas", 13)

PREVIEW_SCALE = 0.50        # 320 x 240 — 두 개 + 콘솔이 세로에 들어간다
CONSOLE_LINES = 400

#  ---- 화면 크기에 맞춰 자동으로 잡는다 ----
#  창을 띄우는 화면의 해상도를 보고 3D 뷰 크기를 정한다.  1080p 에서
#  아래가 잘리던 문제 때문에, 고정 크기 대신 남는 공간을 계산해서 쓴다.
#
#    세로에서 3D 뷰 말고 들어가는 것들:
#      제목 표시줄 ~32 · 탭 머리 ~30 · 프리셋 버튼 줄 ~34
#      하단 조작 바 ~170 · 상태줄 ~22 · 여백/테두리 ~24 · 작업 표시줄 ~48
CHROME_H = 408
CHROME_W = 60               # 좌우 여백/테두리
RIGHT_COL_W = 360           # 오른쪽 상태 패널
BOTTOM_BAR_H = 224

VIEW3D_MIN_H, VIEW3D_MAX_H = 380, 900
VIEW3D_MIN_W, VIEW3D_MAX_W = 620, 1200


# ============================================================================
# 콘솔 가로채기 — print 를 화면 로그에도 보낸다
# ============================================================================

class _ConsoleTee:
    """stdout 을 원래 자리와 큐 양쪽으로 보낸다.

    모션 스레드에서도 print 하므로 Tk 위젯을 직접 건드리면 안 된다.
    큐에 넣고 메인 루프에서 꺼내 그린다.
    """

    def __init__(self, original, q):
        self.original = original
        self.q = q

    def write(self, text):
        try:
            self.original.write(text)
        except Exception:
            pass
        try:
            self.q.put_nowait(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


# ============================================================================
# 탐지 영역만 잘라서 검출하는 껍데기
# ============================================================================

class CropDetector:
    """원본 BallDetector 를 감싸, ROI 를 감싸는 사각형만 처리하게 한다.

    검출 비용은 화면 넓이에 비례한다.  탐지 영역이 화면의 절반이면 비용도
    절반이 되고, 그만큼 루프가 빨라져 카메라 프레임을 더 많이 쓴다.

    바깥에서 보기에는 원본과 똑같이 동작한다.  후보 좌표는 원래 화면 기준
    으로 되돌리고, 마스크를 요구하면 전체 크기로 되돌려 준다.
    """

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "box", None)

    #  --- 원본 속성 그대로 통과 ---
    @property
    def p(self):
        return self.inner.p

    @p.setter
    def p(self, v):
        self.inner.p = v

    @property
    def stats(self):
        return self.inner.stats

    def set(self, key, value):
        self.inner.set(key, value)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "inner"), name)

    #  --- 자르기 ---
    def set_box(self, box):
        object.__setattr__(self, "box", box)

    def detect(self, bgr, return_mask=False, roi_mask=None, roi_poly=None):
        box = self.box
        if box is None or roi_mask is None or not cfg.ROI_CROP:
            return self.inner.detect(bgr, return_mask, roi_mask, roi_poly)
        x0, y0, w, h = box
        sub = bgr[y0:y0 + h, x0:x0 + w]
        sub_mask = roi_mask[y0:y0 + h, x0:x0 + w]
        sub_poly = None
        if roi_poly is not None:
            sub_poly = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2) \
                - np.array([x0, y0], dtype=np.float32)
        out, m = self.inner.detect(sub, return_mask, sub_mask, sub_poly)
        for c in out:                       # 좌표를 원래 화면 기준으로
            c["uv"] = (c["uv"][0] + x0, c["uv"][1] + y0)
        if return_mask and m is not None:
            full = np.zeros(bgr.shape[:2], np.uint8)
            full[y0:y0 + h, x0:x0 + w] = m
            m = full
        return out, m


# ============================================================================
# 통합 앱
# ============================================================================

class RobotTrackerApp(tb.TrackerApp):

    def __init__(self, args):
        self.worker = None
        self.planner = None
        self.fit = None
        self._log_q = queue.Queue()
        self._stdout_backup = sys.stdout
        sys.stdout = _ConsoleTee(sys.stdout, self._log_q)
        super().__init__(args)
        self.title("탁구 로봇 — 비전 · 예측 · 모션")

    # ------------------------------------------------------------------
    #  화면 구성 (원본 _build_ui 를 완전히 대체)
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.preview_scale = PREVIEW_SCALE
        self._layout_for_screen()

        #  ---- 로봇 계층 (프레임 루프가 돌기 전에 준비돼야 한다) ----
        self.tune = tuning.load()
        tuning.apply_to_config(self.tune)
        self.kin = ArmKinematics(tuning.geometry_from(self.tune))
        self.planner = ShotPlanner(kinematics=self.kin, verbose=True)
        self.worker = MotionWorker(self._make_backend(), kinematics=self.kin,
                                   armed=bool(getattr(self.args, "arm", False)),
                                   verbose=True)
        self.fit = PredictorRenderer(self.geo)
        line_m = (float(self.geo["table_length"]) / 2.0
                  if cfg.TURN_LINE_CM is None
                  else float(cfg.TURN_LINE_CM) / 100.0)
        self.turn = TurnState(line_m, verbose=True)
        self._fed_t = None
        self._last_plan = None
        self._fit_photo = None
        self._fit_last_draw = 0.0
        self._last_render = 0.0
        self._last_panel = 0.0
        self._last_swung = 0
        self._last_pair_key = None
        self._loop_n, self._loop_t0 = 0, time.perf_counter()
        self.loop_hz = 0.0
        self._sample_n, self._sample_t0 = 0, time.perf_counter()
        self.sample_hz = 0.0

        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TButton", font=FONT, padding=3)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", font=FONT, padding=(12, 4))

        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=6, pady=6)
        top = tk.Frame(root, bg=BG)
        top.pack(fill="both", expand=True)

        self._build_left(top)
        self._build_center(top)
        self._build_right(top)
        self._build_bottom(root)

        self.lbl_status = tk.Label(root, bg=BG, fg=DIM, font=FONT, anchor="w")
        self.lbl_status.pack(fill="x", pady=(4, 0))

        #  검출기를 '영역만 잘라 처리하는' 껍데기로 감싼다
        for i in (0, 1):
            if not isinstance(self.detector[i], CropDetector):
                self.detector[i] = CropDetector(self.detector[i])
        self._update_crop_boxes()
        self.on_opt()               # 검출 기본 옵션을 검출기에 반영
        self.worker.start()
        self.apply_decide_line()
        self.apply_swing_span()
        self._sync_link_label()
        self._banner()

    def _layout_for_screen(self):
        """화면 해상도를 보고 3D 뷰와 예측 뷰 크기를 정한다."""
        sw = max(1024, int(self.winfo_screenwidth()))
        sh = max(720, int(self.winfo_screenheight()))
        left_w = int(self.calib[0].w * self.preview_scale) + 8

        h = sh - CHROME_H
        w = sw - left_w - RIGHT_COL_W - CHROME_W
        self.view3d_h = max(VIEW3D_MIN_H, min(VIEW3D_MAX_H, h))
        self.view3d_w = max(VIEW3D_MIN_W, min(VIEW3D_MAX_W, w))
        self.viewer.w, self.viewer.h = self.view3d_w, self.view3d_h
        print("[UI] 화면 %dx%d -> 3D 뷰 %dx%d"
              % (sw, sh, self.view3d_w, self.view3d_h))

    def _resolve_linear_paths(self):
        """리니어 DLL/.mot 경로를 정한다. 설정이 비어 있으면 자동 탐색."""
        cached = getattr(self, "_linear_paths", None)
        if cached is not None:
            return cached
        dll, mot = cfg.LINEAR_DLL_PATH, cfg.LINEAR_MOT_PATH
        if dll is None or mot is None:
            fd, fm = find_linear_files()
            dll = dll or fd
            mot = mot or fm
            if fd:
                print("[UI] 리니어 DLL 자동 탐색: %s" % fd)
                print("[UI] .mot 파일       : %s" % fm)
            else:
                print("[UI] 리니어 DLL 을 찾지 못했습니다. "
                      "robot_config.LINEAR_DLL_PATH 에 경로를 적거나 "
                      "native/ 폴더를 프로젝트 안에 두세요.")
        self._linear_paths = (dll, mot)
        return dll, mot

    def _make_backend(self, *, real: bool | None = None):
        """실기 백엔드를 만든다. 실패하면 시뮬레이터로 떨어진다.

        실기라도 리니어 DLL 을 못 찾으면 그 축만 시뮬레이션으로 돈다.
        (다이나믹셀은 실기, 리니어는 가상 — 팔부터 시험할 때 쓴다)
        """
        a = self.args
        want = getattr(a, "real", True) if real is None else bool(real)
        be = None
        if want:
            from robot_backend import BackendError, RealBackend
            dll, mot = self._resolve_linear_paths()
            try:
                be = RealBackend(
                    port=a.port, baudrate=a.baud,
                    dll_path=dll, mot_path=mot,
                    simulate_dxl=False, simulate_linear=(dll is None),
                    start_cmd=cfg.LINEAR_START_CMD_AT_LEFT_END,
                    verbose=True)
                be.enable_linear = cfg.ENABLE_LINEAR
                be.enable_tilt = cfg.ENABLE_TILT
                be.enable_swing = cfg.ENABLE_SWING
                be.open()
            except (BackendError, Exception) as e:
                messagebox.showerror(
                    "실기 연결 실패",
                    "%s\n\n시뮬레이터로 계속합니다.\n"
                    "고친 뒤 [로봇] 탭의 [실기 연결] 로 다시 시도하세요." % e)
                try:
                    if be is not None and getattr(be, "opened", False):
                        be.close()
                except Exception:
                    pass
                be = None
        if be is None:
            be = SimBackend(verbose=True, realtime=True)
            be.enable_linear = cfg.ENABLE_LINEAR
            be.enable_tilt = cfg.ENABLE_TILT
            be.enable_swing = cfg.ENABLE_SWING
            be.open()
        return be

    # ------------------------------------------------------------- 왼쪽
    def _build_left(self, parent):
        pw = int(self.calib[0].w * self.preview_scale)
        ph = int(self.calib[0].h * self.preview_scale)
        col = tk.Frame(parent, bg=BG, width=pw + 8)
        col.pack(side="left", fill="y")
        col.pack_propagate(False)

        self.cam_canvas, self.cam_label = [], []
        for i in (0, 1):
            tk.Label(col, text="cam%d" % i, bg=BG, fg=ACC, font=FONT_B,
                     anchor="w").pack(fill="x")
            c = tk.Canvas(col, width=pw, height=ph, bg="#000",
                          highlightthickness=1, highlightbackground="#444")
            c.pack()
            c.bind("<Button-1>", lambda e, k=i: self.on_pick(k, e))
            self.cam_canvas.append(c)
            lb = tk.Label(col, text="", bg=BG, fg=DIM, font=FONT_M, anchor="w")
            lb.pack(fill="x", pady=(0, 4))
            self.cam_label.append(lb)

        head = tk.Frame(col, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="콘솔", bg=BG, fg=ACC, font=FONT_B,
                 anchor="w").pack(side="left")
        ttk.Button(head, text="지우기", width=8,
                   command=self.clear_console).pack(side="right")

        box = tk.Frame(col, bg=BG)
        box.pack(fill="both", expand=True)
        sb = tk.Scrollbar(box)
        sb.pack(side="right", fill="y")
        self.txt_console = tk.Text(box, width=1, height=8, bg=CONSOLE_BG,
                                   fg=FG, font=("Consolas", 8),
                                   insertbackground=FG, wrap="none",
                                   yscrollcommand=sb.set,
                                   highlightthickness=1,
                                   highlightbackground="#444")
        self.txt_console.pack(side="left", fill="both", expand=True)
        sb.config(command=self.txt_console.yview)
        self.txt_console.configure(state="disabled")

    # ----------------------------------------------------------- 가운데
    def _build_center(self, parent):
        mid = tk.Frame(parent, bg=BG)
        mid.pack(side="left", fill="both", expand=True, padx=(8, 0))
        nb = ttk.Notebook(mid)
        nb.pack(fill="both", expand=True)
        self.center_nb = nb

        f3d = tk.Frame(nb, bg=BG)
        nb.add(f3d, text="  3D 뷰  ")
        self.canvas3d = tk.Canvas(f3d, width=self.view3d_w,
                                  height=self.view3d_h, bg="#000",
                                  highlightthickness=1,
                                  highlightbackground="#444")
        self.canvas3d.pack()
        vb = tk.Frame(f3d, bg=BG)
        vb.pack(fill="x", pady=(4, 0))
        for name in ("비스듬히", "로봇 시점", "사람 시점", "측면", "상단"):
            ttk.Button(vb, text=name, width=9,
                       command=lambda n=name: self.viewer.set_preset(n)
                       ).pack(side="left", padx=2)

        ffit = tk.Frame(nb, bg=BG)
        nb.add(ffit, text="  궤적 예측  ")
        #  예측 그림이 3D 뷰보다 세로로 길면 줄여서 넣는다
        self.fit_scale = min(1.0, float(self.view3d_h) / self.fit.IH)
        self.fit_w = int(self.fit.IW * self.fit_scale)
        self.fit_h = int(self.fit.IH * self.fit_scale)
        self.canvas_fit = tk.Canvas(ffit, width=self.fit_w,
                                    height=self.fit_h, bg="#000",
                                    highlightthickness=1,
                                    highlightbackground="#444")
        self.canvas_fit.pack()
        self.lbl_fit = tk.Label(ffit, text="", bg=BG, fg=DIM, font=FONT_M,
                                anchor="w", justify="left")
        self.lbl_fit.pack(fill="x", pady=(4, 0))

    # ------------------------------------------------- 값 조절 행 (공용)
    def _value_row(self, parent, var, lo, hi, *, step=1.0, decimals=0,
                   length=90, command=None, label=None, label_width=11):
        """[이름] [-] [입력칸] [+] [슬라이더] 한 줄을 만든다.

        슬라이더만으로는 1 단위를 맞추기 어려워서, 숫자를 직접 넣거나
        버튼으로 한 칸씩 움직일 수 있게 한다.  셋 중 무엇으로 바꾸든
        같은 콜백이 불려 검출기에 바로 반영된다.
        """
        row = tk.Frame(parent, bg=PANEL)
        if label is not None:
            tk.Label(row, text=label, bg=PANEL, fg=DIM, font=FONT,
                     width=label_width, anchor="w").pack(side="left")

        fmt = "%%.%df" % decimals
        txt = tk.StringVar(value=fmt % float(var.get()))
        pending = {"id": None}
        internal = {"on": False}          # 우리가 텍스트를 바꾼 경우 표시

        def set_txt(v):
            internal["on"] = True
            try:
                txt.set(fmt % v)
            finally:
                internal["on"] = False

        def apply(v, *, from_entry=False):
            v = max(lo, min(hi, float(v)))
            if decimals == 0:
                v = round(v)
            var.set(v)
            set_txt(v)
            if command is not None:
                command(v)

        def bump(d):
            apply(float(var.get()) + d * step)

        def commit(_=None):
            if pending.get("id") is not None:
                try:
                    self.after_cancel(pending["id"])
                except Exception:
                    pass
                pending["id"] = None
            raw = txt.get().strip()
            if raw in ("", "-", "+", ".", "-."):
                return
            try:
                apply(float(raw), from_entry=True)
            except ValueError:
                set_txt(float(var.get()))

        #  타이핑이 멈추면 자동으로 반영한다.
        #  Enter 키에만 의존하면 포커스 상황에 따라 안 걸리는 경우가 있다.
        def schedule(*_):
            if pending["id"] is not None:
                try:
                    self.after_cancel(pending["id"])
                except Exception:
                    pass
            pending["id"] = self.after(600, commit)

        ttk.Button(row, text="-", width=2,
                   command=lambda: bump(-1)).pack(side="left")
        e = tk.Entry(row, textvariable=txt, width=5, bg="#15151a", fg=FG,
                     font=FONT_M, insertbackground=FG, justify="right")
        e.pack(side="left", padx=1)
        e.bind("<Return>", commit)
        e.bind("<KP_Enter>", commit)
        e.bind("<FocusOut>", commit)
        e.bind("<KeyRelease>", schedule)
        #  칸을 클릭하면 전체 선택 -> 바로 덮어쓸 수 있다
        e.bind("<FocusIn>", lambda _e: e.select_range(0, "end"))
        ttk.Button(row, text="+", width=2,
                   command=lambda: bump(+1)).pack(side="left")

        sc = tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
                      bg=PANEL, fg=FG, font=("Consolas", 7),
                      highlightthickness=0, troughcolor="#15151a",
                      length=length, sliderlength=14, width=9,
                      showvalue=False,
                      resolution=(step if decimals else 1),
                      command=lambda v: (set_txt(float(v)),
                                         command(float(v))
                                         if command else None))
        sc.pack(side="left", padx=(2, 0))

        #  다른 곳에서 var 를 직접 바꿔도(기본값 복원, 색 찍기, 대상 전환)
        #  입력칸이 따라오게 한다.
        def sync(*_):
            try:
                set_txt(float(var.get()))
            except Exception:
                pass
        var.trace_add("write", sync)

        #  ★ 텍스트가 바뀌면(키보드 입력 포함) 잠시 뒤 자동 반영 ★
        #    키 이벤트 바인딩만으로는 포커스 상황에 따라 놓치는 경우가 있어,
        #    변수 자체를 감시한다.  우리가 바꾼 경우는 건너뛴다.
        def on_text(*_):
            if internal["on"]:
                return
            schedule()
        txt.trace_add("write", on_text)
        return row

    # ----------------------------------------------------------- 오른쪽
    def _panel(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=ACC, font=FONT_B,
                 anchor="w").pack(fill="x", pady=(6, 0))
        f = tk.Frame(parent, bg=PANEL, padx=8, pady=6)
        f.pack(fill="x")
        return f

    def _build_right(self, parent):
        col = tk.Frame(parent, bg=BG, width=RIGHT_COL_W)
        col.pack(side="left", fill="y", padx=(8, 0))
        col.pack_propagate(False)

        #  ★ 높이를 줄 수로 고정한다 ★
        #    검출 여부에 따라 줄 수가 달라지면 아래 패널들이 통째로 위아래로
        #    튀어서 눈으로 따라가기 어렵다.  가장 긴 상태에 맞춰 잡아 둔다.
        f = self._panel(col, "차례")
        self.canvas_turn = tk.Canvas(f, width=300, height=104, bg=PANEL,
                                     highlightthickness=0)
        self.canvas_turn.pack()
        self.lbl_turn = tk.Label(f, text="", bg=PANEL, fg=DIM, font=FONT_M,
                                 anchor="w", height=1)
        self.lbl_turn.pack(fill="x")
        self._turn_drawn = None

        f = self._panel(col, "측정값")
        self.lbl_pos = tk.Label(f, bg=PANEL, fg=OK_C, font=FONT_BIG,
                                justify="left", anchor="nw", height=4)
        self.lbl_pos.pack(fill="x")
        self.lbl_det = tk.Label(f, bg=PANEL, fg=FG, font=FONT_M,
                                justify="left", anchor="nw", height=4)
        self.lbl_det.pack(fill="x", pady=(4, 0))
        self.lbl_rate = tk.Label(f, bg=PANEL, fg=ACC, font=FONT_M,
                                 justify="left", anchor="nw", height=1)
        self.lbl_rate.pack(fill="x", pady=(2, 0))

        f = self._panel(col, "예측 — 라켓 평면 통과")
        self.lbl_shot = tk.Label(f, text="예측 없음", bg=PANEL, fg=DIM,
                                 font=FONT_M, justify="left", anchor="nw",
                                 height=2)
        self.lbl_shot.pack(fill="x")

        f = self._panel(col, "로봇")
        self.lbl_arm = tk.Label(f, text="안전 (무장 안 됨)", bg=PANEL,
                                fg=WARN_C, font=FONT_BIG, anchor="w")
        self.lbl_arm.pack(fill="x")
        self.lbl_motion = tk.Label(f, text="", bg=PANEL, fg=FG, font=FONT_M,
                                   justify="left", anchor="nw", height=3)
        self.lbl_motion.pack(fill="x", pady=(4, 0))
        self.lbl_stats = tk.Label(f, text="", bg=PANEL, fg=DIM, font=FONT_M,
                                  justify="left", anchor="nw", height=5)
        self.lbl_stats.pack(fill="x", pady=(4, 0))

        f = self._panel(col, "캘리브레이션")
        info = ("cam0  RMS %.3f px  (%+.2f, %+.2f, %+.2f)\n"
                "cam1  RMS %.3f px  (%+.2f, %+.2f, %+.2f)\n"
                "기선 %.3f m"
                % (self.calib[0].rms, *self.calib[0].C,
                   self.calib[1].rms, *self.calib[1].C,
                   float(np.linalg.norm(self.calib[0].C - self.calib[1].C))))
        tk.Label(f, text=info, bg=PANEL, fg=DIM, font=FONT_M,
                 justify="left", anchor="w").pack(fill="x")

    # ------------------------------------------------------------- 하단
    def _build_bottom(self, parent):
        nb = ttk.Notebook(parent, height=BOTTOM_BAR_H)
        nb.pack(fill="x", pady=(6, 0))
        self._tab_view(nb)
        self._tab_detect(nb)
        self._tab_predict(nb)
        self._tab_robot(nb)
        self._tab_tune(nb)

    def _tab_view(self, nb):
        f = tk.Frame(nb, bg=PANEL, padx=10, pady=8)
        nb.add(f, text="  보기  ")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        self.btn_pause = ttk.Button(row, text="일시정지 (Space)", width=17,
                                    command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=2)
        ttk.Button(row, text="궤적 지우기 (T)", width=15,
                   command=self.clear_trail).pack(side="left", padx=2)
        self.btn_mask = ttk.Button(row, text="마스크 보기 (M)", width=15,
                                   command=self.toggle_mask)
        self.btn_mask.pack(side="left", padx=2)
        self.btn_rec = ttk.Button(row, text="기록 시작 (R)", width=14,
                                  command=self.toggle_record)
        self.btn_rec.pack(side="left", padx=2)
        self.btn_cov = ttk.Button(row, text="시야 커버리지", width=14,
                                  command=self.check_coverage)
        self.btn_cov.pack(side="left", padx=2)
        ttk.Button(row, text="커버리지 끄기", width=14,
                   command=self.hide_coverage).pack(side="left", padx=2)
        ttk.Button(row, text="설정 저장", width=12,
                   command=self.save_profile).pack(side="left", padx=(16, 2))
        ttk.Button(row, text="설정 불러오기", width=14,
                   command=self.load_profile).pack(side="left", padx=2)
        self.lbl_profile = tk.Label(row, text="", bg=PANEL, fg=DIM,
                                    font=FONT_M)
        self.lbl_profile.pack(side="left", padx=6)

        row2 = tk.Frame(f, bg=PANEL)
        row2.pack(fill="x", pady=(8, 0))
        self.var_showroi = tk.BooleanVar(value=True)
        self.var_showcam = tk.BooleanVar(value=True)
        for txt, var in (("카메라 화면에 탐지 영역 표시", self.var_showroi),
                         ("3D 뷰에 카메라 위치 표시", self.var_showcam)):
            tk.Checkbutton(row2, text=txt, variable=var, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left", padx=8)

    def _tab_detect(self, nb):
        f = tk.Frame(nb, bg=PANEL, padx=10, pady=6)
        nb.add(f, text="  검출  ")

        left = tk.Frame(f, bg=PANEL)
        left.pack(side="left", fill="y")
        tr = tk.Frame(left, bg=PANEL)
        tr.pack(fill="x")
        tk.Label(tr, text="적용 대상", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left")
        self.var_target = tk.IntVar(value=2)
        for lab, val in (("양쪽", 2), ("cam0", 0), ("cam1", 1)):
            tk.Radiobutton(tr, text=lab, variable=self.var_target, value=val,
                           command=self.on_target, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left")
        mr = tk.Frame(left, bg=PANEL)
        mr.pack(fill="x")
        tk.Label(mr, text="검출 방식", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left")
        self.var_mode = tk.StringVar(
            value=str(self.params[0].get("mode", "both")))
        for lab, val in (("색상", "color"), ("밝기", "bright"), ("둘 다", "both")):
            tk.Radiobutton(mr, text=lab, variable=self.var_mode, value=val,
                           command=self.on_mode, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left")
        br = tk.Frame(left, bg=PANEL)
        br.pack(fill="x", pady=(4, 0))
        self.btn_pick = ttk.Button(br, text="공 색 찍기 (P)", width=15,
                                   command=self.toggle_pick)
        self.btn_pick.pack(side="left", padx=1)
        ttk.Button(br, text="주황 기본값", width=12,
                   command=self.reset_hsv).pack(side="left", padx=1)
        br2 = tk.Frame(left, bg=PANEL)
        br2.pack(fill="x", pady=(3, 0))
        ttk.Button(br2, text="검출 진단", width=12,
                   command=self.diagnose_detection).pack(side="left", padx=1)
        ttk.Button(br2, text="검출값 저장", width=12,
                   command=self.save_detector).pack(side="left", padx=1)
        ttk.Button(br2, text="기본값 복원", width=12,
                   command=self.reset_detector).pack(side="left", padx=1)

        self.sliders = {}
        specs = [("h_lo", "H 하한", 0, 179), ("h_hi", "H 상한", 0, 179),
                 ("s_lo", "S 하한", 0, 255), ("v_lo", "V 하한", 0, 255),
                 ("bright_v_lo", "밝기 V 하한", 100, 255),
                 ("bright_s_hi", "밝기 S 상한", 0, 200),
                 ("min_radius_px", "최소 반지름", 1, 30),
                 ("max_radius_px", "최대 반지름", 5, 80),
                 ("min_circularity", "최소 원형도", 0.0, 0.95),
                 ("border_band_px", "경계 밴드", 0, 80)]
        grid = tk.Frame(f, bg=PANEL)
        grid.pack(side="left", padx=(14, 0))
        for n, (key, label, lo, hi) in enumerate(specs):
            var = tk.DoubleVar(value=float(self.params[0][key]))
            is_float = isinstance(lo, float)
            row = self._value_row(
                grid, var, lo, hi,
                step=(0.05 if is_float else 1), decimals=(2 if is_float else 0),
                length=110, label=label,
                command=lambda v, k=key: self.on_slider(k, v))
            row.grid(row=n % 4, column=n // 4, sticky="w", padx=(0, 8))
            self.sliders[key] = var

        opt = tk.Frame(f, bg=PANEL)
        opt.pack(side="left", padx=(14, 0))
        #  ★ 시작 기본값을 설정으로 강제한다 (detector.json 보다 우선) ★
        if cfg.DEFAULT_USE_BGSUB is not None:
            for i in (0, 1):
                self.params[i]["use_bgsub"] = bool(cfg.DEFAULT_USE_BGSUB)
        if cfg.DEFAULT_COMPENSATE_ASYNC is not None:
            for i in (0, 1):
                self.params[i]["compensate_async"] = \
                    bool(cfg.DEFAULT_COMPENSATE_ASYNC)
        self.var_bg = tk.BooleanVar(value=bool(self.params[0]["use_bgsub"]))
        self.var_async = tk.BooleanVar(
            value=bool(self.params[0]["compensate_async"]))
        self.var_roimask = tk.BooleanVar(
            value=bool(self.det_params.get("use_roi_mask", True)))
        for txt, var in (("배경 차분", self.var_bg),
                         ("비동기 셔터 보정", self.var_async),
                         ("ROI 안으로 탐지 제한", self.var_roimask)):
            tk.Checkbutton(opt, text=txt, variable=var, command=self.on_opt,
                           bg=PANEL, fg=FG, selectcolor=BG, font=FONT,
                           activebackground=PANEL, activeforeground=FG,
                           anchor="w").pack(fill="x")
        self.var_roimargin = tk.DoubleVar(
            value=float(self.det_params.get("roi_mask_margin_px", 20)))
        self._value_row(opt, self.var_roimargin, 0, 80, length=100,
                        label="탐지 여유 px",
                        command=self.on_roi_margin).pack(fill="x")
        er = tk.Frame(opt, bg=PANEL)
        er.pack(fill="x")
        self.var_autoexp = tk.BooleanVar(value=True)
        tk.Checkbutton(er, text="자동 노출", variable=self.var_autoexp,
                       command=self.on_exposure, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG).pack(side="left")
        self.var_exp = tk.DoubleVar(value=-6)
        self._value_row(er, self.var_exp, -13, 0, length=90,
                        command=lambda v: self.on_exposure()).pack(side="left")

    def _tab_predict(self, nb):
        f = tk.Frame(nb, bg=PANEL, padx=10, pady=8)
        nb.add(f, text="  예측  ")
        p = self.fit.pred

        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="타겟 Y (m)", bg=PANEL, fg=DIM, font=FONT,
                 width=12, anchor="w").pack(side="left")
        self.var_target_y = tk.DoubleVar(value=p.target_y)
        tk.Scale(row, from_=0.0, to=float(self.geo["table_length"]),
                 resolution=0.05, orient="horizontal",
                 variable=self.var_target_y, bg=PANEL, fg=FG, font=FONT_M,
                 highlightthickness=0, troughcolor="#15151a", length=260,
                 command=lambda v: p.set_target_y(float(v))).pack(side="left")

        row2 = tk.Frame(f, bg=PANEL)
        row2.pack(fill="x", pady=(4, 0))
        tk.Label(row2, text="횡 기울기 배수", bg=PANEL, fg=DIM, font=FONT,
                 width=12, anchor="w").pack(side="left")
        ttk.Button(row2, text="-", width=3,
                   command=lambda: self._gain(-1)).pack(side="left")
        self.lbl_gain = tk.Label(row2, text="%.2f" % p.gain, bg=PANEL, fg=FG,
                                 font=FONT_M, width=6)
        self.lbl_gain.pack(side="left")
        ttk.Button(row2, text="+", width=3,
                   command=lambda: self._gain(+1)).pack(side="left")
        ttk.Button(row2, text="진행 방향 반전", width=14,
                   command=p.toggle_direction).pack(side="left", padx=(12, 2))
        ttk.Button(row2, text="예측 지우기", width=12,
                   command=p.clear).pack(side="left", padx=2)

        rowd = tk.Frame(f, bg=PANEL)
        rowd.pack(fill="x", pady=(4, 0))
        tk.Label(rowd, text="결정선 (cm)", bg=PANEL, fg=ACC, font=FONT_B,
                 width=12, anchor="w").pack(side="left")
        self.var_decide_cm = tk.DoubleVar(value=cfg.DECIDE_LINE_CM)
        self._value_row(rowd, self.var_decide_cm, 0, 137, length=200,
                        command=lambda v: self.apply_decide_line()
                        ).pack(side="left")
        self.var_decide_on = tk.BooleanVar(value=cfg.DECIDE_LINE_ENABLE)
        tk.Checkbutton(rowd, text="사용", variable=self.var_decide_on,
                       command=self.apply_decide_line, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG).pack(side="left", padx=6)
        self.lbl_decide = tk.Label(rowd, text="", bg=PANEL, fg=DIM,
                                   font=FONT_M)
        self.lbl_decide.pack(side="left", padx=4)

        rowr = tk.Frame(f, bg=PANEL)
        rowr.pack(fill="x", pady=(4, 0))
        tk.Label(rowr, text="역방향 문턱 cm", bg=PANEL, fg=DIM, font=FONT,
                 width=12, anchor="w").pack(side="left")
        self.var_dir_rev = tk.DoubleVar(value=p.cfg["dir_rev_m"] * 100.0)
        self._value_row(rowr, self.var_dir_rev, 0, 30, length=150,
                        command=lambda v: p.set_dir_rev(float(v) / 100.0)
                        ).pack(side="left")
        tk.Label(rowr, text="뒤로 이만큼 가면 새 트랙", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=6)

        row3 = tk.Frame(f, bg=PANEL)
        row3.pack(fill="x", pady=(4, 0))
        tk.Label(row3, text="바운스 반발계수", bg=PANEL, fg=DIM, font=FONT,
                 width=12, anchor="w").pack(side="left")
        self.var_restitution = tk.DoubleVar(value=p.cfg["restitution"])
        tk.Scale(row3, from_=0.5, to=1.2, resolution=0.01,
                 orient="horizontal", variable=self.var_restitution,
                 bg=PANEL, fg=FG, font=FONT_M, highlightthickness=0,
                 troughcolor="#15151a", length=200, sliderlength=16,
                 command=lambda v: p.set_restitution(float(v))
                 ).pack(side="left")
        tk.Label(row3, text="수평 감쇠", bg=PANEL, fg=DIM, font=FONT,
                 anchor="w").pack(side="left", padx=(12, 4))
        self.var_friction = tk.DoubleVar(value=p.cfg["friction"])
        tk.Scale(row3, from_=0.5, to=1.2, resolution=0.01,
                 orient="horizontal", variable=self.var_friction,
                 bg=PANEL, fg=FG, font=FONT_M, highlightthickness=0,
                 troughcolor="#15151a", length=160, sliderlength=16,
                 command=lambda v: p.set_friction(float(v))).pack(side="left")

        tk.Label(f, text="반발계수는 아직 관측되지 않은 바운스 뒤를 그릴 때만 "
                         "쓰입니다. 관측된 바운스는 실측으로 피팅합니다.",
                 bg=PANEL, fg=DIM, font=FONT, anchor="w",
                 wraplength=680, justify="left").pack(fill="x", pady=(4, 0))

    def _tab_robot(self, nb):
        f = tk.Frame(nb, bg=PANEL, padx=10, pady=8)
        nb.add(f, text="  로봇  ")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        self.btn_arm = ttk.Button(row, text="무장 (A)", width=16,
                                  command=self.toggle_arm)
        self.btn_arm.pack(side="left", padx=2)
        ttk.Button(row, text="비상정지 (E)", width=16,
                   command=self.emergency).pack(side="left", padx=2)
        ttk.Button(row, text="X=0 으로 정렬", width=16,
                   command=self.align_center).pack(side="left", padx=2)
        ttk.Button(row, text="준비운동", width=12,
                   command=self.warmup).pack(side="left", padx=2)
        ttk.Button(row, text="리니어 0 복귀", width=14,
                   command=self.park_linear_now).pack(side="left", padx=2)
        self.btn_practice = ttk.Button(row, text="연습 스윙", width=12,
                                       command=self.practice_swing)
        self.btn_practice.pack(side="left", padx=(12, 2))
        self.btn_swingmeas = ttk.Button(row, text="스윙 시간 측정", width=16,
                                        command=self.measure_swing)
        self.btn_swingmeas.pack(side="left", padx=2)
        self.btn_real = ttk.Button(row, text="실기 연결", width=14,
                                   command=self.toggle_real)
        self.btn_real.pack(side="left", padx=(12, 2))
        self.lbl_link = tk.Label(row, text="", bg=PANEL, fg=DIM, font=FONT_M)
        self.lbl_link.pack(side="left", padx=6)
        self.lbl_swmeas = tk.Label(row, text="", bg=PANEL, fg=OK_C,
                                   font=FONT_M)
        self.lbl_swmeas.pack(side="left", padx=6)
        axr = tk.Frame(f, bg=PANEL)
        axr.pack(fill="x", pady=(8, 0))
        tk.Label(axr, text="움직일 축", bg=PANEL, fg=ACC, font=FONT_B,
                 width=10, anchor="w").pack(side="left")
        be = self.worker.backend
        self.var_ax_linear = tk.BooleanVar(value=bool(be.enable_linear))
        self.var_ax_tilt = tk.BooleanVar(value=bool(be.enable_tilt))
        self.var_ax_swing = tk.BooleanVar(value=bool(be.enable_swing))
        for txt, var in (("리니어 (폭 이동)", self.var_ax_linear),
                         ("팔 (높이 · 라켓 면)", self.var_ax_tilt),
                         ("스윙 (바닥 원판)", self.var_ax_swing)):
            tk.Checkbutton(axr, text=txt, variable=var,
                           command=self.apply_axes, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left", padx=6)

        swr = tk.Frame(f, bg=PANEL)
        swr.pack(fill="x", pady=(6, 0))
        tk.Label(swr, text="스윙 폭", bg=PANEL, fg=ACC, font=FONT_B,
                 width=10, anchor="w").pack(side="left")
        tk.Label(swr, text="대기각(후방)", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left")
        self.var_sw_rest = tk.DoubleVar(value=cfg.SWING_REST_DEG)
        tk.Scale(swr, from_=-90, to=0, resolution=1, orient="horizontal",
                 variable=self.var_sw_rest, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0, width=9,
                 troughcolor="#15151a", length=150, sliderlength=14,
                 command=lambda v: self.apply_swing_span()).pack(side="left")
        tk.Label(swr, text="타격각(전방)", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=(10, 0))
        self.var_sw_front = tk.DoubleVar(value=cfg.SWING_FRONT_DEG)
        tk.Scale(swr, from_=0, to=90, resolution=1, orient="horizontal",
                 variable=self.var_sw_front, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0, width=9,
                 troughcolor="#15151a", length=150, sliderlength=14,
                 command=lambda v: self.apply_swing_span()).pack(side="left")
        self.lbl_swspan = tk.Label(swr, text="", bg=PANEL, fg=DIM,
                                   font=FONT_M)
        self.lbl_swspan.pack(side="left", padx=6)

        tk.Label(f, text=("무장 시 X=0 으로 정렬합니다. 꺼진 축은 지령도 "
                          "이동 시간 계산도 빠집니다."),
                 bg=PANEL, fg=DIM, font=FONT, anchor="w",
                 justify="left").pack(fill="x", pady=(4, 0))
        self.lbl_robot_info = tk.Label(f, text="", bg=PANEL, fg=DIM,
                                       font=FONT_M, anchor="w",
                                       justify="left")
        self.lbl_robot_info.pack(fill="x", pady=(2, 0))

    def _tab_tune(self, nb):
        f = tk.Frame(nb, bg=PANEL, padx=10, pady=6)
        nb.add(f, text="  보정  ")
        self.tune_vars = {}

        left = tk.Frame(f, bg=PANEL)
        left.pack(side="left", fill="y")
        r1 = tk.Frame(left, bg=PANEL)
        r1.pack(fill="x")
        tk.Label(r1, text="X 오프셋 mm", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        self.tune_vars["arm_x_offset_mm"] = tk.StringVar(
            value="%.1f" % self.tune["arm_x_offset_mm"])
        tk.Entry(r1, textvariable=self.tune_vars["arm_x_offset_mm"], width=10,
                 bg="#15151a", fg=FG, font=FONT_M,
                 insertbackground=FG).pack(side="left")
        tk.Label(r1, text="라켓 오프셋", bg=PANEL, fg=DIM, font=FONT,
                 width=10, anchor="w").pack(side="left", padx=(10, 0))
        self.tune_vars["paddle_offset_mm"] = tk.StringVar(
            value="%.1f" % self.tune.get("paddle_offset_mm",
                                         cfg.ARM.paddle_offset_mm))
        tk.Entry(r1, textvariable=self.tune_vars["paddle_offset_mm"], width=8,
                 bg="#15151a", fg=FG, font=FONT_M,
                 insertbackground=FG).pack(side="left")
        tk.Label(r1, text="손목 축 -> 라켓 중심 mm", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=4)

        r2 = tk.Frame(left, bg=PANEL)
        r2.pack(fill="x", pady=(2, 0))
        self.var_armsign = tk.BooleanVar(value=self.tune["arm_x_sign"] < 0)
        self.var_linsign = tk.BooleanVar(value=self.tune["linear_sign"] < 0)
        self.var_id5sign = tk.BooleanVar(
            value=self.tune.get("id5_sign", cfg.ID5_SIGN) < 0)
        for txt, var in (("팔 방향 반전", self.var_armsign),
                         ("리니어 방향 반전", self.var_linsign),
                         ("손목(ID5) 방향 반전", self.var_id5sign)):
            tk.Checkbutton(r2, text=txt, variable=var, bg=PANEL, fg=FG,
                           selectcolor=BG, font=FONT, activebackground=PANEL,
                           activeforeground=FG).pack(side="left", padx=4)

        r3 = tk.Frame(left, bg=PANEL)
        r3.pack(fill="x", pady=(2, 0))
        tk.Label(r3, text="타격 폭 X", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        #  None(자동)이면 판정기가 계산한 실제 도달 범위를 슬라이더에 넣는다
        auto_lo, auto_hi = self.planner.x_min, self.planner.x_max
        self.var_hit_lo = tk.DoubleVar(
            value=(auto_lo if self.tune["hit_x_min_mm"] is None
                   else self.tune["hit_x_min_mm"]))
        self.var_hit_hi = tk.DoubleVar(
            value=(auto_hi if self.tune["hit_x_max_mm"] is None
                   else self.tune["hit_x_max_mm"]))
        #  자동 유지 여부.  체크하면 기구학이 계산한 범위를 그대로 쓴다.
        self.var_hit_auto = tk.BooleanVar(
            value=self.tune["hit_x_min_mm"] is None)
        tk.Checkbutton(r3, text="자동", variable=self.var_hit_auto,
                       bg=PANEL, fg=FG, selectcolor=BG, font=FONT,
                       activebackground=PANEL,
                       activeforeground=FG).pack(side="left")
        for var in (self.var_hit_lo, self.var_hit_hi):
            tk.Scale(r3, from_=-760, to=760, resolution=10,
                     orient="horizontal", variable=var, bg=PANEL, fg=FG,
                     font=("Consolas", 7), highlightthickness=0,
                     troughcolor="#15151a", length=150, sliderlength=14,
                     width=9).pack(side="left", padx=2)

        right = tk.Frame(f, bg=PANEL)
        right.pack(side="left", padx=(20, 0))
        r4 = tk.Frame(right, bg=PANEL)
        r4.pack(fill="x")
        tk.Label(r4, text="스윙 소요 ms", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        self.var_swing_ms = tk.DoubleVar(
            value=self.tune["swing_duration_sec"] * 1000)
        tk.Scale(r4, from_=40, to=400, resolution=1, orient="horizontal",
                 variable=self.var_swing_ms, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0, width=9,
                 troughcolor="#15151a", length=200,
                 sliderlength=14).pack(side="left")

        r5 = tk.Frame(right, bg=PANEL)
        r5.pack(fill="x", pady=(2, 0))
        tk.Label(r5, text="지령 지연 ms", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        self.var_lat_ms = tk.DoubleVar(
            value=self.tune["swing_latency_sec"] * 1000)
        tk.Scale(r5, from_=0, to=80, resolution=1, orient="horizontal",
                 variable=self.var_lat_ms, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0, width=9,
                 troughcolor="#15151a", length=200,
                 sliderlength=14).pack(side="left")

        r6 = tk.Frame(right, bg=PANEL)
        r6.pack(fill="x", pady=(2, 0))
        tk.Label(r6, text="대기 theta °", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        self.var_ready_th = tk.DoubleVar(value=self.tune["ready_theta_deg"])
        tk.Scale(r6, from_=cfg.ARM.theta_min_deg, to=-10, resolution=1,
                 orient="horizontal", variable=self.var_ready_th, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0, width=9,
                 troughcolor="#15151a", length=200,
                 sliderlength=14).pack(side="left")

        r64 = tk.Frame(right, bg=PANEL)
        r64.pack(fill="x", pady=(2, 0))
        tk.Label(r64, text="조준 Z 보정", bg=PANEL, fg=ACC, font=FONT_B,
                 width=13, anchor="w").pack(side="left")
        self.var_aim_z = tk.DoubleVar(
            value=self.tune.get("aim_z_offset_mm", cfg.AIM_Z_OFFSET_MM))
        self._value_row(r64, self.var_aim_z, -150, 150, length=150,
                        command=lambda v: self.apply_aim()).pack(side="left")
        tk.Label(r64, text="mm  (라켓이 낮으면 +)", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=4)

        r65 = tk.Frame(right, bg=PANEL)
        r65.pack(fill="x", pady=(2, 0))
        tk.Label(r65, text="스윙 속도/가속", bg=PANEL, fg=DIM, font=FONT,
                 width=13, anchor="w").pack(side="left")
        self.var_sw_vel = tk.IntVar(value=cfg.SWING_VELOCITY)
        self.var_sw_acc = tk.IntVar(value=cfg.SWING_ACCELERATION)
        for var, hi in ((self.var_sw_vel, 500), (self.var_sw_acc, 300)):
            tk.Scale(r65, from_=0, to=hi, resolution=5, orient="horizontal",
                     variable=var, bg=PANEL, fg=FG, font=("Consolas", 7),
                     highlightthickness=0, width=9, troughcolor="#15151a",
                     length=95, sliderlength=14,
                     command=lambda v: self.apply_swing_profile()
                     ).pack(side="left", padx=1)
        tk.Label(r65, text="0=최대", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=2)

        r7 = tk.Frame(right, bg=PANEL)
        r7.pack(fill="x", pady=(4, 0))
        ttk.Button(r7, text="적용", width=10,
                   command=self.apply_tuning).pack(side="left", padx=2)
        ttk.Button(r7, text="적용 + 저장", width=12,
                   command=lambda: self.apply_tuning(save=True)
                   ).pack(side="left", padx=2)
        ttk.Button(r7, text="되돌리기", width=10,
                   command=self.reload_tuning).pack(side="left", padx=2)
        self.lbl_tune = tk.Label(right, text="", bg=PANEL, fg=DIM,
                                 font=FONT_M, anchor="w")
        self.lbl_tune.pack(fill="x", pady=(2, 0))

    # ------------------------------------------------------------------
    #  조작
    # ------------------------------------------------------------------
    def _bind_events(self):
        super()._bind_events()
        for k in "aA":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.toggle_arm())
        for k in "eE":
            self.bind_all("<KeyPress-%s>" % k, lambda e: self.emergency())

    def _m_pan(self, ev):
        """3D 뷰 우클릭 드래그 — 카메라 이동.

        원본은 드래그한 방향으로 카메라가 움직여서, 화면 속 장면은 반대로
        흘러간다.  PAN_INVERT 면 부호를 뒤집어 '화면을 손으로 끄는' 느낌으로
        바꾼다.  track_ball.py 는 건드리지 않고 이 메서드만 덮어쓴다.
        """
        dx = ev.x - self._mx
        dy = ev.y - self._my
        if cfg.PAN_INVERT:
            dx, dy = -dx, -dy
        self.viewer.pan(dx, dy)
        self._mx, self._my = ev.x, ev.y

    def _gain(self, sign):
        p = self.fit.pred
        p.adjust_gain(sign * p.cfg["lat_gain_step"])
        self.lbl_gain.config(text="%.2f" % p.gain)

    def toggle_arm(self):
        if self.worker is None:
            return
        self.worker.set_armed(not self.worker.armed)
        armed = self.worker.armed
        self.lbl_arm.config(text="무장 — 축이 움직입니다" if armed
                            else "안전 (무장 안 됨)",
                            fg=BAD_C if armed else WARN_C)
        self.btn_arm.config(text="무장 해제 (A)" if armed else "무장 (A)")

    def emergency(self):
        if self.worker is None:
            return
        self.worker.emergency_stop()
        self.lbl_arm.config(text="비상정지 — 무장 해제됨", fg=BAD_C)
        self.btn_arm.config(text="무장 (A)")

    def _is_real(self) -> bool:
        return type(self.worker.backend).__name__ == "RealBackend"

    def toggle_real(self):
        """실행 중에 실기 <-> 시뮬레이터 를 갈아끼운다."""
        if self.worker is None:
            return
        going_real = not self._is_real()
        if going_real and not messagebox.askyesno(
                "실기 연결",
                "다이나믹셀과 리니어축에 연결합니다.\n\n"
                "· 캐리지가 좌측 끝(지령 0)에 있습니까?\n"
                "· 팔 주변에 사람이나 물건이 없습니까?\n\n"
                "연결 후에도 [무장] 전에는 축이 움직이지 않습니다."):
            return
        try:
            be = self._make_backend(real=going_real)
        except Exception as e:
            messagebox.showerror("연결", str(e))
            return
        self.worker.set_backend(be)
        self._sync_link_label()
        self.lbl_arm.config(text="안전 (무장 안 됨)", fg=WARN_C)
        self.btn_arm.config(text="무장 (A)")

    def _sync_link_label(self):
        real = self._is_real()
        self.btn_real.config(text="시뮬레이터로" if real else "실기 연결")
        be = self.worker.backend
        extra = ""
        if real and getattr(be, "linear", None) is not None:
            extra = ("  리니어 %s"
                     % ("가상" if getattr(be.linear, "simulated", False)
                        else "실기"))
        self.lbl_link.config(text=("실기 연결됨" if real else "시뮬레이터") + extra,
                             fg=OK_C if real else DIM)

    def apply_axes(self):
        """축별 on/off 를 백엔드에 반영한다."""
        be = self.worker.backend
        be.enable_linear = bool(self.var_ax_linear.get())
        be.enable_tilt = bool(self.var_ax_tilt.get())
        be.enable_swing = bool(self.var_ax_swing.get())
        print("[로봇] 움직일 축: %s" % be.axes_text())
        self._status("움직일 축: %s" % be.axes_text())

    def apply_swing_span(self):
        """스윙 폭(대기각 ~ 타격각)을 즉시 반영한다.

        접촉은 호의 중간(각도 0)에서 일어난다고 보고 타이밍을 잡으므로,
        대기각과 타격각의 크기를 비슷하게 두는 편이 예측과 맞는다.
        폭을 바꾸면 스윙 시간도 달라지니 [스윙 시간 측정] 을 다시 하세요.
        """
        rest = float(self.var_sw_rest.get())
        front = float(self.var_sw_front.get())
        cfg.SWING_REST_DEG = rest
        cfg.SWING_FRONT_DEG = front
        #  보정값에도 반영해 두어야 설정 저장에 함께 담긴다
        self.tune["swing_rest_deg"] = rest
        self.tune["swing_front_deg"] = front
        span = front - rest
        warn = ""
        if abs(abs(rest) - abs(front)) > 15:
            warn = "  (좌우 비대칭 — 접촉 시점이 중간에서 벗어납니다)"
        self.lbl_swspan.config(text="폭 %.0f°%s" % (span, warn))
        self._status("스윙 폭 %.0f° (대기 %+.0f° -> 타격 %+.0f°)%s"
                     % (span, rest, front, warn))

    def apply_decide_line(self):
        """결정선을 즉시 반영한다.

        공이 이 선을 넘는 순간 목적지를 한 번 확정하고 그 자리로 간다.
        선이 멀수록(네트에 가까울수록) 일찍 확정해 여유 있게 움직이지만
        예측이 덜 여물고, 가까울수록 정확하지만 시간이 모자라기 쉽다.
        """
        cfg.DECIDE_LINE_ENABLE = bool(self.var_decide_on.get())
        cfg.DECIDE_LINE_CM = float(self.var_decide_cm.get())
        self.tune["decide_line_cm"] = cfg.DECIDE_LINE_CM
        if cfg.DECIDE_LINE_ENABLE:
            txt = "끝선에서 %.0f cm" % cfg.DECIDE_LINE_CM
        else:
            txt = "끄면 계속 따라감"
        self.lbl_decide.config(text=txt)
        self._status("결정선: %s" % txt)

    def apply_aim(self):
        """조준 보정을 즉시 반영한다 (예측 타점에 그대로 더한다)."""
        cfg.AIM_Z_OFFSET_MM = float(self.var_aim_z.get())
        self.tune["aim_z_offset_mm"] = cfg.AIM_Z_OFFSET_MM
        self._status("조준 Z 보정 %+.0f mm" % cfg.AIM_Z_OFFSET_MM)

    def apply_swing_profile(self):
        """스윙 속도/가속을 즉시 반영한다 (0 = 제한 없음 = 가장 빠름)."""
        cfg.SWING_VELOCITY = int(self.var_sw_vel.get())
        cfg.SWING_ACCELERATION = int(self.var_sw_acc.get())
        self._status("스윙 프로파일  속도 %d  가속 %d  (0 = 최대)"
                     % (cfg.SWING_VELOCITY, cfg.SWING_ACCELERATION))

    def park_linear_now(self):
        """지금 바로 리니어를 지령 0 으로 되돌린다."""
        if self.worker is None:
            return
        be = self.worker.backend
        self._status("리니어 원위치 복귀 중...")

        def run():
            try:
                be.park_linear(float(cfg.PARK_LINEAR_CMD_ON_EXIT or 0.0),
                               timeout=cfg.PARK_TIMEOUT_SEC)
                msg = "리니어 원위치 복귀 완료 (%.2f)" % be.current_pose()[1]
            except Exception as e:
                msg = "리니어 복귀 실패: %s" % e
            self.after(0, lambda: (self._status(msg), print("[로봇] " + msg)))

        threading.Thread(target=run, daemon=True, name="park").start()

    def practice_swing(self):
        """공과 무관하게 스윙을 한 번 휘둘러 본다."""
        if self.worker is None:
            return
        be = self.worker.backend
        if not self.worker.armed:
            self._status("무장한 뒤에 스윙할 수 있습니다 (A).")
            return
        if not be.active_axes()["swing"]:
            self._status("스윙 축이 꺼져 있습니다.")
            return
        self.btn_practice.config(state="disabled")
        self._status("연습 스윙...")

        def run():
            try:
                sec = be.swing()
                err = None
            except Exception as e:
                sec, err = 0.0, e
            self.after(0, lambda: self._practice_done(sec, err))

        threading.Thread(target=run, daemon=True, name="practice").start()

    def _practice_done(self, sec, err):
        self.btn_practice.config(state="normal")
        if err is not None:
            self._status("연습 스윙 실패: %s" % err)
            print("[로봇] 연습 스윙 실패: %s" % err)
            return
        self._status("연습 스윙 완료 — 현재 설정 스윙 %.0f ms, "
                     "리드타임 %.0f ms"
                     % (cfg.SWING_DURATION_SEC * 1000, swing_lead_sec() * 1000))
        print("[로봇] 연습 스윙 — 속도 %d 가속 %d (0=최대)"
              % (cfg.SWING_VELOCITY, cfg.SWING_ACCELERATION))

    def measure_swing(self):
        """스윙 한 번을 실제로 휘둘러 소요 시간을 잰다."""
        if self.worker is None:
            return
        be = self.worker.backend
        if not self.worker.armed:
            self._status("무장한 뒤에 측정할 수 있습니다 (A).")
            return
        if not be.active_axes()["swing"]:
            self._status("스윙 축이 꺼져 있습니다.")
            return
        if not messagebox.askyesno(
                "스윙 시간 측정",
                "라켓을 한 번 휘두릅니다.\n주변에 사람이나 물건이 없습니까?"):
            return
        self.btn_swingmeas.config(state="disabled")
        self._status("스윙 측정 중...")

        def run():
            try:
                sec = be.measure_swing()
                err = None
            except Exception as e:
                sec, err = 0.0, e
            self.after(0, lambda: self._swing_measured(sec, err))

        threading.Thread(target=run, daemon=True, name="swing-measure").start()

    def _swing_measured(self, sec, err):
        self.btn_swingmeas.config(state="normal")
        if err is not None or sec <= 0:
            messagebox.showerror("스윙 시간 측정",
                                 "측정에 실패했습니다.\n%s" % (err or ""))
            return
        ms = round(sec * 1000)
        self.var_swing_ms.set(ms)
        self.apply_tuning()                 # cfg 에 즉시 반영
        lead = swing_lead_sec() * 1000
        self.lbl_swmeas.config(text="측정 스윙 %d ms -> 리드타임 %.0f ms"
                                    % (ms, lead))
        messagebox.showinfo(
            "스윙 시간 측정",
            "대기각 -> 타격각  %d ms\n"
            "발사 리드타임  %.0f ms  (접촉 지점 %.0f%% + 지연 %.0f ms)\n\n"
            "보정값에 바로 적용했습니다.\n"
            "다음 실행에도 쓰려면 [보정] 탭에서 [적용 + 저장] 을 누르세요."
            % (ms, lead, cfg.SWING_CONTACT_FRACTION * 100,
               cfg.SWING_COMMAND_LATENCY_SEC * 1000))

    def warmup(self):
        """타격 가능 영역의 윤곽을 한 바퀴 그린다 (육안 확인용)."""
        if self.worker is None:
            return
        if not self.worker.armed:
            self._status("무장한 뒤에 준비운동을 할 수 있습니다 (A).")
            return
        self.worker.request_warmup()
        self._status("준비운동 — 타격 가능 영역의 윤곽을 그립니다.")

    def align_center(self):
        if self.worker is None:
            return
        if not self.worker.armed:
            self._status("무장한 뒤에 정렬할 수 있습니다 (A).")
            return
        self.worker.request_ready(slow=True)

    # -------------------------------------------------------- 설정 저장
    def save_profile(self):
        """지금 설정을 파일 하나로 저장한다."""
        import os
        os.makedirs(app_profile.PROFILE_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="설정 저장", defaultextension=".json",
            initialdir=app_profile.PROFILE_DIR,
            initialfile=app_profile.default_name(),
            filetypes=[("설정 파일", "*.json")])
        if not path:
            return
        try:
            data = app_profile.collect(self)
            app_profile.save(data, path)
        except Exception as e:
            messagebox.showerror("설정 저장", str(e))
            return
        name = os.path.basename(path)
        self.lbl_profile.config(text=name)
        self._status("설정을 저장했습니다: %s" % path)
        print("[설정] 저장: %s" % path)

    def load_profile(self, path=None):
        """저장해 둔 설정을 불러와 지금 화면에 적용한다."""
        import os
        if path is None:
            path = filedialog.askopenfilename(
                title="설정 불러오기",
                initialdir=(app_profile.PROFILE_DIR
                            if os.path.isdir(app_profile.PROFILE_DIR)
                            else app_profile.SETTINGS_DIR),
                filetypes=[("설정 파일", "*.json")])
        if not path:
            return
        try:
            data = app_profile.load(path)
            skipped = app_profile.apply(self, data)
        except Exception as e:
            messagebox.showerror("설정 불러오기", str(e))
            return
        self.lbl_profile.config(text=os.path.basename(path))
        msg = app_profile.summary(data)
        if skipped:
            msg += "\n\n일부 항목을 적용하지 못했습니다:\n" + "\n".join(skipped)
        messagebox.showinfo("설정 불러오기", msg)
        print("[설정] 불러옴: %s" % path)
        self._status("설정을 불러왔습니다: %s" % os.path.basename(path))

    def clear_console(self):
        self.txt_console.configure(state="normal")
        self.txt_console.delete("1.0", "end")
        self.txt_console.configure(state="disabled")

    def apply_tuning(self, save: bool = False):
        try:
            v = dict(self.tune)
            v["arm_x_offset_mm"] = float(
                self.tune_vars["arm_x_offset_mm"].get())
            v["paddle_offset_mm"] = float(
                self.tune_vars["paddle_offset_mm"].get())
            v["arm_x_sign"] = -1.0 if self.var_armsign.get() else 1.0
            v["linear_sign"] = -1.0 if self.var_linsign.get() else 1.0
            v["id5_sign"] = -1.0 if self.var_id5sign.get() else 1.0
            if self.var_hit_auto.get():
                v["hit_x_min_mm"] = v["hit_x_max_mm"] = None
            else:
                lo = float(self.var_hit_lo.get())
                hi = float(self.var_hit_hi.get())
                v["hit_x_min_mm"], v["hit_x_max_mm"] = min(lo, hi), max(lo, hi)
            v["swing_duration_sec"] = float(self.var_swing_ms.get()) / 1000.0
            v["swing_latency_sec"] = float(self.var_lat_ms.get()) / 1000.0
            v["ready_theta_deg"] = float(self.var_ready_th.get())
            v["aim_z_offset_mm"] = float(self.var_aim_z.get())
            v["decide_line_cm"] = float(self.var_decide_cm.get())
            v["swing_rest_deg"] = float(self.var_sw_rest.get())
            v["swing_front_deg"] = float(self.var_sw_front.get())
        except ValueError:
            messagebox.showerror("보정", "숫자 형식을 확인하세요.")
            return

        self.tune = v
        tuning.apply_to_config(v)
        self.kin = ArmKinematics(tuning.geometry_from(v))
        self.planner = ShotPlanner(kinematics=self.kin, verbose=True)
        self.worker.set_kinematics(self.kin)
        if v["hit_x_min_mm"] is None:
            self.var_hit_lo.set(round(self.planner.x_min))
            self.var_hit_hi.set(round(self.planner.x_max))
        msg = tuning.describe(v)
        if save:
            path = tuning.save(v)
            msg += "   -> 저장됨"
            print("[보정] 저장: %s" % path)
        self.lbl_tune.config(text=msg)
        print("[보정] 적용: %s" % msg)
        print("[보정] %s   스윙 리드타임 %.0f ms"
              % (self.planner.describe_window(), swing_lead_sec() * 1000))
        self._status("보정값 적용됨. " + msg)

    def reload_tuning(self):
        self.tune = tuning.load()
        self.tune_vars["arm_x_offset_mm"].set(
            "%.1f" % self.tune["arm_x_offset_mm"])
        self.tune_vars["paddle_offset_mm"].set(
            "%.1f" % self.tune.get("paddle_offset_mm",
                                   cfg.ARM.paddle_offset_mm))
        self.var_armsign.set(self.tune["arm_x_sign"] < 0)
        self.var_linsign.set(self.tune["linear_sign"] < 0)
        self.var_id5sign.set(self.tune.get("id5_sign", cfg.ID5_SIGN) < 0)
        self.var_hit_auto.set(self.tune["hit_x_min_mm"] is None)
        if self.tune["hit_x_min_mm"] is not None:
            self.var_hit_lo.set(self.tune["hit_x_min_mm"])
        if self.tune["hit_x_max_mm"] is not None:
            self.var_hit_hi.set(self.tune["hit_x_max_mm"])
        self.var_swing_ms.set(self.tune["swing_duration_sec"] * 1000)
        self.var_lat_ms.set(self.tune["swing_latency_sec"] * 1000)
        self.var_ready_th.set(self.tune["ready_theta_deg"])
        self.var_aim_z.set(self.tune.get("aim_z_offset_mm",
                                         cfg.AIM_Z_OFFSET_MM))
        self.var_decide_cm.set(self.tune.get("decide_line_cm",
                                             cfg.DECIDE_LINE_CM))
        self.var_sw_rest.set(self.tune.get("swing_rest_deg",
                                           cfg.SWING_REST_DEG))
        self.var_sw_front.set(self.tune.get("swing_front_deg",
                                            cfg.SWING_FRONT_DEG))
        self.apply_tuning()

    # ------------------------------------------------------------------
    #  프레임 루프
    # ------------------------------------------------------------------
    def _tick(self):
        """원본보다 빠르게 돈다.

        원본은 20ms 주기(50Hz)로 잡혀 있어 120fps 카메라의 프레임을
        절반 넘게 버렸다.  여기서는 목표 주기를 카메라보다 빠르게 두고,
        무거운 그리기는 _draw 쪽에서 따로 솎아낸다.
        """
        if self._closing:
            return
        t0 = time.perf_counter()
        try:
            self._process()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status("오류: %s" % e)
        dt = time.perf_counter() - t0

        #  루프 속도 측정 (진단용)
        self._loop_n += 1
        if t0 - self._loop_t0 >= 1.0:
            self.loop_hz = self._loop_n / (t0 - self._loop_t0)
            self._loop_n, self._loop_t0 = 0, t0

        period_ms = 1000.0 / max(1.0, cfg.LOOP_TARGET_HZ)
        delay = max(1, int(period_ms - dt * 1000))
        self._after_id = self.after(delay, self._tick)

    def _update_crop_boxes(self):
        """ROI 마스크를 감싸는 사각형을 구해 검출기에 알려 준다."""
        for i in (0, 1):
            box = None
            m = self.roi_mask[i]
            if cfg.ROI_CROP and m is not None:
                nz = cv2.findNonZero(m)
                if nz is not None:
                    x, y, w, h = cv2.boundingRect(nz)
                    if w > 8 and h > 8:
                        box = (x, y, w, h)
            if isinstance(self.detector[i], CropDetector):
                self.detector[i].set_box(box)
            if box is not None and m is not None:
                ratio = 100.0 * (box[2] * box[3]) / float(m.shape[0]
                                                          * m.shape[1])
                print("[검출] cam%d 처리 영역 %dx%d @(%d,%d) — 화면의 %.0f%%"
                      % (i, box[2], box[3], box[0], box[1], ratio))
            elif cfg.ROI_CROP:
                print("[검출] cam%d 자를 영역이 없어 전체 화면을 처리합니다"
                      % i)

    def _rebuild_roi_masks(self):
        super()._rebuild_roi_masks()
        if getattr(self, "detector", None):
            self._update_crop_boxes()

    def _pair_frames(self):
        """같은 프레임 쌍을 두 번 처리하지 않는다.

        루프가 카메라보다 빨라졌으므로, 새 프레임이 아직 안 왔는데 루프가
        도는 경우가 생긴다.  그대로 두면 같은 관측이 칼만에 두 번 들어가고
        검출도 헛돈다.  타임스탬프가 그대로면 건너뛴다.
        """
        best = super()._pair_frames()
        if best is None:
            return None
        key = (best[1], best[3])          # (t0, t1)
        if key == getattr(self, "_last_pair_key", None):
            return None
        self._last_pair_key = key
        return best

    def _draw(self, f0, f1, masks, result, pos, pdt):
        """그리기는 솎아낸다.

        검출·삼각측량은 매 루프 돌지만, 카메라 미리보기와 3D 뷰는 사람이
        보는 속도(RENDER_HZ)로만 갱신한다.  이 하나로 루프가 두 배 넘게
        빨라지고, 그만큼 예측에 쓰는 샘플이 늘어난다.
        """
        self._last_frames = [f0, f1]        # 색 찍기용은 항상 최신으로
        now = time.perf_counter()
        if now - self._last_render < 1.0 / max(1.0, cfg.RENDER_HZ):
            return
        self._last_render = now
        super()._draw(f0, f1, masks, result, pos, pdt)
        #  ★ 배경을 다시 그린 직후에 이어서 라켓을 얹는다 ★
        #    super()._draw 가 canvas3d 를 통째로 지우므로, 라켓을 다른 주기로
        #    그리면 지워진 채로 남는 프레임이 생겨 깜빡인다.
        self._draw_robot_overlay()

    def _process(self):
        super()._process()
        if self.worker is None or self.planner is None:
            return

        #  새로 확정된 궤적 샘플만 예측기로 넘긴다
        #  로봇이 친 뒤 나가는 공은 차단해서 트랙을 만들지 않는다
        if self.trail:
            t, p = self.trail[-1]
            if self._fed_t is None or t > self._fed_t + 1e-9:
                self._fed_t = t
                self.turn.update(float(p[1]))
                if self.turn.should_feed():
                    self.fit.push(t, p)
                    self._sample_n += 1
        self.turn.tick()
        now = time.perf_counter()
        if now - self._sample_t0 >= 1.0:
            self.sample_hz = self._sample_n / (now - self._sample_t0)
            self._sample_n, self._sample_t0 = 0, now

        plan = self.planner.plan(self.fit.pred)
        self._last_plan = plan
        if plan.accepted and plan.target is not None:
            self.worker.submit(plan.target)

        #  로봇이 스윙했으면 다음 공이 넘어올 때까지 입력을 막는다
        sw = self.worker.stats.swung
        if sw != getattr(self, "_last_swung", 0):
            self._last_swung = sw
            self.turn.note_swing()

        #  그리기와 패널은 사람이 보는 속도로만
        if now - self._last_panel >= 1.0 / max(1.0, cfg.PANEL_HZ):
            self._last_panel = now
            self._draw_turn()
            self._draw_fit()
            self._update_panels(plan)
            self._drain_console()

    # ------------------------------------------------------ 3D 로봇 표시
    def _draw_turn(self):
        """지금 칠 차례를 사람/로봇 실루엣으로 보여 준다.

        사진 대신 코드로 그린 아이콘을 쓴다.  차례인 쪽은 밝게, 아닌 쪽은
        어둡게 그려서 멀리서도 한눈에 보이게 한다.
        """
        who = self.turn.who
        blocked = self.turn.blocked
        key = (who, blocked)
        if key == self._turn_drawn:
            return
        self._turn_drawn = key
        c = self.canvas_turn
        c.delete("all")

        on, off = "#e8e8ec", "#4a4a52"
        hi = OK_C if who == HUMAN else BAD_C
        hcol = on if who == HUMAN else off
        rcol = on if who == ROBOT else off

        #  ---- 사람 (왼쪽) ----
        c.create_oval(48, 14, 74, 40, outline=hcol, width=2)          # 머리
        c.create_line(61, 40, 61, 66, fill=hcol, width=3)             # 몸통
        c.create_line(61, 48, 40, 60, fill=hcol, width=2)             # 왼팔
        c.create_line(61, 48, 86, 38, fill=hcol, width=2)             # 라켓 팔
        c.create_oval(84, 26, 100, 44, outline=hcol, width=2)         # 라켓
        c.create_line(61, 66, 48, 86, fill=hcol, width=2)             # 다리
        c.create_line(61, 66, 76, 86, fill=hcol, width=2)
        c.create_text(61, 98, text="사람", fill=hcol, font=FONT)

        #  ---- 로봇 (오른쪽) ----
        c.create_rectangle(224, 16, 252, 40, outline=rcol, width=2)   # 머리
        c.create_oval(231, 24, 237, 30, outline=rcol, width=1)        # 눈
        c.create_oval(240, 24, 246, 30, outline=rcol, width=1)
        c.create_line(238, 8, 238, 16, fill=rcol, width=2)            # 안테나
        c.create_rectangle(228, 44, 248, 68, outline=rcol, width=2)   # 몸통
        c.create_line(228, 50, 210, 40, fill=rcol, width=2)           # 라켓 팔
        c.create_oval(194, 28, 210, 46, outline=rcol, width=2)        # 라켓
        c.create_line(248, 50, 262, 62, fill=rcol, width=2)
        c.create_line(233, 68, 233, 86, fill=rcol, width=2)           # 다리
        c.create_line(243, 68, 243, 86, fill=rcol, width=2)
        c.create_text(238, 98, text="로봇", fill=rcol, font=FONT)

        #  ---- 가운데 네트와 화살표 ----
        c.create_line(150, 18, 150, 78, fill="#6b6b75", width=2,
                      dash=(4, 3))
        #  화살표는 '공이 앞으로 갈 방향'을 가리킨다.
        #  사람 차례면 사람이 칠 참이므로 공은 로봇 쪽(오른쪽)으로 간다.
        if who == HUMAN:
            c.create_line(122, 50, 180, 50, fill=hi, width=3,
                          arrow="last", arrowshape=(12, 14, 5))
        else:
            c.create_line(178, 50, 120, 50, fill=hi, width=3,
                          arrow="last", arrowshape=(12, 14, 5))
        c.create_text(150, 96, text=("차단 중" if blocked else ""),
                      fill=WARN_C, font=FONT)

    def _decided_point(self):
        """결정선을 넘어 확정된 목적지 (m). 아직이면 None."""
        fr = getattr(self.planner, "_frozen", None) if self.planner else None
        if fr is None:
            return None
        return (fr[0] / 1000.0, cfg.PADDLE_PLANE_Y_MM / 1000.0,
                fr[1] / 1000.0)

    def _overlay_note(self, msg, *, quiet: bool = False):
        """3D 라켓 표시 상태. 문제가 있을 때만 콘솔에 남긴다."""
        if getattr(self, "_ov_last", None) == msg:
            return
        self._ov_last = msg
        if not quiet:
            print("[3D] 라켓 표시: %s" % msg)
        try:
            self.lbl_motion.config(text="%s\n3D 라켓 표시: %s"
                                        % (self.worker.state, msg))
        except Exception:
            pass

    def _draw_robot_overlay(self):
        try:
            self._draw_robot_overlay_inner()
        except Exception as e:
            self._overlay_note("오류 %s: %s" % (type(e).__name__, e))

    def _draw_robot_overlay_inner(self):
        """3D 뷰 위에 로봇이 '지금 자기가 있다고 믿는' 라켓 위치를 그린다.

        비전이 준 공의 위치와 이 라켓이 눈으로 맞아떨어지는지 보면,
        기구 상수(x 오프셋 · 힌지 높이 · 부호)가 맞는지 바로 알 수 있다.
        타격 창도 라켓 평면 위에 사각형으로 함께 그린다.
        """
        if not cfg.SHOW_PADDLE_IN_3D or self.worker is None:
            return
        if not hasattr(self.viewer, "_pose"):     # 뷰어가 투영을 지원해야 한다
            self._overlay_note("뷰어가 투영을 지원하지 않음")
            return
        be = self.worker.backend
        th, cmd = be.current_pose()          # ★ 지령이 아니라 실제 위치 ★
        th, cmd = float(th), float(cmd)
        g = self.kin.g
        x_mm, z_mm = self.kin.forward(th, cmd)
        y = cfg.PADDLE_PLANE_Y_MM / 1000.0
        hinge = ((cmd * g.linear_mm_per_unit + g.arm_x_offset_mm) / 1000.0,
                 y, g.hinge_z_mm / 1000.0)
        wx, wz = self.kin.wrist_xz(th, cmd)      # 막대 끝(손목 축)
        wrist = (wx / 1000.0, y, wz / 1000.0)
        paddle = (x_mm / 1000.0, y, z_mm / 1000.0)

        R, t, _ = self.viewer._pose()

        def proj(pts):
            uv, zz = self.viewer.project(R, t, np.array(pts, dtype=float))
            if not np.all(np.isfinite(uv)) or np.max(np.abs(uv)) > 1e4:
                return None
            return uv



        c = self.canvas3d
        c.delete("robot")          # 이전 프레임의 표시를 지운다

        #  ---- 결정선 ----
        #  공이 이 면을 넘는 순간 목적지가 확정된다.
        if cfg.DECIDE_LINE_ENABLE:
            ly = float(cfg.DECIDE_LINE_CM) / 100.0
            hw = float(self.geo["table_width"]) / 2.0
            dl = proj([(-hw, ly, 0.0), (hw, ly, 0.0),
                       (hw, ly, 0.5), (-hw, ly, 0.5)])
            if dl is not None:
                c.create_polygon([v for uv in dl for v in uv],
                                 outline="#7bd88f", fill="", width=2,
                                 dash=(6, 4), tags="robot")
                c.create_text(dl[3][0], dl[3][1] - 8, anchor="sw",
                              fill="#7bd88f", font=("Consolas", 9),
                              tags="robot",
                              text="결정선 %.0f cm" % cfg.DECIDE_LINE_CM)

        #  ---- 확정된 목적지 ----
        dec = self._decided_point()
        if dec is not None:
            dp = proj([dec])
            if dp is not None:
                u, v = dp[0]
                c.create_oval(u - 11, v - 11, u + 11, v + 11,
                              outline="#4fd07a", width=2, tags="robot")
                c.create_line(u - 16, v, u + 16, v, fill="#4fd07a",
                              width=1, tags="robot")
                c.create_line(u, v - 16, u, v + 16, fill="#4fd07a",
                              width=1, tags="robot")
                c.create_text(u + 18, v + 14, anchor="w", fill="#4fd07a",
                              font=("Consolas", 9), tags="robot",
                              text="확정 X %+.0f  Z %.0f"
                                   % (dec[0] * 1000, dec[2] * 1000))
        #  ---- 타격 창 ----
        #  창은 높이마다 좌우로 움직인다(라켓이 힌지보다 앞서 있어서).
        #  그래서 사각형이 아니라 **높이별 좌우 한계를 이은 띠**로 그린다.
        p = self.planner
        zl, zr = [], []
        n = 12
        for i in range(n + 1):
            zz = p.z_reach[0] + (p.z_reach[1] - p.z_reach[0]) * i / n
            lo, hi = p._x_limits(zz)
            zl.append((lo / 1000.0, y, zz / 1000.0))
            zr.append((hi / 1000.0, y, zz / 1000.0))
        band = proj(zl + list(reversed(zr)))
        if band is not None:
            #  좌표를 캔버스 안으로 잘라 둔다 (Tk 가 아주 큰 값에 취약하다)
            pts = [v for uv in band for v in
                   (max(-4000.0, min(4000.0, float(uv[0]))),
                    max(-4000.0, min(4000.0, float(uv[1]))))]
            c.create_line(*pts, *pts[:2], fill="#4ea3ff", width=1,
                          dash=(4, 3), tags="robot")
        #  라켓 중심이 실제로 갈 수 있는 범위(블레이드 여유 제외).
        #  ROI 밖은 애초에 치지 않기로 했으므로 타격 창으로 잘라 둔다.
        #  (자르지 않으면 리니어 상한까지 뻗어 ROI 를 크게 넘어 보인다)
        def _clip(v):
            if cfg.HIT_X_MIN_MM is not None:
                v = max(v, float(cfg.HIT_X_MIN_MM))
            if cfg.HIT_X_MAX_MM is not None:
                v = min(v, float(cfg.HIT_X_MAX_MM))
            return v

        core = []
        for i in range(n + 1):
            zz = p.z_reach[0] + (p.z_reach[1] - p.z_reach[0]) * i / n
            w = self.kin.x_window_mm(zz)
            if w:
                core.append((_clip(w[0]) / 1000.0, y, zz / 1000.0))
        for i in range(n, -1, -1):
            zz = p.z_reach[0] + (p.z_reach[1] - p.z_reach[0]) * i / n
            w = self.kin.x_window_mm(zz)
            if w:
                core.append((_clip(w[1]) / 1000.0, y, zz / 1000.0))
        cw = proj(core) if core else None
        if cw is not None:
            pts = [v for uv in cw for v in
                   (max(-4000.0, min(4000.0, float(uv[0]))),
                    max(-4000.0, min(4000.0, float(uv[1]))))]
            c.create_line(*pts, *pts[:2], fill="#2f6ea8", width=1,
                          tags="robot")
        #  막대 (힌지 -> 손목 축) 와 라켓 목 (손목 축 -> 라켓 중심)
        arm = proj([hinge, wrist])
        neck = proj([wrist, paddle])
        if neck is not None:
            c.create_line(*neck[0], *neck[1], fill="#ff8c42", width=3,
                          capstyle="round", tags="robot")
        if arm is not None:
            c.create_line(*arm[0], *arm[1], fill="#b58cff", width=4,
                          capstyle="round", tags="robot")
            c.create_oval(arm[0][0] - 3, arm[0][1] - 3,
                          arm[0][0] + 3, arm[0][1] + 3,
                          outline="#ddd", fill="#555", tags="robot")
        #  블레이드 (라켓 평면 위의 원)
        r = g.blade_radius_mm / 1000.0
        ring = proj([(paddle[0] + r * math.cos(a), y,
                      paddle[2] + r * math.sin(a))
                     for a in np.linspace(0, 2 * math.pi, 25)])
        if ring is not None:
            c.create_polygon([v for uv in ring for v in uv],
                             outline="#ff8c42", fill="", width=2, tags="robot")
        pc = proj([paddle])
        if pc is None:
            self._overlay_note("투영 실패 (카메라 뒤 · 라켓 X %+.0f Y %+.0f "
                               "Z %.0f mm)"
                               % (x_mm, cfg.PADDLE_PLANE_Y_MM, z_mm))
        else:
            u, v = pc[0]
            inside = (0 <= u < self.view3d_w and 0 <= v < self.view3d_h)
            if inside:
                self._overlay_note("정상 (화면 %d,%d)" % (u, v),
                                   quiet=True)
            else:
                #  캔버스 밖이면 가장자리에 화살표라도 남겨 어느 쪽인지 보인다
                eu = max(12, min(self.view3d_w - 12, u))
                ev = max(12, min(self.view3d_h - 12, v))
                c.create_oval(eu - 8, ev - 8, eu + 8, ev + 8,
                              outline="#ff6b6b", width=2, tags="robot")
                c.create_text(eu, ev + 16, anchor="n", fill="#ff6b6b",
                              font=("Consolas", 9), tags="robot",
                              text="라켓 화면 밖")
                self._overlay_note(
                    "캔버스 밖 — 화면 좌표 (%d, %d), 캔버스 %dx%d, "
                    "라켓 X %+.0f Y %+.0f Z %.0f mm"
                    % (u, v, self.view3d_w, self.view3d_h,
                       x_mm, cfg.PADDLE_PLANE_Y_MM, z_mm))
            u, v = pc[0]
            c.create_line(u - 6, v, u + 6, v, fill="#ff8c42", width=1,
                          tags="robot")
            c.create_line(u, v - 6, u, v + 6, fill="#ff8c42", width=1,
                          tags="robot")
            c.create_text(u + 10, v - 12, anchor="w", fill="#ff8c42",
                          font=("Consolas", 9), tags="robot",
                          text="라켓 X %+.0f  Z %.0f" % (x_mm, z_mm))
        self._paddle_now = (x_mm / 1000.0, y, z_mm / 1000.0)

    def _draw_fit(self):
        now = time.perf_counter()
        if now - self._fit_last_draw < 0.05:
            return
        self._fit_last_draw = now
        try:
            img = self.fit.image()
            self._draw_robot_on_fit(img)
            if self.fit_scale < 0.999:
                img = cv2.resize(img, (self.fit_w, self.fit_h),
                                 interpolation=cv2.INTER_AREA)
            self._fit_photo = _to_photo(img)
            self.canvas_fit.delete("all")
            self.canvas_fit.create_image(0, 0, image=self._fit_photo,
                                         anchor="nw")
            self.lbl_fit.config(text=self.fit.status_text())
        except Exception:
            pass

    def _draw_robot_on_fit(self, img):
        """궤적 예측 그림에도 라켓의 현재 실제 위치를 겹쳐 그린다.

        위 패널은 Y-Z(측면), 아래 패널은 Y-X(위) 다.  블레이드는 라켓 평면
        (Y 고정)에 놓인 원이라, 두 패널에서는 각각 세로 선분으로 보인다.
        """
        if not cfg.SHOW_PADDLE_IN_3D or self.worker is None:
            return
        try:
            th, cmd = self.worker.backend.current_pose()
            x_mm, z_mm = self.kin.forward(float(th), float(cmd))
            y = cfg.PADDLE_PLANE_Y_MM / 1000.0
            x, z = x_mm / 1000.0, z_mm / 1000.0
            r = self.kin.g.blade_radius_mm / 1000.0
            f = self.fit
            col = (66, 140, 255)          # 주황 (BGR)

            #  결정선 (두 패널 모두 세로선)
            if cfg.DECIDE_LINE_ENABLE:
                ly = float(cfg.DECIDE_LINE_CM) / 100.0
                dcol = (143, 216, 123)    # 연두 (BGR)
                cv2.line(img, f._pxA(ly, Z_TOP_M), f._pxA(ly, Z_BOT_M),
                         dcol, 2)
                cv2.line(img, f._pxB(ly, -f.hw), f._pxB(ly, f.hw), dcol, 2)
                cv2.putText(img, "decide %.0fcm" % cfg.DECIDE_LINE_CM,
                            (f._pxA(ly, Z_TOP_M)[0] + 4,
                             f._pxA(ly, Z_TOP_M)[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, dcol, 1)

            #  확정된 목적지
            dec = self._decided_point()
            if dec is not None:
                gcol = (122, 208, 79)     # 초록 (BGR)
                pa = f._pxA(dec[1], dec[2])
                pb = f._pxB(dec[1], dec[0])
                for pt in (pa, pb):
                    cv2.circle(img, pt, 8, gcol, 2)
                    cv2.drawMarker(img, pt, gcol, cv2.MARKER_CROSS, 16, 1)
                cv2.putText(img, "target Z %.0f" % (dec[2] * 1000),
                            (pa[0] + 10, pa[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, gcol, 1)
                cv2.putText(img, "target X %+.0f" % (dec[0] * 1000),
                            (pb[0] + 10, pb[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, gcol, 1)
            #  라켓 평면 위치 (세로 기준선)
            cv2.line(img, f._pxA(y, Z_TOP_M), f._pxA(y, Z_BOT_M),
                     (90, 90, 90), 1)
            cv2.line(img, f._pxB(y, -f.hw), f._pxB(y, f.hw), (90, 90, 90), 1)
            #  블레이드 (Y-Z 패널: Z 방향 선분 / Y-X 패널: X 방향 선분)
            cv2.line(img, f._pxA(y, z - r), f._pxA(y, z + r), col, 3)
            cv2.line(img, f._pxB(y, x - r), f._pxB(y, x + r), col, 3)
            #  라켓 중심
            cv2.circle(img, f._pxA(y, z), 4, (255, 255, 255), -1)
            cv2.circle(img, f._pxB(y, x), 4, (255, 255, 255), -1)
            cv2.putText(img, "racket Z %.0f" % z_mm,
                        (f._pxA(y, z)[0] + 8, f._pxA(y, z)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            cv2.putText(img, "racket X %+.0f" % x_mm,
                        (f._pxB(y, x)[0] + 8, f._pxB(y, x)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        except Exception:
            pass

    def _update_panels(self, plan):
        try:
            if plan.target is not None:
                self.lbl_shot.config(
                    text=("X %+7.1f mm   Z %6.1f mm\n"
                          "임팩트까지 %5.0f ms      랠리 #%d"
                          % (plan.x_mm, plan.z_mm, plan.lead_sec * 1000,
                             plan.target.track_no)), fg=OK_C)
            elif plan.reason:
                soft = ("미미" in plan.reason or "이름" in plan.reason
                        or "없음" in plan.reason)
                self.lbl_shot.config(
                    text=("X %+7.1f mm   Z %6.1f mm\n%s"
                          % (plan.x_mm, plan.z_mm, plan.reason)),
                    fg=DIM if soft else WARN_C)

            self.lbl_turn.config(
                text=("%s%s" % (self.turn.label(),
                                "   (나가는 공 차단 중)"
                                if self.turn.blocked else "")),
                fg=OK_C if self.turn.who == HUMAN else BAD_C)
            self.lbl_rate.config(
                text="루프 %.0f Hz   예측 샘플 %.0f /s" % (self.loop_hz,
                                                       self.sample_hz))
            sol = self.worker.last_solution
            self.lbl_motion.config(
                text="%s\n%s" % (self.worker.state,
                                 "-" if sol is None else str(sol)))
            self.lbl_stats.config(text="%s\n%s" % (self.planner.summary(),
                                                   self.worker.summary()))
            be = self.worker.backend
            self.lbl_robot_info.config(
                text=("축: %s\n%s\n스윙 소요 %.0f ms -> 발사 리드타임 %.0f ms"
                      % (be.axes_text(), self.planner.describe_window(),
                         cfg.SWING_DURATION_SEC * 1000,
                         swing_lead_sec() * 1000)))
        except Exception:
            pass

    def _drain_console(self):
        chunks = []
        while True:
            try:
                chunks.append(self._log_q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        try:
            self.txt_console.configure(state="normal")
            self.txt_console.insert("end", "".join(chunks))
            n = int(self.txt_console.index("end-1c").split(".")[0])
            if n > CONSOLE_LINES:
                self.txt_console.delete("1.0", "%d.0" % (n - CONSOLE_LINES))
            self.txt_console.see("end")
            self.txt_console.configure(state="disabled")
        except Exception:
            pass

    def clear_trail(self):
        super().clear_trail()
        if self.fit is not None:
            self.fit.pred.clear()
        self._fed_t = None

    # ------------------------------------------------------------------
    def _check_tuning_conflict(self):
        """보정 파일이 설계값과 크게 다르면 알린다.

        settings/robot_tuning.json 은 robot_config.py 를 덮어쓴다.  기구
        상수를 코드에서 바꿔도 이 파일이 옛 값을 붙들고 있으면 그대로
        적용되지 않아, 화면과 실물이 어긋난 채로 돌아간다.
        """
        checks = (
            ("arm_x_offset_mm", cfg.ARM.arm_x_offset_mm, 20.0,
             "지령 0 에서 힌지의 X"),
            ("paddle_offset_mm", cfg.ARM.paddle_offset_mm, 10.0,
             "손목 -> 라켓 중심"),
            ("hinge_z_mm", cfg.ARM.hinge_z_mm, 10.0, "힌지 높이"),
            ("radius_mm", cfg.ARM.radius_mm, 10.0, "막대 길이"),
        )
        bad = []
        for key, design, tol, what in checks:
            cur = self.tune.get(key)
            if cur is None:
                continue
            if abs(float(cur) - float(design)) > tol:
                bad.append("  %s (%s): 보정 파일 %.1f  vs  설계값 %.1f"
                           % (key, what, float(cur), float(design)))
        if not bad:
            return
        msg = ("보정 파일(settings/robot_tuning.json)이 설계값과 다릅니다.\n"
               "이 파일이 우선하므로, 의도한 값이 아니면 [보정] 탭에서\n"
               "고치고 [적용 + 저장] 하세요.\n\n" + "\n".join(bad))
        print("[!] " + msg.replace("\n", "\n    "))
        self.after(400, lambda: messagebox.showwarning("보정값 확인", msg))

    def _preflight(self):
        from dynamixel_link import HAS_SDK
        import home_pose as _hp
        dll, mot = self._resolve_linear_paths()
        print("  점검  dynamixel-sdk : %s"
              % ("있음" if HAS_SDK else "없음 -> pip install dynamixel-sdk"))
        print("  점검  리니어 DLL    : %s" % (dll or "없음 (가상으로 대체)"))
        print("  점검  .mot 파일     : %s" % (mot or "없음"))
        print("  점검  기본 자세     : %s"
              % ("있음" if _hp.load() else "없음 -> set_home_pose.py 로 먼저 잡을 것"))
        print("  점검  백엔드        : %s"
              % ("실기 연결됨" if self._is_real() else "시뮬레이터"))

    def _banner(self):
        print("=" * 68)
        print(" 탁구 로봇 통합 실행")
        print("  %s" % self.planner.describe_window())
        print("  스윙 %.0f ms -> 발사 리드타임 %.0f ms"
              % (cfg.SWING_DURATION_SEC * 1000, swing_lead_sec() * 1000))
        print("  움직일 축: %s" % self.worker.backend.axes_text())
        print("  보정: %s" % tuning.describe(self.tune))
        self._preflight()
        self._check_tuning_conflict()
        print("  'A' 로 무장하기 전에는 축 지령이 나가지 않습니다.")
        print("=" * 68)

    def on_close(self):
        if self.worker is not None:
            print("\n[통계] %s" % self.planner.summary())
            print("[통계] %s" % self.worker.summary())
            #  ★ 종료할 때는 무조건 리니어를 원위치(지령 0)로 되돌린다.
            #    엔코더가 없어 지령값을 추적하는 축이라, 다음 실행이
            #    같은 자리에서 시작해야 한다.
            if cfg.PARK_LINEAR_CMD_ON_EXIT is not None:
                try:
                    self._status("리니어를 원위치로 되돌리는 중입니다...")
                    self.update_idletasks()
                    self.worker.armed = False
                    be = self.worker.backend
                    print("[종료] 리니어 원위치 복귀 시작 (현재 %.2f -> %.2f)"
                          % (be.current_pose()[1],
                             cfg.PARK_LINEAR_CMD_ON_EXIT))
                    be.park_linear(float(cfg.PARK_LINEAR_CMD_ON_EXIT),
                                   timeout=cfg.PARK_TIMEOUT_SEC)
                    print("[종료] 리니어 복귀 완료 (%.2f)"
                          % be.current_pose()[1])
                except Exception as e:
                    print("[종료] 리니어 복귀 실패: %s" % e)
            self.worker.stop()
            self.worker = None
        sys.stdout = self._stdout_backup
        super().on_close()


# ============================================================================
# 엔트리 포인트
# ============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="탁구 로봇 통합 실행 (비전 + 예측 + 모션)")
    ap.add_argument("--cam0", type=int, default=None)
    ap.add_argument("--cam1", type=int, default=None)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--backend", default="dshow",
                    choices=list(tb.BACKENDS.keys()))
    ap.add_argument("--preview-scale", type=float, default=PREVIEW_SCALE)
    ap.add_argument("--sim", action="store_true",
                    help="카메라 없이 합성 공으로 파이프라인 점검")
    ap.add_argument("--sim-offset-ms", type=float, default=3.0)
    ap.add_argument("--arm", action="store_true",
                    help="시작부터 무장 (권장하지 않음)")
    ap.add_argument("--real", dest="real", action="store_true",
                    help="실기 연결 (다이나믹셀 + 리니어)")
    ap.add_argument("--sim-robot", dest="real", action="store_false",
                    help="로봇을 시뮬레이터로 (카메라는 그대로 씀)")
    ap.add_argument("--port", default=cfg.DXL_PORT)
    ap.add_argument("--baud", type=int, default=cfg.DXL_BAUDRATE)
    ap.add_argument("--no-linear", dest="ax_linear", action="store_false")
    ap.add_argument("--no-tilt", dest="ax_tilt", action="store_false")
    ap.add_argument("--no-swing", dest="ax_swing", action="store_false")
    ap.set_defaults(arm=cfg.START_ARMED, real=cfg.CONNECT_REAL_BY_DEFAULT,
                    ax_linear=True, ax_tilt=True, ax_swing=True)
    args = ap.parse_args(argv)
    #  명령줄로 끈 축은 설정에 반영한다 (UI 에서 다시 켤 수 있다)
    cfg.ENABLE_LINEAR = cfg.ENABLE_LINEAR and args.ax_linear
    cfg.ENABLE_TILT = cfg.ENABLE_TILT and args.ax_tilt
    cfg.ENABLE_SWING = cfg.ENABLE_SWING and args.ax_swing
    return args


def main(argv=None):
    args = parse_args(argv)
    tb.print_env_banner()
    try:
        app = RobotTrackerApp(args)
    except FileNotFoundError as e:
        print("[!] 설정 파일이 없습니다:\n%s" % e)
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("설정 없음",
                                 "%s\n\nsetup_extrinsics.py 를 먼저 실행하세요."
                                 % e)
            r.destroy()
        except Exception:
            pass
        return 1
    try:
        app.mainloop()
    finally:
        #  창을 정상으로 닫지 않고 빠져나온 경우에도 원위치로 되돌린다
        try:
            if getattr(app, "worker", None) is not None:
                app.worker.armed = False
                if cfg.PARK_LINEAR_CMD_ON_EXIT is not None:
                    app.worker.backend.park_linear(
                        float(cfg.PARK_LINEAR_CMD_ON_EXIT),
                        timeout=cfg.PARK_TIMEOUT_SEC)
                app.worker.stop()
        except Exception as e:
            print("[종료] 마무리 중 오류: %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
