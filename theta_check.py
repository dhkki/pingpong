#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_check.py
==============
기구학 검증 도구 — theta 를 넣어 팔을 기울이고, 상판에서 라켓까지의 높이를
자로 재어 기록한다.  여러 각도에서 모으면 **실제 기구 상수를 역산**할 수 있다.

무엇을 확인하나
---------------
지금 기구학은 이렇게 가정하고 있다.

    Z(theta) = z0 + R·cos(theta)        z0 = 120mm(힌지 높이), R = 220mm(막대)

이 식이 맞는지, 맞다면 z0 와 R 이 도면값 그대로인지를 본다.  틀리면 라켓이
가야 할 높이와 실제 높이가 계속 어긋나므로, 아무리 예측을 잘해도 안 맞는다.

세 가지가 한꺼번에 드러난다
---------------------------
    1. z0, R 이 실제와 다른가          -> 최소자승으로 역산해 비교
    2. 감속비가 1:1 이 아닌가          -> 각도를 2배로 줘야 맞으면 2:1
    3. theta 부호가 반대인가            -> 눕히랬는데 세워지면 반대

쓰는 법
-------
    1. [연결] 후 [토크 켜기]
    2. theta 를 넣고 [이동]  (예: 0, -20, -40, -60, -80)
    3. 상판에서 **라켓 중심**까지의 높이를 자로 재어 입력하고 [기록]
    4. 5점 이상 모이면 [분석] — 실제 z0, R 과 지금 설정의 차이를 보여준다

    python theta_check.py --port COM3
    python theta_check.py --simulate        # 하드웨어 없이 화면만 확인

주의
----
* 기본 자세(settings/home_pose.json)가 있어야 한다.  없으면 set_home_pose.py.
* 프로파일을 아주 느리게 잡아 두었다.  그래도 손은 치우고 움직일 것.
* 창을 닫으면 토크를 끈다.  막대가 내려앉으니 받치고 닫을 것.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import numpy as np

import home_pose as hp
import robot_config as cfg
import robot_tuning as tuning
from dynamixel_link import (DynamixelError, MODE_EXTENDED_POSITION,
                            deg_to_raw, make_link, raw_to_deg)

BG = "#1e1e22"
FG = "#e8e8ec"
ACC = "#4ea3ff"
OK_C = "#4fd07a"
WARN_C = "#ffc451"
BAD_C = "#ff6b6b"
DIM = "#8b8b95"
PANEL = "#26262c"
FONT = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 9, "bold")
FONT_M = ("Consolas", 10)
FONT_BIG = ("Consolas", 14)

BAR_A_ID, BAR_B_ID, PADDLE_ID = 1, 2, 5

#  아주 느리게 움직인다 (사람이 옆에 있는 작업)
JOG_VELOCITY = 20
JOG_ACCELERATION = 8
POLL_MS = 150

RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "records")


# ============================================================================
# 분석
# ============================================================================

def fit_geometry(samples):
    """[(theta_deg, z_mm), ...] -> 실제 z0, R 을 최소자승으로 역산.

        Z = z0 + R·cos(theta)   ->   [1, cos(theta)] · [z0, R]^T = Z

    반환 (z0, R, rms, 잔차 목록). 점이 2개 미만이면 None.
    """
    if len(samples) < 2:
        return None
    th = np.array([math.radians(t) for t, _ in samples])
    z = np.array([v for _, v in samples], dtype=float)
    A = np.column_stack([np.ones_like(th), np.cos(th)])
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    z0, R = float(sol[0]), float(sol[1])
    pred = A @ sol
    res = z - pred
    rms = float(np.sqrt(np.mean(res ** 2)))
    return z0, R, rms, [float(r) for r in res]


def fit_with_gear(samples):
    """감속비까지 함께 본다.

        Z = z0 + R·cos(k·theta)

    k 를 0.25~4 범위에서 훑어 잔차가 가장 작은 값을 고른다.
    k 가 1 에서 크게 벗어나면 모터 각도와 막대 각도가 1:1 이 아니라는 뜻.
    """
    if len(samples) < 3:
        return None
    best = None
    for k in np.arange(0.25, 4.001, 0.01):
        th = np.array([math.radians(k * t) for t, _ in samples])
        z = np.array([v for _, v in samples], dtype=float)
        A = np.column_stack([np.ones_like(th), np.cos(th)])
        sol, *_ = np.linalg.lstsq(A, z, rcond=None)
        rms = float(np.sqrt(np.mean((z - A @ sol) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(k), float(sol[0]), float(sol[1]))
    rms, k, z0, R = best
    return z0, R, k, rms


# ============================================================================
# 앱
# ============================================================================

class ThetaCheckApp(tk.Tk):

    def __init__(self, args):
        super().__init__()
        self.title("기구학 검증 — theta 대 높이")
        self.configure(bg=BG)
        self.args = args
        self.ids = (BAR_A_ID, BAR_B_ID, PADDLE_ID)
        self.link = make_link(args.port, args.baud, self.ids,
                              simulate=args.simulate)
        #  메인 프로그램에서 맞춰 둔 보정값(부호 등)을 그대로 쓴다
        try:
            tuning.apply_to_config(tuning.load())
        except Exception as e:
            print("[검증] 보정값을 읽지 못했습니다: %s" % e)
        self.home = hp.load()
        self.connected = False
        self.torque_on = False
        self.theta_cmd = 0.0
        self.samples = []          # [(theta_deg, z_mm)]
        self._after = None

        self._build()
        self._poll()

    # ------------------------------------------------------------------ UI
    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=ACC, font=FONT_B,
                 anchor="w").pack(fill="x", pady=(8, 0))
        f = tk.Frame(parent, bg=PANEL, padx=10, pady=8)
        f.pack(fill="x")
        return f

    def _build(self):
        root = tk.Frame(self, bg=BG, padx=10, pady=10)
        root.pack(fill="both", expand=True)

        f = self._section(root, "연결")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        tk.Label(row, text="포트", bg=PANEL, fg=DIM, font=FONT).pack(side="left")
        self.var_port = tk.StringVar(value=self.args.port)
        tk.Entry(row, textvariable=self.var_port, width=9, bg="#15151a",
                 fg=FG, font=FONT_M, insertbackground=FG).pack(side="left",
                                                               padx=4)
        self.btn_conn = ttk.Button(row, text="연결", width=10,
                                   command=self.connect)
        self.btn_conn.pack(side="left", padx=4)
        self.btn_torque = ttk.Button(row, text="토크 켜기", width=12,
                                     command=self.toggle_torque)
        self.btn_torque.pack(side="left", padx=4)
        tk.Label(row, text=("기본 자세 %s" % ("있음" if self.home else "없음!")),
                 bg=PANEL, fg=(OK_C if self.home else BAD_C),
                 font=FONT).pack(side="left", padx=10)

        f = self._section(root, "각도 지령")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        tk.Label(row, text="theta °", bg=PANEL, fg=DIM, font=FONT,
                 width=8, anchor="w").pack(side="left")
        self.var_theta = tk.DoubleVar(value=0.0)
        for lab, d in (("-5", -5.0), ("-1", -1.0)):
            ttk.Button(row, text=lab, width=3,
                       command=lambda x=d: self.bump(x)).pack(side="left")
        tk.Entry(row, textvariable=self.var_theta, width=7, bg="#15151a",
                 fg=FG, font=FONT_M, insertbackground=FG,
                 justify="right").pack(side="left", padx=2)
        for lab, d in (("+1", 1.0), ("+5", 5.0)):
            ttk.Button(row, text=lab, width=3,
                       command=lambda x=d: self.bump(x)).pack(side="left")
        tk.Scale(row, from_=cfg.ARM.theta_min_deg, to=cfg.ARM.theta_max_deg,
                 resolution=1, orient="horizontal", variable=self.var_theta,
                 bg=PANEL, fg=FG, font=("Consolas", 7), highlightthickness=0,
                 troughcolor="#15151a", length=200, sliderlength=14,
                 width=9, showvalue=False).pack(side="left", padx=6)
        ttk.Button(row, text="이동", width=8,
                   command=self.move).pack(side="left", padx=4)

        wr = tk.Frame(f, bg=PANEL); wr.pack(fill="x", pady=(6, 0))
        tk.Label(wr, text="손목 ID5", bg=PANEL, fg=DIM, font=FONT,
                 width=8, anchor="w").pack(side="left")
        self.var_id5rev = tk.BooleanVar(value=cfg.ID5_SIGN < 0)
        tk.Checkbutton(wr, text="방향 반전", variable=self.var_id5rev,
                       command=self.apply_id5, bg=PANEL, fg=FG,
                       selectcolor=BG, font=FONT, activebackground=PANEL,
                       activeforeground=FG).pack(side="left")
        tk.Label(wr, text="보정 비율", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=(10, 2))
        self.var_id5ratio = tk.DoubleVar(value=cfg.ID5_COMPENSATION_RATIO)
        tk.Scale(wr, from_=0.0, to=2.0, resolution=0.05, orient="horizontal",
                 variable=self.var_id5ratio, bg=PANEL, fg=FG,
                 font=("Consolas", 7), highlightthickness=0,
                 troughcolor="#15151a", length=130, sliderlength=14, width=9,
                 command=lambda _v: self.apply_id5()).pack(side="left")
        self.lbl_id5 = tk.Label(wr, text="", bg=PANEL, fg=DIM, font=FONT_M)
        self.lbl_id5.pack(side="left", padx=6)
        tk.Label(f, text=("theta 를 훑는 동안 라켓 면이 같은 각도를 유지하면 "
                          "부호와 비율이 맞습니다."),
                 bg=PANEL, fg=DIM, font=FONT, anchor="w").pack(fill="x",
                                                               pady=(4, 0))

        seq = tk.Frame(f, bg=PANEL); seq.pack(fill="x", pady=(6, 0))
        tk.Label(seq, text="빠른 지정", bg=PANEL, fg=DIM, font=FONT,
                 width=8, anchor="w").pack(side="left")
        for t in (0, -10, -20, -30, -40, -50, -60, -70, -80):
            ttk.Button(seq, text="%d°" % t, width=4,
                       command=lambda x=t: self.goto(x)).pack(side="left",
                                                              padx=1)

        f = self._section(root, "예상 · 실측")
        self.lbl_pred = tk.Label(f, bg=PANEL, fg=FG, font=FONT_BIG,
                                 justify="left", anchor="w", height=3)
        self.lbl_pred.pack(fill="x")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text="잰 높이 mm", bg=PANEL, fg=DIM, font=FONT,
                 width=10, anchor="w").pack(side="left")
        self.var_meas = tk.StringVar()
        e = tk.Entry(row, textvariable=self.var_meas, width=8, bg="#15151a",
                     fg=FG, font=FONT_M, insertbackground=FG, justify="right")
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda _e: self.record())
        ttk.Button(row, text="기록", width=8,
                   command=self.record).pack(side="left", padx=4)
        ttk.Button(row, text="마지막 지우기", width=13,
                   command=self.undo).pack(side="left", padx=2)
        ttk.Button(row, text="전체 지우기", width=11,
                   command=self.clear).pack(side="left", padx=2)
        tk.Label(f, text="상판 윗면에서 라켓 중심까지의 수직 높이를 재세요.",
                 bg=PANEL, fg=DIM, font=FONT, anchor="w").pack(fill="x",
                                                               pady=(4, 0))

        f = self._section(root, "기록")
        self.txt = tk.Text(f, height=9, bg="#141418", fg=FG,
                           font=("Consolas", 9), highlightthickness=1,
                           highlightbackground="#444")
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="분석", width=10,
                   command=self.analyze).pack(side="left", padx=2)
        ttk.Button(row, text="CSV 저장", width=10,
                   command=self.save_csv).pack(side="left", padx=2)

        self.lbl_status = tk.Label(root, text="연결하세요.", bg=BG, fg=WARN_C,
                                   font=FONT, anchor="w", wraplength=700,
                                   justify="left")
        self.lbl_status.pack(fill="x", pady=(8, 0))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.apply_id5()
        self._refresh_pred()

    def _say(self, msg, color=DIM):
        self.lbl_status.config(text=msg, fg=color)
        print("[검증] %s" % msg)

    def _log(self, line):
        self.txt.configure(state="normal")
        self.txt.insert("end", line + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    # ------------------------------------------------------------- 연결
    def connect(self):
        if self.connected:
            self._torque(False)
            self.link.close()
            self.connected = False
            self.btn_conn.config(text="연결")
            self._say("연결을 끊었습니다.", WARN_C)
            return
        if self.home is None:
            messagebox.showerror("기본 자세 없음",
                                 "settings/home_pose.json 이 없습니다.\n"
                                 "set_home_pose.py 로 먼저 잡아 주세요.")
            return
        try:
            self.link.port_name = self.var_port.get().strip()
            self.link.baudrate = int(self.args.baud)
            self.link.open()
        except (DynamixelError, ValueError, OSError) as e:
            messagebox.showerror("연결 실패", str(e))
            return
        dead = [i for i, m in self.link.ping_all().items() if m is None]
        self.connected = True
        self.btn_conn.config(text="연결 끊기")
        if dead:
            self._say("응답 없는 모터: %s"
                      % ", ".join("ID %d" % i for i in dead), BAD_C)
            return
        try:
            for i in self.ids:
                if self.link.ensure_extended_mode(i):
                    self._log("ID %d 확장 위치 모드로 변경" % i)
        except DynamixelError as e:
            self._say("모드 확인 실패: %s" % e, BAD_C)
            return
        self.torque_on = False
        self.btn_torque.config(text="토크 켜기")
        self._say("연결됨. [토크 켜기] 후 각도를 지령하세요.", OK_C)

    def _torque(self, on):
        if not self.connected:
            return
        try:
            for i in self.ids:
                if on:
                    #  ★ 목표를 먼저 쓰고 토크를 켠다 ★
                    #    반대로 하면 옛 목표로 튀어 한 바퀴 돌 수 있다.
                    pos = self.link.read_position(i)
                    self.link.set_profile(i, JOG_VELOCITY, JOG_ACCELERATION)
                    self.link.write_position(i, pos)
                    self.link.torque(i, True)
                else:
                    self.link.torque(i, False)
        except DynamixelError as e:
            self._say("토크 설정 실패: %s" % e, BAD_C)
            return
        self.torque_on = bool(on)
        self.btn_torque.config(text="토크 끄기" if on else "토크 켜기")

    def toggle_torque(self):
        if not self.connected:
            self._say("먼저 연결하세요.", WARN_C)
            return
        if self.torque_on and not messagebox.askyesno(
                "토크 끄기", "막대가 내려앉습니다. 받치고 있습니까?"):
            return
        self._torque(not self.torque_on)
        self._say("토크 %s" % ("켬" if self.torque_on else "끔"))

    def apply_id5(self):
        """손목 방향과 보정 비율을 즉시 반영하고, 지금 각도로 다시 보낸다."""
        cfg.ID5_SIGN = -1 if self.var_id5rev.get() else 1
        cfg.ID5_COMPENSATION_RATIO = float(self.var_id5ratio.get())
        self.lbl_id5.config(text="부호 %+d  비율 %.2f"
                                 % (cfg.ID5_SIGN, cfg.ID5_COMPENSATION_RATIO))
        if self.connected and self.torque_on:
            self.move()

    # ------------------------------------------------------------- 이동
    def bump(self, d):
        self.var_theta.set(round(float(self.var_theta.get()) + d, 1))
        self._refresh_pred()

    def goto(self, t):
        self.var_theta.set(float(t))
        self.move()

    def move(self):
        try:
            th = float(self.var_theta.get())
        except (ValueError, tk.TclError):
            self._say("각도를 숫자로 넣으세요.", WARN_C)
            return
        th = max(cfg.ARM.theta_min_deg, min(cfg.ARM.theta_max_deg, th))
        self.var_theta.set(th)
        self._refresh_pred()
        if not self.connected:
            self._say("연결되지 않아 지령을 보내지 않았습니다 (예상값만 갱신).",
                      WARN_C)
            return
        if not self.torque_on:
            self._say("토크를 먼저 켜세요.", WARN_C)
            return
        d = deg_to_raw(th * cfg.BASE_SIGN)
        d5 = deg_to_raw(th * cfg.BASE_SIGN * cfg.ID5_SIGN
                        * cfg.ID5_COMPENSATION_RATIO)
        targets = {BAR_A_ID: self.home[BAR_A_ID] + d,
                   BAR_B_ID: self.home[BAR_B_ID] - d,
                   PADDLE_ID: self.home[PADDLE_ID] + d5}
        try:
            for i in self.ids:
                self.link.set_profile(i, JOG_VELOCITY, JOG_ACCELERATION)
            for i, raw in targets.items():
                self.link.write_position(i, raw)
        except DynamixelError as e:
            self._say("지령 실패: %s" % e, BAD_C)
            return
        self.theta_cmd = th
        self._say("theta %+.1f° 지령  (ID1 %d  ID2 %d  ID5 %d)"
                  % (th, targets[BAR_A_ID], targets[BAR_B_ID],
                     targets[PADDLE_ID]), OK_C)

    def _refresh_pred(self):
        try:
            th = float(self.var_theta.get())
        except (ValueError, tk.TclError):
            return
        from robot_kinematics import ArmKinematics
        g = cfg.ARM
        _, z = ArmKinematics().forward(th, 0.0)
        self.lbl_pred.config(
            text=("지령 theta   %+7.1f °\n"
                  "예상 높이 Z  %7.1f mm\n"
                  "  %.0f + %.0f·cos(th) %+.0f·sin(th)"
                  % (th, z, g.hinge_z_mm, g.radius_mm, g.paddle_perp_mm)))

    # ------------------------------------------------------------- 기록
    def record(self):
        try:
            z = float(self.var_meas.get())
        except ValueError:
            self._say("잰 높이를 숫자로 넣으세요.", WARN_C)
            return
        th = float(self.var_theta.get())
        from robot_kinematics import ArmKinematics
        pred = ArmKinematics().forward(th, 0.0)[1]
        self.samples.append((th, z))
        self._log("  theta %+6.1f°   실측 %7.1f mm   예상 %7.1f   차 %+7.1f"
                  % (th, z, pred, z - pred))
        self.var_meas.set("")
        self._say("기록 %d개" % len(self.samples), OK_C)

    def undo(self):
        if self.samples:
            self.samples.pop()
            self._log("  (마지막 기록 삭제, 남은 %d개)" % len(self.samples))

    def clear(self):
        self.samples = []
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")
        self._say("기록을 지웠습니다.")

    # ------------------------------------------------------------- 분석
    def analyze(self):
        if len(self.samples) < 2:
            messagebox.showinfo("분석", "최소 2점이 필요합니다. "
                                      "각도를 바꿔 가며 5점 이상 모으세요.")
            return
        g = cfg.ARM
        z0, R, rms, res = fit_geometry(self.samples)
        lines = ["측정 %d점으로 역산한 결과" % len(self.samples),
                 "",
                 "  힌지 높이 z0   실측 %7.1f mm   설정 %7.1f mm   차 %+.1f"
                 % (z0, g.hinge_z_mm, z0 - g.hinge_z_mm),
                 "  막대 길이 R    실측 %7.1f mm   설정 %7.1f mm   차 %+.1f"
                 % (R, g.radius_mm, R - g.radius_mm),
                 "  잔차 RMS       %7.1f mm" % rms,
                 ""]

        if rms > 15:
            lines.append("! 잔차가 큽니다. Z = z0 + R·cos(theta) 형태 자체가")
            lines.append("  실제와 다를 수 있습니다 (감속비, 부호, 링크 구조).")
        gear = fit_with_gear(self.samples)
        if gear is not None:
            gz0, gR, k, grms = gear
            lines.append("감속비까지 함께 맞춘 결과")
            lines.append("  각도 배수 k    %.2f   (1.00 이면 1:1)" % k)
            lines.append("  z0 %.1f mm   R %.1f mm   잔차 %.1f mm"
                         % (gz0, gR, grms))
            if abs(k - 1.0) > 0.1 and grms < rms * 0.7:
                lines.append("")
                lines.append("! 모터 각도와 막대 각도가 1:1 이 아닌 것 같습니다.")
                lines.append("  지령 theta 에 %.2f 를 곱해야 실제 각도가 됩니다."
                             % k)

        lines.append("")
        lines.append("설정에 반영하려면 robot_config.py 의 ArmGeometry 에서")
        lines.append("  hinge_z_mm = %.1f" % z0)
        lines.append("  radius_mm  = %.1f" % R)
        lines.append("로 바꾸세요. (도달 Z 범위 %.0f ~ %.0f mm)"
                     % (z0 + R * math.cos(math.radians(g.theta_min_deg)),
                        z0 + R))
        msg = "\n".join(lines)
        self._log("")
        for ln in lines:
            self._log(ln)
        messagebox.showinfo("분석", msg)

    def save_csv(self):
        if not self.samples:
            self._say("기록이 없습니다.", WARN_C)
            return
        os.makedirs(RECORD_DIR, exist_ok=True)
        path = os.path.join(RECORD_DIR, "theta_check_%s.csv"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
        g = cfg.ARM
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["theta_deg", "measured_z_mm", "predicted_z_mm",
                        "diff_mm", "cfg_hinge_z_mm", "cfg_radius_mm"])
            from robot_kinematics import ArmKinematics
            kin = ArmKinematics()
            for th, z in self.samples:
                pred = kin.forward(th, 0.0)[1]
                w.writerow([round(th, 2), round(z, 2), round(pred, 2),
                            round(z - pred, 2), g.hinge_z_mm, g.radius_mm])
        self._say("저장: %s" % path, OK_C)

    # -------------------------------------------------------------- 루프
    def _poll(self):
        if self.connected and self.home is not None:
            try:
                raw = self.link.read_position(BAR_A_ID)
                th = raw_to_deg(raw - self.home[BAR_A_ID]) / cfg.BASE_SIGN
                self.title("기구학 검증 — 현재 theta %+.1f°" % th)
            except DynamixelError:
                pass
        self._after = self.after(POLL_MS, self._poll)

    def on_close(self):
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except Exception:
                pass
        if self.connected:
            if messagebox.askyesno("종료", "토크를 끄고 종료합니다.\n"
                                         "막대를 받치고 있습니까?"):
                self._torque(False)
            self.link.close()
        self.destroy()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="기구학 검증 (theta 대 높이)")
    ap.add_argument("--port", default=cfg.DXL_PORT)
    ap.add_argument("--baud", type=int, default=cfg.DXL_BAUDRATE)
    ap.add_argument("--simulate", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("기구학 검증 — %s" % ("시뮬레이션" if args.simulate
                            else "%s @ %d" % (args.port, args.baud)))
    g = cfg.ARM
    print("현재 설정  힌지 %0.1f mm  막대 %0.1f mm  ->  Z = %.0f + %.0f·cos(theta)"
          % (g.hinge_z_mm, g.radius_mm, g.hinge_z_mm, g.radius_mm))
    ThetaCheckApp(args).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
