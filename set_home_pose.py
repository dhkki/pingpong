#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set_home_pose.py
================
기본 자세 잡기 — 모터를 교체했거나 원점이 어긋났을 때 쓴다.

절차
----
    1. [연결] 후 [확장 위치 모드 확인] — 전 축이 모드 4 여야 한다
    2. [전체 토크 끄기] 를 누르고 **손으로** 자세를 잡는다
         · 막대를 수직으로 (theta = 0)
         · 라켓 면을 기준 각도로
         · 스윙 원판을 라켓 평면 위치(스윙 각도 0)로
    3. 화면의 raw 값이 안정되면 [기준 자세로 저장]
    4. [검증] — 토크를 켜고 그 자리를 유지시킨 뒤 처짐을 잰다

손으로 잡기 어려운 미세 조정은 조그 버튼(±0.5° / ±5°)을 쓴다.  조그는
토크가 켜져 있어야 동작한다.

실행
----
    python set_home_pose.py                    # COM3, 1000000
    python set_home_pose.py --port COM4
    python set_home_pose.py --simulate         # 하드웨어 없이 절차 확인

안전
----
* 창을 닫으면 토크를 끄고 나간다.  막대가 내려앉으므로 손으로 받치거나
  낮은 자세에서 종료할 것.
* 조그는 한 번에 최대 5°만 움직인다.
* 이 도구는 스윙을 하지 않는다.  ID 10 도 조그로만 움직인다.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import home_pose as hp
from dynamixel_link import (DynamixelError, MODE_EXTENDED_POSITION,
                            RAW_PER_REV, deg_to_raw, make_link,
                            nearest_equivalent, raw_to_deg)

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
FONT_BIG = ("Consolas", 15)

JOG_VELOCITY = 20           # 아주 느리게
JOG_ACCELERATION = 8
POLL_MS = 120


class HomePoseApp(tk.Tk):

    def __init__(self, args):
        super().__init__()
        self.title("기본 자세 잡기")
        self.configure(bg=BG)
        self.args = args
        self.ids = tuple(int(x) for x in args.ids.split(","))
        self.link = make_link(args.port, args.baud, self.ids,
                              simulate=args.simulate)
        self.connected = False
        self.torque_on = False
        self.raw = {i: None for i in self.ids}
        self.previous = hp.load()
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
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="포트", bg=PANEL, fg=DIM, font=FONT).pack(side="left")
        self.var_port = tk.StringVar(value=self.args.port)
        tk.Entry(row, textvariable=self.var_port, width=10, bg="#15151a",
                 fg=FG, font=FONT_M, insertbackground=FG).pack(side="left",
                                                               padx=4)
        tk.Label(row, text="보레이트", bg=PANEL, fg=DIM,
                 font=FONT).pack(side="left", padx=(8, 0))
        self.var_baud = tk.StringVar(value=str(self.args.baud))
        tk.Entry(row, textvariable=self.var_baud, width=9, bg="#15151a",
                 fg=FG, font=FONT_M, insertbackground=FG).pack(side="left",
                                                               padx=4)
        self.btn_conn = ttk.Button(row, text="연결", width=10,
                                   command=self.connect)
        self.btn_conn.pack(side="left", padx=6)
        ttk.Button(row, text="확장 위치 모드 확인", width=18,
                   command=self.check_mode).pack(side="left", padx=2)

        f = self._section(root, "현재 위치")
        self.rows = {}
        for i in self.ids:
            r = tk.Frame(f, bg=PANEL)
            r.pack(fill="x", pady=1)
            tk.Label(r, text="ID %-2d  %s" % (i, hp.ROLE.get(i, "")),
                     bg=PANEL, fg=FG, font=FONT, width=16,
                     anchor="w").pack(side="left")
            v = tk.Label(r, text="----", bg=PANEL, fg=DIM, font=FONT_BIG,
                         width=9, anchor="e")
            v.pack(side="left")
            d = tk.Label(r, text="", bg=PANEL, fg=DIM, font=FONT_M, width=10,
                         anchor="e")
            d.pack(side="left", padx=(4, 10))
            for label, deg in (("-5°", -5.0), ("-0.5°", -0.5),
                               ("+0.5°", +0.5), ("+5°", +5.0)):
                ttk.Button(r, text=label, width=6,
                           command=lambda k=i, g=deg: self.jog(k, g)
                           ).pack(side="left", padx=1)
            self.rows[i] = (v, d)

        f = self._section(root, "토크")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        self.btn_torque = ttk.Button(row, text="전체 토크 끄기 (손으로 잡기)",
                                     width=28, command=self.toggle_torque)
        self.btn_torque.pack(side="left", padx=2)
        ttk.Button(row, text="현재 자리에서 잡기", width=18,
                   command=self.hold_here).pack(side="left", padx=2)
        tk.Label(f, text=("토크를 끄면 막대가 내려앉습니다. 손으로 받친 뒤에 "
                          "끄세요. 자세를 잡고 값이 안정되면 저장합니다."),
                 bg=PANEL, fg=DIM, font=FONT, anchor="w", wraplength=560,
                 justify="left").pack(fill="x", pady=(6, 0))

        f = self._section(root, "저장")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        ttk.Button(row, text="기준 자세로 저장", width=18,
                   command=self.save).pack(side="left", padx=2)
        ttk.Button(row, text="검증", width=12,
                   command=self.verify).pack(side="left", padx=2)
        ttk.Button(row, text="저장값 보기", width=12,
                   command=self.show_saved).pack(side="left", padx=2)
        self.lbl_prev = tk.Label(f, bg=PANEL, fg=DIM, font=FONT_M,
                                 anchor="w", justify="left")
        self.lbl_prev.pack(fill="x", pady=(6, 0))
        self.lbl_prev.config(
            text=("이전 저장값 없음" if not self.previous
                  else "이전 저장값  "
                       + "  ".join("ID%d %d" % (k, self.previous[k])
                                   for k in sorted(self.previous))))

        self.lbl_status = tk.Label(root, text="연결하세요.", bg=BG, fg=WARN_C,
                                   font=FONT, anchor="w", wraplength=620,
                                   justify="left")
        self.lbl_status.pack(fill="x", pady=(10, 0))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _say(self, msg, color=DIM):
        self.lbl_status.config(text=msg, fg=color)
        print("[기본자세] %s" % msg)

    # ------------------------------------------------------------- 연결
    def connect(self):
        if self.connected:
            self._safe_torque_off()
            self.link.close()
            self.connected = False
            self.btn_conn.config(text="연결")
            self._say("연결을 끊었습니다.", WARN_C)
            return
        try:
            self.link.port_name = self.var_port.get().strip()
            self.link.baudrate = int(self.var_baud.get())
            self.link.open()
        except (DynamixelError, ValueError, OSError) as e:
            messagebox.showerror("연결 실패", str(e))
            return
        found = self.link.ping_all()
        dead = [i for i, m in found.items() if m is None]
        self.connected = True
        self.btn_conn.config(text="연결 끊기")
        if dead:
            self._say("응답 없는 모터: %s — 배선과 ID 를 확인하세요."
                      % ", ".join("ID %d" % i for i in dead), BAD_C)
        else:
            self._say("연결됨. 전 축 응답 확인. 다음은 [확장 위치 모드 확인].",
                      OK_C)

    def check_mode(self):
        if not self._require_conn():
            return
        changed, cur = [], {}
        try:
            for i in self.ids:
                if self.link.ensure_extended_mode(i):
                    changed.append(i)
                cur[i] = self.link.read_operating_mode(i)
        except DynamixelError as e:
            messagebox.showerror("모드 확인", str(e))
            return
        self.torque_on = False
        self._sync_torque_button()
        bad = [i for i, m in cur.items() if m != MODE_EXTENDED_POSITION]
        if bad:
            self._say("확장 위치 모드가 아닌 축이 있습니다: %s"
                      % ", ".join("ID %d" % i for i in bad), BAD_C)
        elif changed:
            self._say("확장 위치 모드로 바꿨습니다 (%s). 모드 변경 때문에 "
                      "토크가 꺼졌습니다."
                      % ", ".join("ID %d" % i for i in changed), WARN_C)
        else:
            self._say("전 축이 이미 확장 위치 모드입니다.", OK_C)

    # ------------------------------------------------------------- 토크
    def toggle_torque(self):
        if not self._require_conn():
            return
        target = not self.torque_on
        if target:
            #  그냥 토크만 켜면 레지스터에 남은 옛 목표로 튄다.
            #  현재 자리를 목표로 써 둔 뒤 켠다.
            self.hold_here()
            return
        else:
            if not messagebox.askyesno(
                    "토크 끄기",
                    "토크를 끄면 막대가 중력으로 내려앉습니다.\n"
                    "손으로 받치고 있습니까?"):
                return
            self.link.torque_all(False)
        self.torque_on = target
        self._sync_torque_button()
        self._say("토크 %s" % ("켬 — 현재 자리를 유지합니다" if target
                              else "끔 — 손으로 자세를 잡으세요"),
                  WARN_C if target else DIM)

    def _sync_torque_button(self):
        self.btn_torque.config(
            text="전체 토크 켜기" if not self.torque_on
            else "전체 토크 끄기 (손으로 잡기)")

    def hold_here(self):
        """현재 위치를 목표로 잡고 토크를 켠다."""
        if not self._require_conn():
            return
        try:
            for i in self.ids:
                #  ★ 목표를 먼저 쓰고 토크를 켠다 ★
                #    토크를 먼저 켜면 레지스터에 남은 옛 목표로 튀어
                #    모터가 한 바퀴를 돌 수 있다.
                pos = self.link.read_position(i)
                self.link.set_profile(i, JOG_VELOCITY, JOG_ACCELERATION)
                self.link.write_position(i, pos)     # 목표 먼저
                self.link.torque(i, True)            # 그다음 토크
        except DynamixelError as e:
            messagebox.showerror("잡기", str(e))
            return
        self.torque_on = True
        self._sync_torque_button()
        self._say("현재 자리에서 잡았습니다.", OK_C)

    def jog(self, dxl_id: int, deg: float):
        if not self._require_conn():
            return
        if not self.torque_on:
            self._say("조그는 토크가 켜져 있어야 합니다. [현재 자리에서 잡기] "
                      "를 먼저 누르세요.", WARN_C)
            return
        try:
            cur = self.link.read_position(dxl_id)
            self.link.set_profile(dxl_id, JOG_VELOCITY, JOG_ACCELERATION)
            #  현재 위치 기준 상대 이동이라 랩어라운드 걱정이 없다
            self.link.write_position(dxl_id, cur + deg_to_raw(deg))
        except DynamixelError as e:
            messagebox.showerror("조그", str(e))
            return
        self._say("ID %d %+.1f° 이동" % (dxl_id, deg))

    # ------------------------------------------------------------- 저장
    def save(self):
        if not self._require_conn():
            return
        raw = self.link.read_all()
        if any(v is None for v in raw.values()):
            messagebox.showerror("저장", "응답이 없는 모터가 있어 저장하지 "
                                       "않습니다.\n\n" + hp.describe(raw))
            return
        warn = hp.validate(raw, self.ids, previous=self.previous)
        for i, v in raw.items():
            if abs(int(v)) >= RAW_PER_REV:
                warn.append("ID %d 의 raw 가 %+d 로 한 바퀴(%d)를 넘습니다. "
                            "손으로 여러 바퀴 돌렸다면 제자리로 되돌린 뒤 "
                            "다시 저장하는 편이 안전합니다."
                            % (i, int(v), RAW_PER_REV))
        msg = ("아래 값을 기준 자세로 저장합니다.\n\n" + hp.describe(raw)
               + "\n\n막대가 수직이고, 라켓 면이 기준이며,\n"
                 "스윙 원판이 라켓 평면 위치에 있습니까?")
        if warn:
            msg = ("\n".join("! " + w for w in warn)) + "\n\n" + msg
        if not messagebox.askyesno("기준 자세로 저장", msg):
            return
        path = hp.save(raw, port=self.link.port_name,
                       baudrate=self.link.baudrate,
                       note="set_home_pose%s"
                            % (" (시뮬레이션)" if self.link.simulated else ""))
        self.previous = dict(raw)
        self.lbl_prev.config(text="이전 저장값  "
                                  + "  ".join("ID%d %d" % (k, raw[k])
                                              for k in sorted(raw)))
        self._say("저장했습니다: %s" % path, OK_C)
        print(hp.describe(raw))

    def verify(self):
        """토크를 켜 현재 자리를 유지시키고, 얼마나 밀리는지 잰다."""
        if not self._require_conn():
            return
        before = self.link.read_all()
        self.hold_here()
        self.after(700, lambda: self._verify_finish(before))

    def _verify_finish(self, before):
        after = self.link.read_all()
        lines = []
        worst = 0.0
        for i in self.ids:
            if before.get(i) is None or after.get(i) is None:
                continue
            d = after[i] - before[i]
            worst = max(worst, abs(raw_to_deg(d)))
            lines.append("  ID %-3d %+5d raw  (%+.2f°)"
                         % (i, d, raw_to_deg(d)))
        text = ("토크를 켠 뒤 각 축이 움직인 양입니다.\n"
                "중력 처짐과 서보 편차를 합친 값이라 0 이 되지는 않습니다.\n\n"
                + "\n".join(lines))
        if worst > 5.0:
            text += ("\n\n! 최대 %.1f° 나 움직였습니다. 자세를 다시 잡고 "
                     "저장하는 편이 좋습니다." % worst)
        else:
            text += "\n\n[OK] 최대 %.2f° — 정상 범위입니다." % worst
        messagebox.showinfo("검증", text)

    def show_saved(self):
        messagebox.showinfo("저장된 기본 자세", hp.info())

    # -------------------------------------------------------------- 루프
    def _poll(self):
        if self.connected:
            try:
                self.raw = self.link.read_all()
            except DynamixelError:
                pass
            for i in self.ids:
                v = self.raw.get(i)
                lv, ld = self.rows[i]
                if v is None:
                    lv.config(text="----", fg=BAD_C)
                    ld.config(text="")
                else:
                    lv.config(text="%d" % v,
                              fg=OK_C if self.torque_on else FG)
                    ld.config(text="%+.1f°" % raw_to_deg(v))
        self._after = self.after(POLL_MS, self._poll)

    def _require_conn(self) -> bool:
        if not self.connected:
            self._say("먼저 [연결] 하세요.", WARN_C)
            return False
        return True

    def _safe_torque_off(self):
        try:
            self.link.torque_all(False)
        except Exception:
            pass
        self.torque_on = False

    def on_close(self):
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except Exception:
                pass
        if self.connected:
            if messagebox.askyesno(
                    "종료", "종료하면서 토크를 끕니다.\n"
                            "막대가 내려앉으니 받치고 있습니까?\n\n"
                            "(아니오를 누르면 토크를 켠 채로 종료합니다)"):
                self._safe_torque_off()
            self.link.close()
        self.destroy()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="기본 자세 잡기")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--ids", default="1,2,5,10")
    ap.add_argument("--simulate", action="store_true",
                    help="하드웨어 없이 절차만 확인")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("기본 자세 잡기 — %s" % ("시뮬레이션" if args.simulate
                              else "%s @ %d" % (args.port, args.baud)))
    print(hp.info())
    HomePoseApp(args).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
