#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shot_planner.py
===============
예측기(trajectory_predictor) 의 통과 보고를 모션 계층의 ShotTarget 으로
바꾼다.  좌표계·단위 변환과 "칠 공인가" 판정이 전부 여기 모여 있다.

하는 일
-------
1. 단위 변환      예측기는 m, 기구학은 mm
2. 라켓 평면 정렬 예측기가 보는 통과면(Y=0)과 라켓 평면이 다르면 보정
3. 타격 창 판정   폭(X) 구간 · 높이(Z) 구간 밖이면 조준하지 않는다
4. 중복 억제      같은 트랙에서 의미 없는 변화는 다시 제출하지 않는다

모든 공을 칠 필요는 없다.  사람이 로봇이 치기 좋은 영역으로 넣어 준다는
전제이므로, 창 밖의 공은 조용히 흘려보내고 다음 공을 기다린다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import robot_config as cfg
from motion_control import ShotTarget
from robot_kinematics import ArmKinematics


@dataclass
class PlanResult:
    """한 프레임의 판정 결과."""
    target: ShotTarget | None = None
    accepted: bool = False
    reason: str = ""              # 거부 사유 (accepted=False 일 때)
    x_mm: float = 0.0
    z_mm: float = 0.0
    lead_sec: float = 0.0
    fresh: bool = False           # 이번 프레임에 새 예측이 있었는가
    z_clamped_from: float | None = None   # 높이를 잘라 냈으면 원래 값


class ShotPlanner:
    """예측 보고 -> ShotTarget."""

    def __init__(self, *, kinematics=None, verbose: bool = True):
        self.kin = kinematics or ArmKinematics()
        self.verbose = bool(verbose)

        #  ---- 타격 창 (mm) ----
        #  기구학이 주는 것은 '라켓 중심'이 갈 수 있는 범위다.  실제로는
        #  블레이드 반경만큼 더 넓은 공도 맞으므로 그만큼 여유를 준다.
        g = self.kin.g
        blade = (g.blade_radius_mm * g.blade_usable_frac
                 if cfg.HIT_Z_BLADE_MARGIN_MM is None
                 else float(cfg.HIT_Z_BLADE_MARGIN_MM))
        self.blade_margin_mm = blade
        #  X 창.  None 이면 기구학이 계산한 도달 구간을 쓴다.
        #  라켓이 힌지보다 앞서 있어 좌우가 대칭이 아니므로, 실제로 갈 수
        #  있는 범위를 쓰는 편이 안전하다(못 갈 공을 조준하지 않는다).
        #  높이마다 창이 좌우로 움직이므로 한 높이만 보면 실제보다 좁다.
        #  표시·기본 판정에는 전 범위의 합집합을 쓰고, 실제 채택 여부는
        #  그 공의 높이에 맞는 창으로 다시 본다(_x_ok).
        auto_lo, auto_hi = self.kin.x_span_mm()
        self.x_min = ((auto_lo if cfg.HIT_X_MIN_MM is None
                       else float(cfg.HIT_X_MIN_MM))
                      - float(cfg.HIT_X_BLADE_MARGIN_MM))
        self.x_max = ((auto_hi if cfg.HIT_X_MAX_MM is None
                       else float(cfg.HIT_X_MAX_MM))
                      + float(cfg.HIT_X_BLADE_MARGIN_MM))
        z_lo, z_hi = self.kin.z_limits_with_swing_mm()
        self.z_reach = (z_lo, z_hi)          # 라켓 중심 도달 범위 (표시용)
        self.z_min = (z_lo - blade if cfg.HIT_Z_MIN_MM is None
                      else float(cfg.HIT_Z_MIN_MM))
        self.z_max = (z_hi + blade if cfg.HIT_Z_MAX_MM is None
                      else float(cfg.HIT_Z_MAX_MM))

        self._rally = 0               # 판정기가 붙이는 랠리 번호
        self._frozen = None           # 결정선을 넘은 뒤 확정된 (x_mm, z_mm)
        self._frozen_from = None      # 확정한 순간의 공 위치(로그용)
        self._raw_track = None        # 예측기가 준 트랙 번호
        self._last_sample_t = None
        self._last_track = -1
        self._last_x = None
        self._last_z = None
        self._last_impact = None
        self._seq = 0
        self._reported_track = -1     # 콘솔에 사유를 이미 찍은 트랙

        self.stats = dict(seen=0, accepted=0, out_x=0, out_z=0,
                          stale=0, no_time=0)

    # ------------------------------------------------------------------
    def _x_limits(self, z_mm: float) -> tuple:
        """그 높이에서 실제로 닿는 X 구간.

        라켓이 힌지보다 앞서 있어 창이 높이에 따라 좌우로 움직인다.
        수직에 가까울수록 왼쪽으로 더 간다.  사용자가 창을 직접 지정했으면
        그 값과 교집합을 취한다.
        """
        w = self.kin.x_window_mm(float(z_mm))
        if w is None:
            return self.x_min, self.x_max
        m = float(cfg.HIT_X_BLADE_MARGIN_MM)
        lo, hi = w[0] - m, w[1] + m
        if cfg.HIT_X_MIN_MM is not None:
            lo = max(lo, float(cfg.HIT_X_MIN_MM) - m)
        if cfg.HIT_X_MAX_MM is not None:
            hi = min(hi, float(cfg.HIT_X_MAX_MM) + m)
        return lo, hi

    def describe_window(self) -> str:
        aim = ""
        line = ("   결정선 %.0f cm" % cfg.DECIDE_LINE_CM
                if cfg.DECIDE_LINE_ENABLE else "   결정선 없음")
        if cfg.AIM_Z_OFFSET_MM or cfg.AIM_X_OFFSET_MM:
            aim = ("   [조준 보정 X %+.0f  Z %+.0f]"
                   % (cfg.AIM_X_OFFSET_MM, cfg.AIM_Z_OFFSET_MM))
        return (("타격 창  X %+.0f ~ %+.0f mm(높이별 이동)   Z %.0f ~ %.0f mm"
                 % (self.x_min, self.x_max, self.z_min, self.z_max))
                + aim + line + "\n"
                + ("  (라켓 중심 도달 %.0f ~ %.0f + 블레이드 여유 %.0f mm)"
                   % (self.z_reach[0], self.z_reach[1],
                      self.blade_margin_mm)))

    # ------------------------------------------------------------------
    def plan(self, predictor, now: float | None = None) -> PlanResult:
        """예측기의 최신 통과 보고를 읽어 판정한다.

        predictor : ParabolaPredictor
        반환      : PlanResult (target 이 있으면 모션 워커에 제출하면 된다)
        """
        now = time.perf_counter() if now is None else now
        rep = getattr(predictor, "last_report", None)
        if rep is None:
            return PlanResult(reason="예측 없음")

        #  ---- 랠리 구분 ----
        #  예측기의 track_no 는 트랙이 '끝날 때' 증가하므로, 그것만 믿으면
        #  수동 초기화 등으로 번호가 겹칠 수 있다.  샘플 시각이 뒤로 가거나
        #  크게 끊긴 것도 새 랠리로 본다.
        raw = int(rep.get("track_no", 0))
        t_s = float(rep.get("t_sample", 0.0))
        impact = float(rep.get("t_impact", 0.0))
        new_rally = (self._raw_track is None
                     or raw != self._raw_track
                     or self._last_sample_t is None
                     or t_s < self._last_sample_t - 1e-6
                     or t_s - self._last_sample_t > 0.30)
        if new_rally:
            self._rally += 1
            self._seq = 0
            self._last_x = self._last_z = self._last_impact = None
            self._reported_track = -1
            self._frozen = None       # 새 공이면 다시 결정선부터
        self._raw_track = raw
        self._last_sample_t = t_s
        track = self._rally
        fresh = (self._last_impact is None
                 or abs(impact - self._last_impact) > 1e-9)

        #  ---- 단위/좌표 변환 (예측기 m -> mm) ----
        #  예측기는 진행축 Y = end_report_value 평면의 통과점을 준다.
        #  그 면이 라켓 평면과 다르면, 통과 시각을 라켓 평면 기준으로 옮긴다.
        x_mm = float(rep["X"]) * 1000.0
        z_mm = float(rep["Z"]) * 1000.0
        vy = float(rep["vY"])                 # m/s, 진행축 속도 (음수)
        plane_y_m = cfg.PADDLE_PLANE_Y_MM / 1000.0
        dy = plane_y_m - float(rep["Y"])      # 남은 진행축 거리
        if abs(dy) > 1e-6 and abs(vy) > 1e-6:
            dt = dy / vy
            impact += dt
            x_mm += float(rep["vX"]) * dt * 1000.0
            z_mm += (float(rep["vZ"]) * dt - 0.5 * 9.81 * dt * dt) * 1000.0

        #  ★ 조준 보정 ★ 원인 미상의 계통 오차를 임시로 상쇄한다
        x_mm += float(cfg.AIM_X_OFFSET_MM)
        z_mm += float(cfg.AIM_Z_OFFSET_MM)

        #  ---- 결정선 ----
        #  선을 넘기 전에는 목적지를 확정하지 않는다(잔무빙 방지).
        #  넘는 순간 한 번 확정하고, 이후에는 그 값을 그대로 쓴다.
        frozen_now = False
        if cfg.DECIDE_LINE_ENABLE:
            line_m = float(cfg.DECIDE_LINE_CM) / 100.0
            ball_y = rep.get("last_Y")
            if self._frozen is None:
                crossed = (ball_y is not None
                           and (ball_y <= line_m if float(rep.get("vY", -1)) < 0
                                else ball_y >= line_m))
                if not crossed:
                    return PlanResult(
                        x_mm=x_mm, z_mm=z_mm, lead_sec=impact - now,
                        fresh=fresh,
                        reason=("결정선 대기 (공 Y %.2f m, 선 %.2f m)"
                                % (ball_y if ball_y is not None else -1,
                                   line_m)))
                self._frozen = (x_mm, z_mm)
                self._frozen_from = ball_y
                frozen_now = True
            x_mm, z_mm = self._frozen

        lead = impact - now
        res = PlanResult(x_mm=x_mm, z_mm=z_mm, lead_sec=lead, fresh=fresh)

        if fresh:
            self.stats["seen"] += 1

        #  ---- 판정 ----
        if lead < cfg.MIN_LEAD_SEC:
            self.stats["no_time"] += 1 if fresh else 0
            res.reason = "시간 없음 (%.0f ms)" % (lead * 1000)
            self._log_once(track, res.reason)
            return res
        if lead > cfg.MAX_LEAD_SEC:
            res.reason = "아직 이름 (%.2f s)" % lead
            return res
        lo, hi = self._x_limits(z_mm)
        if not (lo <= x_mm <= hi):
            self.stats["out_x"] += 1 if fresh else 0
            res.reason = ("폭 창 밖 (X %+.0f mm, Z %.0f 에서 창 %+.0f~%+.0f)"
                          % (x_mm, z_mm, lo, hi))
            self._log_once(track, res.reason)
            return res
        z_clamped = None
        if not (self.z_min <= z_mm <= self.z_max):
            if not cfg.CLAMP_Z_TO_REACH:
                self.stats["out_z"] += 1 if fresh else 0
                res.reason = ("높이 창 밖 (Z %.0f mm, 창 %.0f~%.0f)"
                              % (z_mm, self.z_min, self.z_max))
                self._log_once(track, res.reason)
                return res
            #  ★ 포기하지 않는다 ★
            #  낮으면 갈 수 있는 가장 낮은 자세로, 높으면 가장 높은 자세로
            #  두고 그대로 휘두른다. 블레이드 반경 안에 들어오면 맞는다.
            self.stats["out_z"] += 1 if fresh else 0
            z_clamped = z_mm
            z_mm = max(self.z_reach[0], min(self.z_reach[1], z_mm))
            res.z_mm = z_mm
            self._log_once(track,
                           "!높이 한계 (Z %.0f mm) -> 라켓을 %s 두고 시도"
                           % (z_clamped,
                              "가장 낮게" if z_clamped < self.z_reach[0]
                              else "가장 높게"))

        #  ---- 잔변화 억제 ----
        #  목적지가 확정된 뒤에는 위치가 어차피 고정이므로 건너뛴다.
        #  임팩트 시각만 계속 갱신해 스윙 타이밍을 최신으로 유지한다.
        if (self._frozen is None and track == self._last_track
                and self._last_x is not None):
            dx = abs(x_mm - self._last_x)
            dz = abs(z_mm - self._last_z)
            band = (cfg.EARLY_DEADBAND_MM if lead > cfg.EARLY_DEADBAND_SEC
                    else cfg.MIN_RETARGET_MM)
            if dx < band and dz < band and abs(impact - self._last_impact) < 0.005:
                res.reason = "변화 미미"
                return res

        self._seq += 1
        self._last_track, self._last_x = track, x_mm
        self._last_z, self._last_impact = z_mm, impact
        self.stats["accepted"] += 1

        res.z_clamped_from = z_clamped
        res.target = ShotTarget(
            impact_time=impact, x_mm=x_mm, z_mm=z_mm,
            track_no=track, seq=self._seq,
            speed_mms=float(rep.get("v", 0.0)) * 1000.0,
            final=self._frozen is not None)
        res.accepted = True
        if frozen_now and self.verbose:
            print("[판정] 랠리 #%d 목적지 확정 — 공이 Y %.2f m 를 지남  "
                  "X %+.0f  Z %.0f mm  (임팩트까지 %.0f ms)"
                  % (track, self._frozen_from, x_mm, z_mm, lead * 1000))
        return res

    # ------------------------------------------------------------------
    def _log_once(self, track: int, reason: str) -> None:
        """한 트랙당 거부 사유를 한 번만 찍는다 (콘솔 폭주 방지)."""
        if not self.verbose or track == self._reported_track:
            return
        self._reported_track = track
        if reason.startswith("!"):          # 흘려보내는 게 아니라 시도하는 경우
            print("[판정] 트랙 #%d %s" % (track, reason[1:]))
        else:
            print("[판정] 트랙 #%d 흘려보냄 — %s" % (track, reason))

    def summary(self) -> str:
        s = self.stats
        return ("예측 %d  채택 %d\n폭 밖 %d  높이 밖 %d  시간 없음 %d"
                % (s["seen"], s["accepted"], s["out_x"], s["out_z"],
                   s["no_time"]))


# ============================================================================
# 자체 시험
# ============================================================================

def _self_test() -> None:
    print("=" * 70)
    print(" 타격 판정 자체 시험")
    print("=" * 70)

    class _FakePredictor:
        def __init__(self):
            self.last_report = None

        def set(self, x_m, z_m, eta, track, vy=-4.0, vx=-0.1, vz=0.3):
            now = time.perf_counter()
            self.last_report = {
                "axis": "along", "value": 0.0, "seg": "B", "eta": eta,
                "pred": False, "t_sample": now, "t_impact": now + eta,
                "track_no": track, "X": x_m, "Y": 0.0, "Z": z_m,
                "vX": vx, "vY": vy, "vZ": vz,
                "v": math.sqrt(vx * vx + vy * vy + vz * vz),
                "v_h": math.hypot(vx, vy),
                "last_X": x_m, "last_Y": 0.3, "last_Z": z_m,
                "extrap_m": 0.3, "extrap_t": 0.07}

    p = ShotPlanner(verbose=True)
    print("\n" + p.describe_window() + "\n")

    fake = _FakePredictor()
    cases = [
        ("가운데 · 좋은 높이", 0.00, 0.26, 0.60, 1, True),
        ("우측 창 경계 안",   0.34, 0.24, 0.60, 2, True),
        ("우측 창 밖",        0.50, 0.24, 0.60, 3, False),
        ("좌측 창 밖",       -0.48, 0.26, 0.60, 4, False),
        ("너무 높음 (한계로)", 0.00, 0.42, 0.60, 5, True),
        ("너무 낮음 (한계로)", 0.00, 0.09, 0.60, 6, True),
        ("이미 지남",         0.00, 0.26, 0.01, 7, False),
    ]
    ok = True
    for name, x, z, eta, track, expect in cases:
        fake.set(x, z, eta, track)
        r = p.plan(fake)
        mark = "채택" if r.accepted else "거부"
        if r.z_clamped_from is not None:
            mark += "(높이 한계 %.0f->%.0f)" % (r.z_clamped_from, r.z_mm)
        if r.accepted != expect:
            mark += "  [!] 기대와 다름"
            ok = False
        print("  %-16s X %+7.1f  Z %6.1f mm  ->  %s  %s"
              % (name, r.x_mm, r.z_mm, mark, r.reason))

    print("\n[잔변화 억제]  (결정선이 켜져 있으면 위치가 고정되므로 "
          "타이밍 갱신을 위해 매번 채택되는 것이 정상)")
    fake.set(0.0, 0.26, 0.60, 10)
    r1 = p.plan(fake)
    fake.set(0.002, 0.261, 0.599, 10)
    r2 = p.plan(fake)
    print("  1차 %s / 2차(2mm 이동) %s"
          % ("채택" if r1.accepted else "거부",
             "채택" if r2.accepted else "거부 — " + r2.reason))
    if cfg.DECIDE_LINE_ENABLE:
        ok = ok and r1.accepted and r2.accepted and r2.target.final
        print("  확정 표시: %s, 위치 고정 %s"
              % (r2.target.final,
                 "예" if abs(r2.x_mm - r1.x_mm) < 1e-6 else "아니오"))
    else:
        ok = ok and r1.accepted and not r2.accepted

    if cfg.DECIDE_LINE_ENABLE:
        print("\n[결정선]")
        p2 = ShotPlanner(verbose=False)
        fake.set(0.0, 0.26, 0.60, 20)
        fake.last_report["last_Y"] = 1.30          # 아직 선 뒤
        r = p2.plan(fake)
        print("  선 뒤(Y 1.30 m): %s — %s"
              % ("채택" if r.accepted else "대기", r.reason))
        ok = ok and not r.accepted
        fake.last_report["last_Y"] = 0.80          # 선을 넘음
        r = p2.plan(fake)
        print("  선 앞(Y 0.80 m): %s  확정 %s"
              % ("채택" if r.accepted else "대기",
                 r.target.final if r.target else "-"))
        ok = ok and r.accepted and r.target.final

    print("\n통계: %s" % p.summary())
    print("\n%s" % ("모든 시험 통과" if ok else "실패한 항목이 있습니다"))


if __name__ == "__main__":
    _self_test()
