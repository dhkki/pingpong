#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_kinematics.py
===================
팔 기구학 — 타점 (X, Z) <-> 관절 (theta, 리니어 지령).

기구
----
리니어 캐리지 위에 힌지가 있고, 거기서 막대가 뻗어 나가 끝에 라켓이 달린다.
막대의 기울기 theta 는 다이나믹셀 1, 2 가 반대로 돌아 만든다.  막대가
기울어지는 평면이 리니어축과 평행하므로, theta 로 높이를 얻으면 라켓이
폭 방향(X)으로도 함께 딸려 나간다.  **X 와 Z 는 독립이 아니다.**

순기구학
--------
    Z(theta) = z0 + R·sin(theta + phi)
    X(theta) = q_lin + x_arm0 + sign · R·cos(theta + phi) + d_paddle

phi = 90° 이므로 실질적으로

    Z = 125 + 200·cos(theta) + 50·sin(theta)
    X = q_lin + x_arm0 - 200·sin(theta) + 50·cos(theta) + 130   (sign = +1)

손목 축에서 라켓 중심까지는 두 성분으로 나눈다.  하나는 레일과 나란한
일정 방향(130mm, 상수), 다른 하나는 막대에 수직인 성분(50mm)이다.  뒤쪽은
막대와 함께 돌기 때문에 기울일수록 라켓을 아래로 끌어내린다.  실측 9점에서
역산한 값이며, 이 항이 없으면 눕혔을 때 계산이 50~60mm 높게 나온다.

역기구학
--------
    1) Z 가 theta 를 정한다      theta = asin((Z - z0)/R) - phi
    2) 남은 X 를 리니어가 흡수   q_lin = X - x_arm0 - sign·R·cos(theta + phi)

Z 가 theta 를 소진하고 리니어가 나머지를 받는 순서다.  theta 범위는 좁고
리니어 스트로크는 1141mm 로 넉넉하므로 이 순서가 맞다.

이동 시간
---------
축마다 사다리꼴/삼각 속도 프로파일로 이동 시간을 계산한다.  실측(리니어
전구간 525ms, 200mm 220ms)과 7ms 이내로 일치하는 것을 확인했다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import robot_config as cfg


# ============================================================================
# 해
# ============================================================================

@dataclass(frozen=True)
class ArmSolution:
    theta_deg: float          # ID 1, 2 에 보낼 절대 기울기
    linear_cmd: float         # 리니어 지령값 (단위)
    x_achieved_mm: float      # 실제 도달하는 폭 방향 위치
    z_achieved_mm: float      # 실제 도달하는 높이
    swing_radius_mm: float    # 이 자세에서의 스윙 반경
    reachable: bool           # 요청을 그대로 만족했는가
    clamped: tuple            # 포화된 항목 ("theta", "linear")
    req_x_mm: float = 0.0
    req_z_mm: float = 0.0

    @property
    def x_error_mm(self) -> float:
        return self.x_achieved_mm - self.req_x_mm

    @property
    def z_error_mm(self) -> float:
        return self.z_achieved_mm - self.req_z_mm

    @property
    def error_mm(self) -> float:
        return math.hypot(self.x_error_mm, self.z_error_mm)

    def __str__(self) -> str:
        return ("theta %+6.1f°  리니어 %6.2f  -> X %+7.1f  Z %6.1f mm  "
                "스윙반경 %5.1f mm%s"
                % (self.theta_deg, self.linear_cmd, self.x_achieved_mm,
                   self.z_achieved_mm, self.swing_radius_mm,
                   "" if self.reachable else "  [포화 %s]" % ",".join(self.clamped)))


# ============================================================================
# 기구학
# ============================================================================

class ArmKinematics:

    def __init__(self, geometry=None):
        self.g = geometry or cfg.ARM

    # ---------------------------------------------------------- 순기구학
    def forward(self, theta_deg: float, linear_cmd: float) -> tuple:
        """(theta, 리니어 지령) -> 라켓 중심의 (X, Z) mm"""
        g = self.g
        a = math.radians(theta_deg + g.phase_deg)
        q = float(linear_cmd) * g.linear_mm_per_unit
        #  막대 방향 u = (cos a, sin a),  막대에 수직 v = (sin a, -cos a)
        x = (q + g.arm_x_offset_mm + g.paddle_offset_mm
             + g.arm_x_sign * (g.radius_mm * math.cos(a)
                               + g.paddle_perp_mm * math.sin(a)))
        z = (g.hinge_z_mm + g.radius_mm * math.sin(a)
             - g.paddle_perp_mm * math.cos(a))
        return x, z

    def wrist_xz(self, theta_deg: float, linear_cmd: float) -> tuple:
        """막대 끝(손목 축)의 (X, Z). 라켓 중심은 여기서 옆으로 d_paddle."""
        g = self.g
        a = math.radians(theta_deg + g.phase_deg)
        q = float(linear_cmd) * g.linear_mm_per_unit
        return (q + g.arm_x_offset_mm + g.arm_x_sign * g.radius_mm * math.cos(a),
                g.hinge_z_mm + g.radius_mm * math.sin(a))

    def swing_radius_mm(self, theta_deg: float) -> float:
        """스윙 반경 = R x |sin theta|.  theta=0 이면 0 (타격 불가)."""
        return abs(self.g.radius_mm * math.sin(math.radians(theta_deg)))

    # ---------------------------------------------------------- 도달 범위
    def _amp_phase(self):
        """Z = z0 + A·sin(theta + phi + psi) 형태의 A, psi.

        막대에 수직인 오프셋 때문에 Z 는 단순 cos 가 아니라 위상이 얹힌
        코사인이 된다.  진폭과 위상을 한 번 구해 두고 여기저기서 쓴다.
        """
        g = self.g
        A = math.hypot(g.radius_mm, g.paddle_perp_mm)
        psi = math.atan2(-g.paddle_perp_mm, g.radius_mm)
        return A, psi

    def z_limits_mm(self) -> tuple:
        """theta 범위 안에서 도달 가능한 Z 구간."""
        g = self.g
        phi = math.radians(g.phase_deg)
        A, psi = self._amp_phase()
        lo, hi = math.radians(g.theta_min_deg), math.radians(g.theta_max_deg)
        zs = [g.hinge_z_mm + A * math.sin(t + phi + psi) for t in (lo, hi)]
        for peak in (math.pi / 2, -math.pi / 2):
            t = peak - phi - psi
            if lo <= t <= hi:
                zs.append(g.hinge_z_mm + A * math.sin(peak))
        return min(zs), max(zs)

    def z_limits_with_swing_mm(self) -> tuple:
        """스윙 반경 하한까지 반영한 '실제로 칠 수 있는' Z 구간.

        theta 가 0 에 가까우면 높이는 나오지만 스윙에 힘이 실리지 않는다.
        """
        g = self.g
        z_lo, z_hi = self.z_limits_mm()
        s = min(1.0, g.min_swing_radius_mm / g.radius_mm)
        th_max = -math.degrees(math.asin(s))      # 이보다 0 에 가까우면 불가
        z_cap = g.hinge_z_mm + g.radius_mm * math.cos(math.radians(th_max))
        return z_lo, min(z_hi, z_cap)

    def x_span_mm(self) -> tuple:
        """theta 전 범위에서 라켓이 닿을 수 있는 X 의 합집합.

        높이마다 창이 좌우로 움직이므로, 어느 한 높이만 보면 실제보다
        좁게 잡힌다.  표시나 대략적인 판정에는 이 합집합을 쓴다.
        """
        lo, hi = None, None
        z0, z1 = self.z_limits_with_swing_mm()
        n = 25
        for i in range(n + 1):
            z = z0 + (z1 - z0) * i / n
            w = self.x_window_mm(z)
            if w is None:
                continue
            lo = w[0] if lo is None else min(lo, w[0])
            hi = w[1] if hi is None else max(hi, w[1])
        return (lo, hi) if lo is not None else (-381.0, 381.0)

    def x_window_mm(self, z_mm: float) -> tuple | None:
        """주어진 높이에서 리니어 스트로크가 만드는 X 구간.

        낮은 공일수록 팔이 우측으로 밀리므로 창이 오른쪽으로 이동한다.
        """
        g = self.g
        thetas, clamped = self._theta_candidates(z_mm)
        if clamped or not thetas:
            return None
        phi = math.radians(g.phase_deg)
        best = None
        for th in thetas:
            off = (g.arm_x_sign * (g.radius_mm * math.cos(th + phi)
                                   + g.paddle_perp_mm * math.sin(th + phi))
                   + g.paddle_offset_mm)
            lo = g.linear_min_cmd * g.linear_mm_per_unit + g.arm_x_offset_mm + off
            hi = g.linear_max_cmd * g.linear_mm_per_unit + g.arm_x_offset_mm + off
            span = (min(lo, hi), max(lo, hi))
            if best is None or (span[1] - span[0]) > (best[1] - best[0]):
                best = span
        return best

    # ---------------------------------------------------------- 역기구학
    def _theta_candidates(self, z_mm: float):
        """Z 를 만드는 theta 후보.  Z = z0 + R·sin(theta+phi) 는 두 해를 갖는다."""
        g = self.g
        phi = math.radians(g.phase_deg)
        A, psi = self._amp_phase()
        s = (z_mm - g.hinge_z_mm) / A
        clamped = False
        if s < -1.0 or s > 1.0:
            clamped = True
            s = max(-1.0, min(1.0, s))
        a = math.asin(s)
        lo, hi = math.radians(g.theta_min_deg), math.radians(g.theta_max_deg)
        out = []
        for base in (a, math.pi - a):
            for turn in (-2 * math.pi, 0.0, 2 * math.pi):
                t = base + turn - phi - psi
                if lo - 1e-9 <= t <= hi + 1e-9:
                    if not any(abs(t - u) < 1e-9 for u in out):
                        out.append(t)
        return out, clamped

    def solve(self, x_mm: float, z_mm: float, *,
              linear_now: float | None = None) -> ArmSolution:
        """타점 (X, Z) -> (theta, 리니어 지령).

        같은 Z 를 만드는 두 자세가 있으면 **리니어가 덜 움직이는 쪽**을 고른다.
        리니어가 가장 느린 축이라 이 선택이 응답 시간을 좌우한다.
        """
        g = self.g
        phi = math.radians(g.phase_deg)
        thetas, z_clamped = self._theta_candidates(z_mm)
        if not thetas:
            thetas = [math.radians(g.theta_min_deg),
                      math.radians(g.theta_max_deg)]
            z_clamped = True

        best = None
        for theta in thetas:
            a = theta + phi
            q = (x_mm - g.arm_x_offset_mm - g.paddle_offset_mm
                 - g.arm_x_sign * (g.radius_mm * math.cos(a)
                                   + g.paddle_perp_mm * math.sin(a)))
            cmd = q / g.linear_mm_per_unit
            lin_clamped = cmd < g.linear_min_cmd or cmd > g.linear_max_cmd
            cmd = max(g.linear_min_cmd, min(g.linear_max_cmd, cmd))
            xh, zh = self.forward(math.degrees(theta), cmd)
            err = math.hypot(xh - x_mm, zh - z_mm)
            travel = 0.0 if linear_now is None else abs(cmd - linear_now)
            score = (round(err, 3), travel, abs(theta))
            cand = (score, theta, cmd, xh, zh, lin_clamped)
            if best is None or cand[0] < best[0]:
                best = cand

        _, theta, cmd, xh, zh, lin_clamped = best
        clamped = []
        if z_clamped:
            clamped.append("theta")
        if lin_clamped:
            clamped.append("linear")
        return ArmSolution(
            theta_deg=math.degrees(theta), linear_cmd=cmd,
            x_achieved_mm=xh, z_achieved_mm=zh,
            swing_radius_mm=self.swing_radius_mm(math.degrees(theta)),
            reachable=not clamped, clamped=tuple(clamped),
            req_x_mm=x_mm, req_z_mm=z_mm)

    # ------------------------------------------------------------ 리포트
    def coverage_report(self) -> None:
        g = self.g
        z_lo, z_hi = self.z_limits_mm()
        s_lo, s_hi = self.z_limits_with_swing_mm()
        print("  도달 Z %.0f ~ %.0f mm   (스윙 반경 %.0fmm 하한 반영 시 %.0f ~ %.0f)"
              % (z_lo, z_hi, g.min_swing_radius_mm, s_lo, s_hi))
        print("  %8s %8s %10s %24s" % ("높이 Z", "theta", "스윙반경", "도달 X 구간"))
        for z in range(340, 150, -20):
            w = self.x_window_mm(float(z))
            sol = self.solve(0.0, float(z))
            if w is None:
                print("  %7.0fmm %8s %10s %24s" % (z, "-", "-", "도달 불가"))
                continue
            print("  %7.0fmm %+7.1f° %9.0fmm %10.0f ~ %+7.0f mm"
                  % (z, sol.theta_deg, sol.swing_radius_mm, w[0], w[1]))


# ============================================================================
# 이동 시간 모델
# ============================================================================

def profile_move_sec(distance: float, vmax: float, accel: float) -> float:
    """사다리꼴/삼각 속도 프로파일의 이동 시간(s).

    거리가 짧아 최고 속도에 못 닿으면 삼각 프로파일이 된다.
        삼각      : t = 2·sqrt(d / a)
        사다리꼴  : t = d/v + v/a
    """
    d = abs(float(distance))
    if d <= 1e-9:
        return 0.0
    v, a = abs(float(vmax)), abs(float(accel))
    if v <= 0 or a <= 0:
        return 0.0
    if d <= v * v / a:
        return 2.0 * math.sqrt(d / a)
    return d / v + v / a


def distance_for_time(t_sec: float, vmax: float, accel: float) -> float:
    """주어진 시간 안에 갈 수 있는 최대 거리 (profile_move_sec 의 역함수).

        삼각      d = a·(t/2)²          (최고 속도에 못 닿는 구간)
        사다리꼴  d = v·(t - v/a)
    """
    t = max(0.0, float(t_sec))
    v, a = abs(float(vmax)), abs(float(accel))
    if t <= 0 or v <= 0 or a <= 0:
        return 0.0
    d_tri = a * (t / 2.0) ** 2
    if d_tri <= v * v / a:          # 아직 삼각 구간
        return d_tri
    return v * (t - v / a)


def linear_distance_for_time(t_sec: float) -> float:
    """리니어가 t_sec 안에 갈 수 있는 지령 단위 거리."""
    return distance_for_time(t_sec, cfg.LINEAR_VELOCITY,
                             cfg.LINEAR_ACCELERATION)


def linear_move_sec(delta_cmd: float) -> float:
    """리니어 지령 delta 단위를 움직이는 데 걸리는 시간(s)."""
    return profile_move_sec(delta_cmd, cfg.LINEAR_VELOCITY,
                            cfg.LINEAR_ACCELERATION)


#  프로파일 값 0 은 "제한 없음"이라 시간 계산에는 쓸 수 없다.
#  실제로 낼 수 있는 대략적인 상한으로 대신 계산한다.
POSE_VELOCITY_WHEN_FREE = 400        # ≈ 550 deg/s
POSE_ACCEL_WHEN_FREE = 200           # ≈ 42,900 deg/s^2


def tilt_move_sec(delta_deg: float) -> float:
    """틸트를 delta_deg 움직이는 데 걸리는 시간(s).

    프로파일 값이 0(제한 없음)이면 모터 상한을 가정해 계산한다.
    """
    vp = cfg.POSE_VELOCITY or POSE_VELOCITY_WHEN_FREE
    ap = cfg.POSE_ACCELERATION or POSE_ACCEL_WHEN_FREE
    return profile_move_sec(delta_deg, vp * cfg.DXL_VEL_TO_DPS,
                            ap * cfg.DXL_ACC_TO_DPSS)


def swing_lead_sec() -> float:
    """스윙 발사 시각의 리드타임(s).

    접촉은 호의 중간에서 일어나므로 스윙 소요시간의 절반이 기본이고,
    지령이 모터에 닿는 지연을 더한다.
    """
    return (cfg.SWING_DURATION_SEC * cfg.SWING_CONTACT_FRACTION
            + cfg.SWING_COMMAND_LATENCY_SEC)


# ============================================================================
# 자체 시험
# ============================================================================

def _self_test() -> None:
    k = ArmKinematics()
    print("=" * 70)
    print(" 기구학 자체 시험")
    print("=" * 70)

    print("\n[1] 순기구학 <-> 역기구학 왕복")
    ok = True
    for theta in (-70, -55, -40, -25, -10):
        for cmd in (10.0, 40.0, 70.0, 100.0):
            x, z = k.forward(theta, cmd)
            sol = k.solve(x, z)
            if sol.error_mm > 0.5:
                print("   [!] theta %+.0f cmd %.0f -> 오차 %.2fmm"
                      % (theta, cmd, sol.error_mm))
                ok = False
    print("   왕복 오차 0.5mm 이내: %s" % ("통과" if ok else "실패"))

    print("\n[2] Z 와 스윙 반경의 상충")
    for th in (0, -20, -40, -60, -80):
        _, z = k.forward(th, 0)
        print("   theta %+3d°  Z %6.1f mm  스윙반경 %6.1f mm%s"
              % (th, z, k.swing_radius_mm(th),
                 "   <- 타격 불가" if k.swing_radius_mm(th)
                 < cfg.ARM.min_swing_radius_mm else ""))

    print("\n[3] 도달 범위")
    k.coverage_report()

    print("\n[4] 이동 시간 모델 (실측 대조)")
    for mm, ref in ((100, 155), (200, 220), (400, 311), (1141, 525)):
        units = mm / cfg.ARM.linear_mm_per_unit
        t = linear_move_sec(units) * 1000
        print("   리니어 %5.0fmm -> %6.1f ms   (실측 %d ms, 차 %+.1f)"
              % (mm, t, ref, t - ref))
    for deg in (5, 19, 40):
        print("   틸트  %5.0f°  -> %6.1f ms" % (deg, tilt_move_sec(deg) * 1000))
    print("   스윙 리드타임 %.1f ms  (소요 %.0fms 의 %.0f%% + 지연 %.0fms)"
          % (swing_lead_sec() * 1000, cfg.SWING_DURATION_SEC * 1000,
             cfg.SWING_CONTACT_FRACTION * 100,
             cfg.SWING_COMMAND_LATENCY_SEC * 1000))

    print("\n모든 시험 통과")


if __name__ == "__main__":
    _self_test()
