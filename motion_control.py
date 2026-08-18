#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motion_control.py
=================
모션 계층 — 타점 예측을 실제 축 지령으로 바꾸는 실시간 스레드.

왜 별도 스레드인가
------------------
스윙은 도달을 기다리며 자고, 리니어 DLL 은 이동이 끝날 때까지 반환하지
않는다.  비전 루프에서 직접 부르면 그 시간 동안 카메라 프레임이 통째로
날아가고, 프레임이 날아가면 다음 예측이 나빠져 악순환이 된다.

큐가 아니라 단일 슬롯
---------------------
0.2초 전 예측은 쓸모가 없다.  "최신 목표가 이전 목표를 덮어쓴다"가 맞는
동작이므로 Queue 가 아니라 락으로 보호되는 슬롯 하나 + Condition 을 쓴다.

시간 축 (전부 perf_counter 절대 시각)
------------------------------------
    impact - move_sec - margin     자세 지령 (리니어 + 틸트 동시)
    impact - swing_lead            스윙 발사
    impact                         라켓이 공을 만나는 순간

자세를 리니어와 틸트로 나눠 보내지 않는 이유: 라켓의 폭 방향 위치가 두 축
모두에 의존하므로(X = 리니어 - 220·sin theta), 따로 보내면 그 사이 라켓이
최대 190mm 옆으로 나갔다 돌아온다.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
import time
from dataclasses import dataclass, field, replace

import robot_config as cfg
from robot_kinematics import (ArmKinematics, linear_distance_for_time,
                              linear_move_sec, swing_lead_sec, tilt_move_sec)


# ============================================================================
# 정밀 타이밍
# ============================================================================

_timer_set = False


def enable_high_resolution_timer() -> bool:
    """Windows 타이머 분해능을 1ms 로 낮춘다.

    기본 15.6ms 로는 sleep(0.08) 이 95ms 를 자버려 스윙이 통째로 늦는다.
    """
    global _timer_set
    if os.name != "nt" or _timer_set:
        return _timer_set
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _timer_set = True
    except Exception:
        pass
    return _timer_set


def release_high_resolution_timer() -> None:
    global _timer_set
    if os.name == "nt" and _timer_set:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        finally:
            _timer_set = False


def sleep_until(deadline: float, spin_sec: float = 0.002) -> float:
    """deadline 까지 정확히 기다린다. 마지막 spin_sec 는 바쁜 대기."""
    while True:
        remain = deadline - time.perf_counter()
        if remain <= 0.0:
            return remain
        if remain > spin_sec:
            time.sleep(remain - spin_sec)


# ============================================================================
# 자료형
# ============================================================================

@dataclass(frozen=True)
class ShotTarget:
    """한 번의 타점 예측."""

    impact_time: float        # perf_counter 절대 시각
    x_mm: float               # 폭 방향 타점
    z_mm: float               # 상판 기준 높이
    track_no: int = 0         # 공(트랙) 번호. 바뀌면 새 랠리
    seq: int = 0              # 같은 공에 대한 갱신 번호
    final: bool = False       # 목적지 확정됨 (조각내지 않고 한 번에 간다)
    speed_mms: float = 0.0    # 참고용 도착 속도

    def lead_sec(self, now: float | None = None) -> float:
        return self.impact_time - (time.perf_counter() if now is None else now)


@dataclass
class MotionStats:
    submitted: int = 0
    posed: int = 0
    swung: int = 0
    too_late: int = 0
    unreachable: int = 0
    errors: int = 0


# ============================================================================
# 하드웨어 백엔드
# ============================================================================

class RobotBackend:
    """모션 워커가 기대하는 최소 인터페이스.

    실기 드라이버(다이나믹셀 + Ajin 리니어)는 이 인터페이스만 구현하면
    워커를 고치지 않고 갈아끼울 수 있다.

    축별 사용 여부
    --------------
    세 가지를 각각 켜고 끌 수 있다.  단계적으로 시험하기 위한 것이다.

        enable_linear  리니어축(폭 이동)
        enable_tilt    틸트(높이) + 라켓 면 보정
        enable_swing   스윙(바닥 원판)

    꺼진 축은 지령이 나가지 않고, 이동 시간 계산에서도 빠진다.
    """

    enable_linear = True
    enable_tilt = True
    enable_swing = True

    def active_axes(self) -> dict:
        return dict(linear=bool(self.enable_linear),
                    tilt=bool(self.enable_tilt),
                    swing=bool(self.enable_swing))

    def axes_text(self) -> str:
        a = self.active_axes()
        return "리니어 %s · 팔 %s · 스윙 %s" % (
            "O" if a["linear"] else "X",
            "O" if a["tilt"] else "X",
            "O" if a["swing"] else "X")

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def set_pose(self, theta_deg: float, linear_cmd: float) -> None:
        """틸트와 리니어를 **같은 순간에** 지령한다."""
        raise NotImplementedError

    def swing(self) -> float:
        """대기각 -> 타격각 -> 대기각. 걸린 시간(s)을 돌려준다."""
        raise NotImplementedError

    def go_ready(self, theta_deg: float, linear_cmd: float, *,
                 slow: bool = False) -> None:
        """대기 자세로 이동.  slow=True 면 무장 직후용 저속."""
        self.set_pose(theta_deg, linear_cmd)

    def park_linear(self, cmd: float = 0.0, timeout: float = 25.0) -> None:
        """리니어를 지정 지령으로 천천히 되돌린다 (종료 시 원위치)."""
        return None

    def park_arm(self) -> None:
        """토크를 풀기 전에 팔을 아래로 천천히 내려 둔다."""
        return None

    def warmup_pose(self, theta_deg: float, linear_cmd: float) -> None:
        """준비운동용 자세 지령 (느린 프로파일)."""
        self.set_pose(theta_deg, linear_cmd)

    def measure_swing(self) -> float:
        """대기각 -> 타격각 실측 시간(s). 지원하지 않으면 0."""
        return 0.0

    def current_pose(self) -> tuple:
        """지금 이 순간의 **실제** (theta_deg, 리니어 지령).

        기본 구현은 마지막 지령값을 돌려준다.  실기/시뮬 백엔드는 이동 중인
        위치를 추정하거나 실측해서 덮어쓴다.
        """
        return (getattr(self, "theta_deg", 0.0),
                getattr(self, "linear_cmd", 0.0))

    def emergency_stop(self) -> None:
        raise NotImplementedError


class SimBackend(RobotBackend):
    """시뮬레이터. 하드웨어 없이 타이밍과 지령만 확인한다.

    실제 축 성능을 흉내내기 위해 이동 시간만큼 실제로 잔다.  그래야
    "지령이 늦어 못 친다" 같은 문제가 시뮬레이션에서도 드러난다.
    """

    def __init__(self, *, verbose: bool = True, realtime: bool = True):
        self.verbose = verbose
        self.realtime = realtime
        self.theta_deg = cfg.READY_THETA_DEG
        self.linear_cmd = 0.0
        self.opened = False
        self.log = []
        self._pose_seq = 0
        #  이동 중 실제 위치 추정용
        self._seg = None      # (t0, dur, th_from, th_to, lin_from, lin_to)

    def open(self) -> None:
        self.opened = True
        if self.verbose:
            print("[SIM] 백엔드 열림 (하드웨어 없음)")

    def close(self) -> None:
        if cfg.TORQUE_OFF_ON_EXIT:
            self.park_arm()
        self.opened = False
        if self.verbose:
            print("[SIM] 백엔드 닫힘")

    def set_pose(self, theta_deg: float, linear_cmd: float) -> None:
        """실기와 같은 성질을 갖도록 **즉시 반환**한다.

        실기에서 리니어는 별도 스레드로 돌고 다이나믹셀 쓰기는 수 ms 라
        set_pose 는 곧바로 반환한다.  시뮬레이터가 여기서 자버리면 그 사이
        스윙 발사 시각을 놓쳐 실기와 다른 결과가 나온다.  그래서 이동은
        배경 스레드에서 흉내낸다.
        """
        dt = 0.0
        if self.enable_tilt:
            dt = max(dt, tilt_move_sec(theta_deg - self.theta_deg))
        if self.enable_linear:
            dt = max(dt, linear_move_sec(linear_cmd - self.linear_cmd))
        self.log.append(("pose", time.perf_counter(), theta_deg, linear_cmd))

        self._pose_seq += 1
        seq = self._pose_seq
        self._seg = (time.perf_counter(), dt, self.theta_deg,
                     float(theta_deg) if self.enable_tilt else self.theta_deg,
                     self.linear_cmd,
                     float(linear_cmd) if self.enable_linear
                     else self.linear_cmd)

        def apply():
            if self.realtime and dt > 0:
                time.sleep(dt)
            if seq != self._pose_seq:      # 더 최신 지령이 들어왔다
                return
            if self.enable_tilt:
                self.theta_deg = float(theta_deg)
            if self.enable_linear:
                self.linear_cmd = float(linear_cmd)

        if self.realtime and dt > 0:
            threading.Thread(target=apply, daemon=True,
                             name="sim-pose").start()
        else:
            apply()

    def measure_swing(self) -> float:
        if not self.enable_swing:
            return 0.0
        if self.realtime:
            time.sleep(cfg.SWING_DURATION_SEC)
        return cfg.SWING_DURATION_SEC

    def warmup_pose(self, theta_deg: float, linear_cmd: float) -> None:
        self.set_pose(theta_deg, linear_cmd)

    def park_arm(self) -> None:
        if not self.enable_tilt:
            return
        target = max(cfg.ARM.theta_min_deg,
                     min(cfg.ARM.theta_max_deg,
                         cfg.READY_THETA_DEG - abs(cfg.PARK_EXTRA_TILT_DEG)))
        if self.realtime:
            time.sleep(0.3)
        self.theta_deg = target
        self._seg = None
        if self.verbose:
            print("[SIM] 안착 theta %+.1f° 후 토크 해제" % target)

    def current_pose(self) -> tuple:
        try:
            from linear_link import profile_progress
        except ImportError:
            def profile_progress(u):
                return max(0.0, min(1.0, float(u)))
        seg = self._seg
        if seg is None or seg[1] <= 1e-6:
            return (self.theta_deg, self.linear_cmd)
        t0, dur, th0, th1, li0, li1 = seg
        u = (time.perf_counter() - t0) / dur
        if u >= 1.0:
            return (th1, li1)
        f = profile_progress(u)
        return (th0 + (th1 - th0) * f, li0 + (li1 - li0) * f)

    def swing(self) -> float:
        """실기처럼 지령만 보내고 즉시 반환한다 (복귀는 배경에서)."""
        t0 = time.perf_counter()
        if not self.enable_swing:
            return 0.0
        self.log.append(("swing", t0, 0.0, 0.0))
        if self.realtime:
            threading.Thread(
                target=lambda: time.sleep(cfg.SWING_DURATION_SEC),
                daemon=True, name="sim-swing").start()
        return time.perf_counter() - t0

    def park_linear(self, cmd: float = 0.0, timeout: float = 25.0) -> None:
        #  시작·종료 위치는 랠리용 범위 밖일 수 있으므로 클램프하지 않는다
        dt = linear_move_sec(cmd - self.linear_cmd)
        if self.realtime and dt > 0:
            time.sleep(min(dt, timeout))
        self.linear_cmd = float(cmd)
        self._seg = None                 # 위치 추정 상태도 초기화
        if self.verbose:
            print("[SIM] 리니어 원위치 %.2f" % self.linear_cmd)

    def emergency_stop(self) -> None:
        if self.verbose:
            print("[SIM] 비상정지")


# ============================================================================
# 모션 워커
# ============================================================================

class MotionWorker:
    """예측 -> 역기구학 -> 축 지령 -> 스윙 을 자기 스레드에서 처리한다."""

    def __init__(self, backend: RobotBackend, *, kinematics=None,
                 armed: bool = False, verbose: bool = True,
                 on_event=None):
        self.backend = backend
        self.kin = kinematics or ArmKinematics()
        self.armed = bool(armed)
        self.verbose = bool(verbose)
        self.on_event = on_event          # 콜백(str, dict) — UI 표시용

        self._slot: ShotTarget | None = None
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

        self._done_track = -1             # 이미 처리한 트랙 번호
        self._warned_track = -1           # 사유를 이미 알린 트랙
        self._pose_warned = None
        self._want_ready = False          # 대기 자세 정렬 요청
        self._ready_slow = False          # 그 정렬을 느리게 할지
        self._want_warmup = False         # 준비운동 요청
        self._cooldown_until = 0.0
        self._last_shot_time = 0.0
        self._at_ready = False

        self.stats = MotionStats()
        self.last_solution = None
        self.state = "대기"

    # ------------------------------------------------------------ 수명주기
    def start(self) -> None:
        enable_high_resolution_timer()
        #  이미 열려 있으면 다시 열지 않는다 (앱이 미리 열어 넘기는 경우)
        if not getattr(self.backend, "opened", False):
            self.backend.open()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="motion")
        self._thread.start()
        self._say("모션 스레드 시작 (%s)"
                  % ("무장" if self.armed else "안전 — 지령 안 나감"))

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        try:
            self.backend.close()
        finally:
            release_high_resolution_timer()

    def set_armed(self, armed: bool) -> None:
        self.armed = bool(armed)
        self._at_ready = False
        if self.armed:
            #  무장하는 순간 폭 중앙(X=0)으로 정렬한다.
            #  사람이 옆에 있는 순간이므로 이때만 저속으로 움직인다.
            if cfg.WARMUP_ON_ARM:
                #  준비운동이 끝나면서 대기 자세 정렬까지 함께 한다
                self._want_warmup = True
            else:
                self._want_ready = True
                self._ready_slow = True
        if self.armed:
            self._say("무장 — 준비운동 후 대기 자세로 정렬합니다"
                      if cfg.WARMUP_ON_ARM else "무장 — X=0 으로 정렬합니다")
        else:
            self._say("무장 해제")

    def request_warmup(self) -> None:
        """타격 가능 영역의 윤곽을 한 바퀴 그리도록 요청한다."""
        self._want_warmup = True
        with self._cond:
            self._cond.notify()

    def request_ready(self, *, slow: bool = True) -> None:
        """대기 자세(폭 중앙)로 정렬을 요청한다. 기본은 저속."""
        self._want_ready = True
        self._ready_slow = bool(slow)
        with self._cond:
            self._cond.notify()

    def set_backend(self, backend) -> None:
        """실행 중에 백엔드를 갈아끼운다 (시뮬레이터 <-> 실기).

        새 백엔드는 **이미 open() 된 것**이어야 한다.  안전을 위해 무장을
        해제하고 바꾼 뒤, 이전 백엔드를 닫는다.
        """
        old = self.backend
        self.armed = False
        self._want_ready = False
        self._at_ready = False
        self.backend = backend
        if old is not None and old is not backend:
            try:
                old.close()
            except Exception as e:
                self._say("이전 백엔드를 닫는 중 오류: %s" % e)
        self._say("백엔드 교체 — %s (무장 해제됨)" % backend.axes_text())

    def set_kinematics(self, kin) -> None:
        """보정값이 바뀌어 기구학을 갈아끼울 때."""
        self.kin = kin
        self._at_ready = False

    def emergency_stop(self) -> None:
        self.armed = False
        try:
            self.backend.emergency_stop()
        except Exception as e:
            self._say("비상정지 중 오류: %s" % e)
        self._say("비상정지 — 무장 해제됨")

    # ------------------------------------------------------------ 목표 투입
    def submit(self, target: ShotTarget) -> None:
        """최신 목표로 덮어쓴다. 논블로킹."""
        with self._cond:
            self._slot = target
            self.stats.submitted += 1
            self._cond.notify()

    def clear(self) -> None:
        with self._cond:
            self._slot = None

    # ---------------------------------------------------------------- 루프
    def _run(self) -> None:
        while self._running:
            with self._cond:
                if self._slot is None:
                    self._cond.wait(timeout=0.05)
                tgt = self._slot
            if not self._running:
                break
            if self._want_warmup:
                self._want_warmup = False
                self._warmup()
                continue
            if self._want_ready and time.perf_counter() >= self._cooldown_until:
                self._want_ready = False
                slow = self._ready_slow
                self._ready_slow = False
                self._go_ready(force=True, slow=slow)
                continue
            if tgt is None:
                self._idle()
                continue
            try:
                self._handle(tgt)
            except Exception as e:
                self.stats.errors += 1
                self._say("모션 오류: %s" % e)
                time.sleep(0.05)

    def _idle(self) -> None:
        """목표가 없을 때 — 필요하면 대기 자세로 복귀."""
        if not (cfg.RETURN_TO_READY and self.armed) or self._at_ready:
            return
        if time.perf_counter() - self._last_shot_time \
                < cfg.RETURN_TO_READY_AFTER_SEC:
            return
        self._go_ready()

    def warmup_path(self):
        """타격 가능 영역의 테두리를 도는 (theta, 리니어) 목록.

        리니어 하한/상한과 theta 하한/상한이 만드는 사각형의 네 변을 따라
        돈다.  실제 라켓은 그 경계를 훑으므로, 사람이 어디로 공을 보내야
        하는지 눈으로 확인할 수 있다.
        """
        g = cfg.ARM
        t0, t1 = g.theta_min_deg, g.theta_max_deg
        n = max(2, int(cfg.WARMUP_STEPS))

        def limits(theta):
            """그 각도에서 '라켓이 타격 창 안에 머무는' 지령 구간.

            준비운동은 실제로 칠 수 있는 영역만 훑어야 의미가 있다.
            ROI 밖으로 나가는 구간은 잘라낸다.
            """
            lo, hi = g.linear_min_cmd, g.linear_max_cmd
            z = self.kin.forward(theta, 0.0)[1]
            if cfg.HIT_X_MIN_MM is not None:
                s = self.kin.solve(float(cfg.HIT_X_MIN_MM), z)
                if s.reachable:
                    lo = max(lo, s.linear_cmd)
            if cfg.HIT_X_MAX_MM is not None:
                s = self.kin.solve(float(cfg.HIT_X_MAX_MM), z)
                if s.reachable:
                    hi = min(hi, s.linear_cmd)
            return (lo, hi) if hi > lo else (lo, lo)

        q0a, q1a = limits(t0)          # 눕힌 자세에서의 좌우 한계
        q0b, q1b = limits(t1)          # 세운 자세에서의 좌우 한계
        pts = []

        def lerp(a, b, i):
            return a + (b - a) * i / n

        for i in range(n + 1):                      # 아래 변: 왼쪽 -> 오른쪽
            pts.append((t0, lerp(q0a, q1a, i)))
        for i in range(1, n + 1):                   # 오른쪽 변: 눕힘 -> 세움
            th = lerp(t0, t1, i)
            pts.append((th, limits(th)[1]))         # 그 각도의 정확한 한계
        for i in range(1, n + 1):                   # 위 변: 오른쪽 -> 왼쪽
            pts.append((t1, lerp(q1b, q0b, i)))
        for i in range(1, n + 1):                   # 왼쪽 변: 세움 -> 눕힘
            th = lerp(t1, t0, i)
            pts.append((th, limits(th)[0]))
        return pts

    def _warmup(self) -> None:
        """준비운동. 타격 가능 영역의 윤곽을 한 바퀴 그린다."""
        if not self.armed:
            self._say("안전 모드 — 준비운동을 건너뜁니다")
            return
        axes = self.backend.active_axes()
        if not (axes["linear"] or axes["tilt"]):
            self._say("움직일 축이 없어 준비운동을 건너뜁니다")
            return
        path = self.warmup_path()
        self.state = "준비운동"
        self._say("준비운동 — 타격 가능 영역의 윤곽을 그립니다 (%d점)"
                  % len(path))
        try:
            for th, q in path:
                if not self._running or not self.armed:
                    break
                self.backend.warmup_pose(th, q)
                time.sleep(max(0.0, cfg.WARMUP_DWELL_SEC))
        except Exception as e:
            self._say("준비운동 중 오류: %s" % e)
        #  ---- 윤곽을 그린 뒤 대기 자세로 정렬 ----
        self._at_ready = False
        self._go_ready(force=True, slow=True)

        #  ---- 연습 스윙 ----
        n = int(cfg.WARMUP_SWINGS)
        if n > 0 and self.armed and self.backend.active_axes()["swing"]:
            self.state = "연습 스윙"
            self._say("연습 스윙 %d회" % n)
            for i in range(n):
                if not self._running or not self.armed:
                    break
                try:
                    self.backend.swing()
                except Exception as e:
                    self._say("연습 스윙 실패: %s" % e)
                    break
                time.sleep(max(0.0, cfg.WARMUP_SWING_GAP_SEC))

        #  방금 정렬했으므로 곧바로 다시 정렬하지 않는다
        self._at_ready = True
        self._last_shot_time = time.perf_counter()
        self.state = "대기"
        self._say("준비운동 완료 — 타격 대기")

    def _go_ready(self, *, force: bool = False,
                  slow: bool = False) -> None:
        """폭 중앙(X=0)의 대기 자세로 천천히 이동."""
        if not self.armed:
            self._say("안전 모드 — 정렬 지령을 보내지 않습니다")
            return
        if self._at_ready and not force:
            return
        sol = self.kin.solve(cfg.READY_X_MM, self._ready_z_mm())
        self.state = "대기 자세 정렬"
        self._say("X=0 정렬%s  %s" % ("(저속)" if slow else "", sol))
        self.backend.go_ready(sol.theta_deg, sol.linear_cmd, slow=slow)
        self._at_ready = True
        self.state = "대기"

    def _ready_z_mm(self) -> float:
        _, z = self.kin.forward(cfg.READY_THETA_DEG, 0.0)
        return z

    def _handle(self, tgt: ShotTarget) -> None:
        now = time.perf_counter()

        if tgt.track_no == self._done_track:
            time.sleep(0.005)
            return
        if now < self._cooldown_until:
            time.sleep(0.005)
            return

        lead = tgt.impact_time - now
        if lead > cfg.START_MOVE_LEAD_SEC:
            time.sleep(0.005)          # 아직 이르다. 예측을 더 받는다
            return
        if lead < cfg.MIN_LEAD_SEC:
            self._reject(tgt, "늦음 (%.0f ms 남음)" % (lead * 1000))
            self.stats.too_late += 1
            return

        sol = self.kin.solve(tgt.x_mm, tgt.z_mm,
                             linear_now=self._current_linear())
        if not self._usable(tgt, sol, lead):
            return

        self._engage(tgt)

    def _usable(self, tgt, sol, lead: float) -> bool:
        """이 자세로 칠 수 있는가.

        아직 시간이 많이 남았으면(초반 예측은 나쁘다) 포기하지 않고 조용히
        기다린다.  REJECT_LEAD_SEC 안으로 들어와서도 안 되면 그때 거부한다.
        """
        self.last_solution = sol
        why = None
        if not sol.reachable:
            usable = cfg.ARM.blade_radius_mm * cfg.ARM.blade_usable_frac
            if sol.error_mm > usable:
                why = ("도달 불가 (오차 %.0f mm, 포화 %s)"
                       % (sol.error_mm, ",".join(sol.clamped)))
        if why is None and sol.swing_radius_mm < cfg.ARM.min_swing_radius_mm:
            why = "스윙 반경 부족 (%.0f mm)" % sol.swing_radius_mm
        if why is None:
            return True
        if lead > cfg.REJECT_LEAD_SEC:
            time.sleep(0.005)          # 예측이 좋아지기를 기다린다
            return False
        #  ★ 여기서 트랙을 닫아 버리면 안 된다 ★
        #    예측은 임팩트 직전까지 계속 좋아진다.  한 번 못 친다고 판단해도
        #    다음 프레임에 될 수 있으므로, 사유만 한 번 알리고 계속 본다.
        self._reject(tgt, why, permanent=False)
        self.stats.unreachable += 1
        time.sleep(0.005)
        return False

    def _chunk_toward(self, cur: float, target: float,
                      final: bool = False) -> float:
        """리니어 지령을 한 조각만큼만 나아간 중간 목표로 자른다.

        조각의 이동 시간이 LINEAR_CHUNK_SEC 을 넘지 않게 한다.  남은 거리가
        한 조각 안에 들어가면 목표를 그대로 준다(마지막 조각).
        """
        d = float(target) - float(cur)
        if not cfg.LINEAR_CHUNK_MODE or final:
            #  목적지가 확정된 뒤에는 나눌 이유가 없다.
            #  조각마다 가감속이 다시 들어가 오히려 느려진다.
            return float(target)
        if linear_move_sec(d) <= cfg.LINEAR_CHUNK_SEC:
            return float(target)
        step = linear_distance_for_time(cfg.LINEAR_CHUNK_SEC)
        return float(cur) + math.copysign(step, d)

    def _engage(self, tgt0: ShotTarget) -> None:
        """스윙 시각까지 목표를 따라간다.

        목적지가 **확정된**(final) 목표면 역기구학을 딱 한 번만 풀고 그 자세로
        한 번에 보낸 뒤, 그 랠리 동안 다시 손대지 않는다.  매 바퀴 다시 풀면
        실제 위치가 조금씩 변하는 탓에 같은 X·Z 에서도 해가 흔들려 라켓이
        미세하게 떨린다.

        확정 전(결정선을 안 쓰는 경우)에는 예전처럼 조각으로 나눠 따라간다.
        어느 쪽이든 임팩트 시각은 계속 최신으로 읽어 스윙 타이밍만 갱신한다.
        """
        track = tgt0.track_no
        tgt = tgt0
        sol = self.last_solution
        sent = 0
        last_lin = None
        last_theta = None
        last_final = None
        locked = None           # 확정 목표를 한 번 푼 결과 (theta, 리니어)
        self.state = "따라가는 중"

        while self._running:
            with self._cond:
                cur = self._slot
            if cur is not None and cur.track_no == track:
                #  ★ 위치는 확정되었으므로 임팩트 시각만 받아 온다 ★
                tgt = cur if locked is None else replace(
                    tgt, impact_time=cur.impact_time)

            now = time.perf_counter()
            fire_at = tgt.impact_time - swing_lead_sec()
            if now >= fire_at:
                break

            axes = self.backend.active_axes()

            if tgt.final:
                #  ---- 확정 목표: 한 번만 풀고 한 번만 보낸다 ----
                if locked is None:
                    sol = self.kin.solve(tgt.x_mm, tgt.z_mm,
                                         linear_now=self._current_linear())
                    self.last_solution = sol
                    locked = (sol.theta_deg, sol.linear_cmd)
                    if self.armed:
                        self.backend.set_pose(locked[0], locked[1])
                    sent = 1
                    self.stats.posed += 1
                    self._at_ready = False
                    self._emit("commit", tgt, sol, lead=tgt.impact_time - now)
                    if self.verbose:
                        print("[모션] 목적지 확정 이동  트랙 #%d  "
                              "X %+.0f  Z %.0f mm  임팩트까지 %.0f ms   "
                              "%s   [%s]"
                              % (track, tgt.x_mm, tgt.z_mm,
                                 (tgt.impact_time - now) * 1000, sol,
                                 self.backend.axes_text()))
                time.sleep(0.004)
                continue

            #  ---- 확정 전: 조각으로 따라간다 ----
            sol = self.kin.solve(tgt.x_mm, tgt.z_mm,
                                 linear_now=self._current_linear())
            self.last_solution = sol
            lin_now = self._current_linear()
            lin_goal = (self._chunk_toward(lin_now, sol.linear_cmd, False)
                        if axes["linear"] else lin_now)
            th_goal = sol.theta_deg

            #  언제 다시 보낼까
            #    (1) 최종 목표가 의미 있게 바뀌었거나
            #    (2) 지금 조각이 거의 끝나 다음 조각이 필요하거나
            #    (3) 틸트 목표가 바뀌었을 때
            dead_cmd = cfg.MIN_RETARGET_MM / abs(cfg.ARM.linear_mm_per_unit)
            arrived = (last_lin is None
                       or abs(lin_now - last_lin) <= dead_cmd)
            goal_changed = (last_final is None
                            or abs(sol.linear_cmd - last_final) > dead_cmd)
            tilt_changed = (last_theta is None
                            or abs(th_goal - last_theta)
                            >= cfg.MIN_RETARGET_DEG)
            if arrived or goal_changed or tilt_changed:
                last_final = sol.linear_cmd
                if self.armed:
                    self.backend.set_pose(th_goal, lin_goal)
                last_lin, last_theta = lin_goal, th_goal
                sent += 1
                if sent == 1:
                    self.stats.posed += 1
                    self._at_ready = False
                    self._emit("commit", tgt, sol,
                               lead=tgt.impact_time - now)
                    if self.verbose:
                        print("[모션] 추적 시작  트랙 #%d  X %+.0f  Z %.0f mm  "
                              "임팩트까지 %.0f ms   [%s]"
                              % (track, tgt.x_mm, tgt.z_mm,
                                 (tgt.impact_time - now) * 1000,
                                 self.backend.axes_text()))
            time.sleep(0.004)

        #  ---- 스윙 ----
        self._done_track = track
        fire_at = tgt.impact_time - swing_lead_sec()
        late = time.perf_counter() - fire_at
        grace = cfg.SWING_LATE_GRACE_SEC if cfg.BEST_EFFORT else 0.020
        if late > grace:
            self._say("스윙 발사가 %.0f ms 늦어 건너뜀" % (late * 1000))
            self.stats.too_late += 1
            self.state = "대기"
            self._cooldown_until = time.perf_counter() + cfg.SWING_COOLDOWN_SEC
            if cfg.RETURN_TO_READY and cfg.RETURN_TO_READY_AFTER_SWING:
                self._want_ready = True
            return
        if late > 0.002:
            self._say("스윙 %.0f ms 늦게 발사" % (late * 1000))

        self.state = "스윙"
        t_fire = time.perf_counter()
        if self.armed:
            self.backend.swing()
        err = (t_fire + swing_lead_sec()) - tgt.impact_time
        self.stats.swung += 1
        self._last_shot_time = time.perf_counter()
        self._cooldown_until = self._last_shot_time + cfg.SWING_COOLDOWN_SEC
        self.state = "대기"
        #  ★ 타격했으면 곧바로 기본 자세로 돌아간다 ★
        if cfg.RETURN_TO_READY and cfg.RETURN_TO_READY_AFTER_SWING:
            self._want_ready = True
        self._emit("swing", tgt, sol, timing_error=err)
        if self.verbose:
            note = ""
            if not self.armed:
                note = "   (안전 모드 — 실제 지령 없음)"
            elif not self.backend.active_axes()["swing"]:
                note = "   (스윙 축 꺼짐 — 타이밍만 계산)"
            print("[모션] 스윙  트랙 #%d  조각 %d개  타이밍 오차 %+.0f ms%s"
                  % (track, sent, err * 1000, note))

    # ---------------------------------------------------------------- 보조
    def _pose_safe(self) -> tuple:
        """실제 위치 조회. 실패하면 마지막 지령값으로 대체한다."""
        try:
            return self.backend.current_pose()
        except Exception as e:
            if self._pose_warned != str(e):
                self._pose_warned = str(e)
                self._say("현재 위치를 읽지 못해 지령값을 씁니다: %s" % e)
            return (getattr(self.backend, "theta_deg", cfg.READY_THETA_DEG),
                    getattr(self.backend, "linear_cmd", 0.0))

    def _current_theta(self) -> float:
        return self._pose_safe()[0]

    def _current_linear(self) -> float:
        return self._pose_safe()[1]

    def _reject(self, tgt: ShotTarget, why: str, *,
                permanent: bool = True) -> None:
        """이번 목표를 쓰지 않는다.

        permanent=True 면 이 트랙을 닫는다(시간이 지났거나 확정적인 경우).
        False 면 사유만 한 번 알리고 계속 지켜본다.
        """
        if permanent:
            self._done_track = tgt.track_no
        elif tgt.track_no == self._warned_track:
            return                      # 같은 트랙에서 반복 출력 금지
        self._warned_track = tgt.track_no
        self._emit("reject", tgt, None, reason=why)
        if self.verbose:
            print("[모션] %s  트랙 #%d  %s"
                  % ("포기" if permanent else "보류", tgt.track_no, why))

    def _emit(self, kind: str, tgt, sol, **extra) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(kind, dict(target=tgt, solution=sol, **extra))
        except Exception:
            pass

    def _say(self, msg: str) -> None:
        if self.verbose:
            print("[모션] %s" % msg)

    def summary(self) -> str:
        s = self.stats
        return ("제출 %d  조준 %d  스윙 %d\n"
                "늦음 %d  도달불가 %d  오류 %d\n"
                "축: %s"
                % (s.submitted, s.posed, s.swung, s.too_late,
                   s.unreachable, s.errors, self.backend.axes_text()))


# ============================================================================
# 자체 시험
# ============================================================================

def _self_test() -> None:
    print("=" * 70)
    print(" 모션 워커 자체 시험 (시뮬레이터, 무장 상태)")
    print("=" * 70)
    be = SimBackend(verbose=True, realtime=True)
    w = MotionWorker(be, armed=True, verbose=True)
    w.start()
    try:
        cases = [
            ("가운데 · 여유 있음", 0.0, 260.0, 0.80),
            ("우측 끝 · 여유 있음", 340.0, 240.0, 0.80),
            ("좌측 · 촉박", -300.0, 280.0, 0.28),
            ("너무 높음 (도달 불가)", 0.0, 520.0, 0.70),
            ("이미 지남 (늦음)", 0.0, 260.0, 0.02),
        ]
        for i, (name, x, z, lead) in enumerate(cases, start=1):
            print("\n--- %s ---" % name)
            t_impact = time.perf_counter() + lead
            #  실제로는 매 프레임 갱신되므로 여러 번 제출한다
            for seq in range(12):
                w.submit(ShotTarget(impact_time=t_impact, x_mm=x, z_mm=z,
                                    track_no=i, seq=seq))
                time.sleep(0.01)
            time.sleep(max(0.0, t_impact - time.perf_counter()) + 0.35)
    finally:
        w.stop()
    print("\n통계: %s" % w.summary())
    print("\n모든 시험 통과")


if __name__ == "__main__":
    _self_test()
