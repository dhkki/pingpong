#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_backend.py
================
실기 백엔드 — 모션 워커의 지령을 실제 축으로 내보낸다.

축 셋을 각각 켜고 끌 수 있다
----------------------------
    enable_linear   리니어축만 (폭 이동)
    enable_tilt     팔만 (틸트 + 라켓 면 보정)
    enable_swing    스윙만 (바닥 원판)

단계적으로 시험하라고 나눠 둔 것이다.  예를 들어 리니어만 켜서 좌우 추종을
먼저 보고, 다음에 팔을 켜 높이를 맞추고, 마지막에 스윙을 켠다.  꺼진 축은
지령이 나가지 않고 이동 시간 계산에서도 빠지므로, 켜져 있는 축만으로
타이밍이 다시 계산된다.

각도 규약 (기본 자세 raw 기준 절대 위치)
----------------------------------------
    ID 1  = home1 + base_sign · theta
    ID 2  = home2 - base_sign · theta          (마주보아 반대로 돈다)
    ID 5  = home5 + base_sign · id5_sign · theta · ratio
    ID 10 = home10 + 스윙 각도

매번 현재 위치를 읽어 델타를 더하는 상대 이동이 아니라, 기본 자세를 원점으로
삼는 **절대 위치**다.  읽기 왕복이 없고 오차가 누적되지 않는다.
"""

from __future__ import annotations

import threading
import time

import home_pose as hp
import robot_config as cfg
from dynamixel_link import (DynamixelError, MODE_EXTENDED_POSITION,
                            deg_to_raw, make_link, nearest_equivalent,
                            raw_to_deg)
from linear_link import make_linear
from motion_control import RobotBackend

BAR_A_ID, BAR_B_ID, PADDLE_ID, SWING_ID = 1, 2, 5, 10


class BackendError(RuntimeError):
    pass


class RealBackend(RobotBackend):
    """다이나믹셀(1,2,5,10) + Ajin 리니어축."""

    def __init__(self, *, port="COM3", baudrate=1000000,
                 dll_path=None, mot_path=None,
                 simulate_dxl=False, simulate_linear=False,
                 enable_linear=True, enable_tilt=True, enable_swing=True,
                 start_cmd=0.0, verbose=True):
        self.verbose = bool(verbose)
        self.enable_linear = bool(enable_linear)
        self.enable_tilt = bool(enable_tilt)
        self.enable_swing = bool(enable_swing)

        self.dxl = make_link(port, baudrate,
                             (BAR_A_ID, BAR_B_ID, PADDLE_ID, SWING_ID),
                             simulate=simulate_dxl)
        self.dxl.retries = int(cfg.DXL_RETRY)
        self.dxl.retry_delay = float(cfg.DXL_RETRY_DELAY_SEC)
        self.linear = make_linear(
            dll_path, mot_path, simulate=simulate_linear,
            velocity=cfg.LINEAR_VELOCITY,
            acceleration=cfg.LINEAR_ACCELERATION,
            start_cmd=start_cmd,
            min_cmd=cfg.ARM.linear_min_cmd, max_cmd=cfg.ARM.linear_max_cmd)

        self.home = None                  # {id: raw}
        self.theta_deg = cfg.READY_THETA_DEG
        self.linear_cmd = float(start_cmd)
        self.swing_deg = cfg.SWING_REST_DEG
        self.opened = False
        self._lock = threading.RLock()
        self._swinging = False
        self.last_swing_sec = 0.0
        self._dxl_opened = False            # 포트를 실제로 열었는가
        self._wrap_turns = {}               # ID 별 랩 보정량 (로그 폭주 방지)
        self._pose_cache = (0.0, None)      # (읽은 시각, (theta, linear))
        self.pose_poll_sec = 0.04           # 실측 읽기 주기(초)

    # ------------------------------------------------------------- 수명주기
    def open(self) -> None:
        self.home = hp.load()
        if self.home is None:
            raise BackendError(
                "기본 자세가 저장되어 있지 않습니다.\n"
                "먼저  python set_home_pose.py  로 잡아 주세요.")
        missing = [i for i in (BAR_A_ID, BAR_B_ID, PADDLE_ID, SWING_ID)
                   if i not in self.home]
        if missing:
            raise BackendError("기본 자세에 없는 ID: %s" % missing)

        if self.enable_tilt or self.enable_swing:
            self.dxl.open()
            self._dxl_opened = True
            found = self.dxl.ping_all()
            dead = [i for i, m in found.items() if m is None]
            if dead:
                raise BackendError("응답 없는 모터: %s"
                                   % ", ".join("ID %d" % i for i in dead))
            for i in self.dxl.ids:
                if self.dxl.ensure_extended_mode(i):
                    self._say("ID %d 를 확장 위치 모드로 바꿨습니다" % i)
            #  ★ 순서가 중요하다 ★
            #    토크를 켜는 순간 모터는 Goal Position 레지스터에 남아 있던
            #    '옛 목표'로 즉시 달려간다.  그 값이 한 바퀴 떨어져 있으면
            #    360° 를 통째로 돈다.  그래서 목표를 현재 위치로 먼저 써 두고
            #    그다음에 토크를 켠다.
            for i in self.dxl.ids:
                pos = self.dxl.read_position(i)
                self.dxl.set_profile(i, cfg.POSE_VELOCITY,
                                     cfg.POSE_ACCELERATION)
                self.dxl.write_position(i, pos)     # 목표 먼저
                self.dxl.torque(i, True)            # 그다음 토크
            self._report_home_gap()

        if self.enable_linear:
            self.linear.open()
            self.linear.servo_on()
            self._say("리니어 준비. 현재 지령 %.2f 단위로 추적을 시작합니다. "
                      "축이 실제로 이 위치에 있어야 합니다."
                      % self.linear.position)

        self.opened = True
        self._say("백엔드 열림 — %s" % self.axes_text())

    def _report_home_gap(self) -> None:
        """지금 자세가 기본 자세에서 얼마나 떨어져 있는지 알려준다."""
        try:
            cur = self.dxl.read_all()
        except DynamixelError:
            return
        parts = []
        for i in sorted(self.home):
            if cur.get(i) is None:
                continue
            parts.append("ID%d %+.1f°" % (i, raw_to_deg(cur[i] - self.home[i])))
        self._say("기본 자세 대비 현재: %s" % "  ".join(parts))

    def park_linear(self, cmd: float = 0.0, timeout: float = 25.0) -> None:
        """리니어를 지정 지령으로 **천천히** 되돌린다.

        엔코더가 없어 지령값을 추적하는 축이라, 매번 같은 자리에서 시작하는
        것이 중요하다.  그래서 종료할 때 반드시 원위치로 돌려 둔다.
        """
        #  ★ 축 사용 여부와 무관하게 되돌린다 ★
        #    시험 중에 리니어 축을 꺼 두었더라도, 이미 움직여 놓은 캐리지는
        #    원위치로 돌려야 다음 실행이 같은 자리에서 시작한다.
        if not getattr(self.linear, "opened", False):
            self._say("리니어가 열려 있지 않아 복귀를 건너뜁니다.")
            return
        target = float(cmd)
        if abs(target - self.linear.position) < 1e-6:
            self._say("리니어가 이미 원위치(%.2f)입니다." % target)
            return
        self._say("리니어 원위치로 복귀 %.2f -> %.2f (천천히)"
                  % (self.linear.position, target))
        try:
            #  시작·종료 위치는 랠리용 범위 밖일 수 있다(테이블 사이드라인).
            #  클램프를 우회해 그대로 보낸다.
            self.linear.move_to(target,
                                velocity=cfg.LINEAR_RETURN_VELOCITY,
                                acceleration=cfg.LINEAR_RETURN_ACCELERATION,
                                allow_outside=True)
            if not self.linear.wait(timeout=timeout):
                self._say("복귀가 %.0f 초 안에 끝나지 않았습니다." % timeout)
        except Exception as e:
            print("[백엔드] 리니어 복귀 실패: %s" % e)
        self.linear_cmd = self.linear.position

    def warmup_pose(self, theta_deg: float, linear_cmd: float) -> None:
        """준비운동용 자세 지령.

        평소 조준보다 느린 프로파일로 움직여, 사람이 눈으로 따라갈 수
        있게 한다.  리니어는 조각 없이 목표까지 그대로 간다.
        """
        with self._lock:
            if self.enable_linear:
                self.linear.move_to(linear_cmd,
                                    velocity=cfg.WARMUP_LINEAR_VELOCITY,
                                    acceleration=cfg.WARMUP_LINEAR_ACCELERATION)
                self.linear_cmd = float(linear_cmd)
            if self.enable_tilt:
                try:
                    for i in (BAR_A_ID, BAR_B_ID, PADDLE_ID):
                        self.dxl.set_profile(i, cfg.WARMUP_DXL_VELOCITY,
                                             cfg.WARMUP_DXL_ACCELERATION)
                    self._write_tilt(self._tilt_targets(theta_deg))
                    self.theta_deg = float(theta_deg)
                except DynamixelError as e:
                    self._say("준비운동 지령 실패: %s" % e)

    def park_arm(self) -> None:
        """토크를 풀기 전에 팔을 아래로 천천히 내려 둔다.

        그냥 토크를 끊으면 막대가 중력으로 낙하해 구조물을 때린다.
        대기 자세보다 PARK_EXTRA_TILT_DEG 만큼 더 눕힌 자세로 아주 느리게
        이동한 뒤, 도착을 확인하고 잠시 쉬었다가 토크를 푼다.
        """
        if not self.enable_tilt or not self._dxl_opened or self.home is None:
            return
        target = max(cfg.ARM.theta_min_deg,
                     min(cfg.ARM.theta_max_deg,
                         cfg.READY_THETA_DEG - abs(cfg.PARK_EXTRA_TILT_DEG)))
        self._say("토크 해제 전 안착: theta %+.1f° 로 천천히 내립니다"
                  % target)
        try:
            for i in (BAR_A_ID, BAR_B_ID, PADDLE_ID):
                self.dxl.set_profile(i, cfg.PARK_DXL_VELOCITY,
                                     cfg.PARK_DXL_ACCELERATION)
            tg = self._tilt_targets(target)
            self._write_tilt(tg)
            self.theta_deg = target
        except DynamixelError as e:
            self._say("안착 지령 실패: %s" % e)
            return

        #  도착까지 기다린다 (ID 1 위치로 판정)
        #  ★ 랩 보정된 목표와 비교해야 한다 ★
        #    보정 전 값과 비교하면 영원히 도달하지 못한 것으로 보인다.
        goal = self._wrap_safe(BAR_A_ID, tg[BAR_A_ID])
        tol = max(12, abs(deg_to_raw(1.5)))
        t_end = time.perf_counter() + cfg.PARK_TILT_TIMEOUT_SEC
        while time.perf_counter() < t_end:
            try:
                if abs(self.dxl.read_position(BAR_A_ID) - goal) <= tol:
                    break
            except DynamixelError:
                break
            time.sleep(0.02)
        else:
            self._say("안착이 %.1f 초 안에 끝나지 않았습니다."
                      % cfg.PARK_TILT_TIMEOUT_SEC)
        time.sleep(max(0.0, cfg.PARK_SETTLE_SEC))
        self._say("안착 완료")

    def close(self) -> None:
        #  무조건 원위치로 돌린 뒤에 닫는다
        if cfg.PARK_LINEAR_CMD_ON_EXIT is not None:
            self.park_linear(float(cfg.PARK_LINEAR_CMD_ON_EXIT),
                             timeout=cfg.PARK_TIMEOUT_SEC)
        #  ★ 토크를 풀기 전에 팔을 내려 둔다 ★
        if cfg.TORQUE_OFF_ON_EXIT:
            try:
                self.park_arm()
            except Exception as e:
                print("[백엔드] 안착 중 오류: %s" % e)
        try:
            if self.linear is not None and getattr(self.linear, "opened", False):
                self.linear.close()
        finally:
            #  ★ 축 사용 여부와 무관하게, 포트를 열었으면 반드시 토크를 끈다 ★
            #    전에는 enable_tilt/enable_swing 을 보고 있어서, 시험하느라
            #    축을 꺼 두면 토크가 걸린 채로 프로그램이 끝났다.
            if self._dxl_opened:
                if cfg.TORQUE_OFF_ON_EXIT:
                    try:
                        self.dxl.torque_all(False)
                        self._say("다이나믹셀 토크 해제")
                    except Exception as e:
                        print("[백엔드] 토크 해제 실패: %s" % e)
                try:
                    self.dxl.close()
                except Exception as e:
                    print("[백엔드] 포트 닫기 실패: %s" % e)
                self._dxl_opened = False
        self.opened = False
        self._say("백엔드 닫힘")

    # ------------------------------------------------------------- 각도 변환
    def recover(self, dxl_id: int) -> bool:
        """통신이 끊겼거나 토크가 풀린 모터를 되살린다.

        스윙 순간의 큰 전류로 버스 전압이 흔들리면 응답이 사라지거나
        모터가 셧다운되며 토크가 풀린다.  그때 그냥 포기하면 그 축은
        랠리 내내 죽어 있으므로, 핑 -> 토크 재투입 -> 현재 위치 유지로
        되살린다.
        """
        if not cfg.DXL_RECOVER_ON_FAIL:
            return False
        try:
            if self.dxl.ping(dxl_id) is None:
                self._say("ID %d 응답 없음 (배선·전원 확인)" % dxl_id)
                return False
            if not self.dxl.is_torque_on(dxl_id):
                self._say("ID %d 토크가 풀려 있어 다시 겁니다" % dxl_id)
            pos = self.dxl.read_position(dxl_id)
            self.dxl.set_profile(dxl_id, cfg.POSE_VELOCITY,
                                 cfg.POSE_ACCELERATION)
            self.dxl.write_position(dxl_id, pos)     # 목표 먼저
            self.dxl.torque(dxl_id, True)            # 그다음 토크
            self._say("ID %d 복구 완료 (raw %d)" % (dxl_id, pos))
            return True
        except DynamixelError as e:
            self._say("ID %d 복구 실패: %s" % (dxl_id, e))
            return False

    def _wrap_safe(self, dxl_id: int, raw: int) -> int:
        """한 바퀴를 통째로 도는 것을 막는다.

        확장 위치 제어에서 누적값이 어긋나 있으면 물리적으로 같은 각도인데도
        4096 만큼 떨어진 값이 되어, 그대로 보내면 모터가 한 바퀴 돈다.
        현재 위치를 읽어 가장 가까운 등가 위치로 옮겨 보낸다.
        """
        if not cfg.WRAP_SAFE_TARGET or dxl_id not in cfg.WRAP_SAFE_IDS:
            return int(raw)
        try:
            cur = self.dxl.read_position(dxl_id)
        except DynamixelError:
            return int(raw)
        fixed = nearest_equivalent(int(raw), cur)
        turns = (fixed - int(raw)) // 4096
        #  보정량이 바뀔 때만 알린다.  매번 찍으면 콘솔이 폭주한다.
        if self._wrap_turns.get(dxl_id) != turns:
            self._wrap_turns[dxl_id] = turns
            if turns:
                self._say("ID %d 목표를 %+d 바퀴 옮겨 보냅니다 "
                          "(한 바퀴 회전 방지, 현재 raw %d)"
                          % (dxl_id, turns, cur))
        return fixed

    def _write_tilt(self, targets: dict) -> None:
        """틸트 지령. 실패하면 그 축만 되살려 다시 보낸다."""
        for i, raw in targets.items():
            raw = self._wrap_safe(i, raw)
            try:
                self.dxl.write_position(i, raw)
            except DynamixelError as e:
                self._say("ID %d 지령 실패: %s" % (i, e))
                if self.recover(i):
                    try:
                        self.dxl.write_position(i, raw)
                    except DynamixelError as e2:
                        self._say("ID %d 재지령 실패: %s" % (i, e2))

    def _tilt_targets(self, theta_deg: float) -> dict:
        th = max(cfg.ARM.theta_min_deg,
                 min(cfg.ARM.theta_max_deg, float(theta_deg)))
        d = deg_to_raw(th * cfg.BASE_SIGN)
        d5 = deg_to_raw(th * cfg.BASE_SIGN * cfg.ID5_SIGN
                        * cfg.ID5_COMPENSATION_RATIO)
        return {BAR_A_ID: self.home[BAR_A_ID] + d,
                BAR_B_ID: self.home[BAR_B_ID] - d,
                PADDLE_ID: self.home[PADDLE_ID] + d5}

    def current_pose(self) -> tuple:
        """지금 이 순간의 **실제** (theta_deg, 리니어 지령).

        theta 는 ID 1 의 현재 위치를 읽어 기본 자세 raw 기준으로 환산한다.
        리니어는 엔코더가 없어 조각의 경과 시간과 속도 프로파일로 추정한다.
        시리얼 왕복이 있으므로 pose_poll_sec 주기로만 읽고 그 사이는 캐시.
        """
        now = time.time()
        t_last, cached = self._pose_cache
        th = self.theta_deg
        if self.enable_tilt and self.home is not None:
            if cached is not None and now - t_last < self.pose_poll_sec:
                th = cached[0]
            else:
                try:
                    raw = self.dxl.read_position(BAR_A_ID)
                    th = raw_to_deg(raw - self.home[BAR_A_ID]) / cfg.BASE_SIGN
                except DynamixelError:
                    th = self.theta_deg
        lin = self.linear_cmd
        if self.enable_linear and getattr(self.linear, "opened", False):
            #  예전 버전 linear_link 에는 추정 함수가 없다 -> 지령값으로 대체
            est = getattr(self.linear, "estimated_position", None)
            lin = est() if callable(est) else getattr(self.linear, "position",
                                                      self.linear_cmd)
        self._pose_cache = (now, (th, lin))
        return (th, lin)

    def _swing_raw(self, deg: float) -> int:
        return self.home[SWING_ID] + deg_to_raw(float(deg))

    # ---------------------------------------------------------------- 지령
    def set_pose(self, theta_deg: float, linear_cmd: float) -> None:
        """틸트와 리니어를 **같은 순간에** 지령한다.

        라켓의 폭 방향 위치가 두 축 모두에 의존하므로, 따로 보내면 그 사이
        라켓이 최대 190mm 옆으로 나갔다 돌아온다.
        """
        with self._lock:
            if self.enable_linear:
                self.linear.move_to(linear_cmd)
                self.linear_cmd = float(linear_cmd)
            if self.enable_tilt:
                self._write_tilt(self._tilt_targets(theta_deg))
                self.theta_deg = float(theta_deg)

    def go_ready(self, theta_deg: float, linear_cmd: float, *,
                 slow: bool = False) -> None:
        """대기 자세로 이동.

        slow=True (무장 직후 첫 정렬, [X=0 으로 정렬] 버튼) 면 저속으로,
        아니면 랠리 사이 복귀이므로 타격 때와 같은 속도로 움직인다.
        """
        lv = (cfg.ARM_ALIGN_LINEAR_VELOCITY if slow
              else cfg.READY_LINEAR_VELOCITY)
        la = (cfg.ARM_ALIGN_LINEAR_ACCELERATION if slow
              else cfg.READY_LINEAR_ACCELERATION)
        dv = cfg.ARM_ALIGN_DXL_VELOCITY if slow else cfg.READY_DXL_VELOCITY
        da = (cfg.ARM_ALIGN_DXL_ACCELERATION if slow
              else cfg.READY_DXL_ACCELERATION)
        with self._lock:
            if self.enable_linear:
                self.linear.move_to(linear_cmd, velocity=lv, acceleration=la)
                self.linear_cmd = float(linear_cmd)
            if self.enable_tilt:
                try:
                    for i in (BAR_A_ID, BAR_B_ID, PADDLE_ID):
                        try:
                            self.dxl.set_profile(i, dv, da)
                        except DynamixelError as e:
                            self._say("ID %d 프로파일 실패: %s" % (i, e))
                            self.recover(i)
                    tg = self._tilt_targets(theta_deg)
                    self._write_tilt(tg)
                    self.theta_deg = float(theta_deg)
                    self._say("틸트 지령 theta %+.1f°  %s  (기본자세 %s)"
                              % (theta_deg,
                                 " ".join("ID%d %d" % (k, tg[k])
                                          for k in sorted(tg)),
                                 " ".join("ID%d %d" % (k, self.home[k])
                                          for k in sorted(tg))))
                    time.sleep(0.05)
                    for i in (BAR_A_ID, BAR_B_ID, PADDLE_ID):
                        self.dxl.set_profile(i, cfg.POSE_VELOCITY,
                                             cfg.POSE_ACCELERATION)
                except DynamixelError as e:
                    print("[백엔드] 대기 자세 실패: %s" % e)
            if self.enable_swing:
                try:
                    self.dxl.set_profile(SWING_ID, dv, da)
                    self.dxl.write_position(
                        SWING_ID,
                        self._wrap_safe(SWING_ID,
                                        self._swing_raw(cfg.SWING_REST_DEG)))
                    self.swing_deg = cfg.SWING_REST_DEG
                except DynamixelError as e:
                    print("[백엔드] 스윙 대기각 실패: %s" % e)

    def swing(self) -> float:
        """대기각 -> 타격각 -> 대기각.  걸린 시간(s)."""
        if not self.enable_swing:
            return 0.0
        t0 = time.perf_counter()
        try:
            self.dxl.set_profile(SWING_ID, cfg.SWING_VELOCITY,
                                 cfg.SWING_ACCELERATION)
            self.dxl.write_position(
                SWING_ID, self._wrap_safe(SWING_ID,
                                          self._swing_raw(cfg.SWING_FRONT_DEG)))
            self.swing_deg = cfg.SWING_FRONT_DEG
        except DynamixelError as e:
            print("[백엔드] 스윙 실패: %s" % e)
            return 0.0
        #  복귀는 별도 스레드에서. 여기서 자면 모션 루프가 멈춘다.
        threading.Thread(target=self._swing_return, daemon=True,
                         name="swing-return").start()
        self.last_swing_sec = time.perf_counter() - t0
        return self.last_swing_sec

    def measure_swing(self, *, timeout: float = 2.0) -> float:
        """대기각 -> 타격각 에 실제로 걸리는 시간을 잰다.

        ID 10 의 현재 위치를 읽어 타격각에 닿는 순간까지의 시간을 재고,
        천천히 대기각으로 되돌린다.  이 값의 절반이 스윙 발사 리드타임이다.
        """
        if not self.enable_swing:
            raise BackendError("스윙 축이 꺼져 있습니다.")
        rest_raw = self._swing_raw(cfg.SWING_REST_DEG)
        front_raw = self._swing_raw(cfg.SWING_FRONT_DEG)
        tol = max(8, abs(deg_to_raw(2.0)))

        #  대기각으로 먼저 천천히 이동
        self.dxl.set_profile(SWING_ID, cfg.READY_DXL_VELOCITY,
                             cfg.READY_DXL_ACCELERATION)
        self.dxl.write_position(SWING_ID, rest_raw)
        t_end = time.perf_counter() + 3.0
        while time.perf_counter() < t_end:
            if abs(self.dxl.read_position(SWING_ID) - rest_raw) <= tol:
                break
            time.sleep(0.01)
        time.sleep(0.2)

        #  타격
        self.dxl.set_profile(SWING_ID, cfg.SWING_VELOCITY,
                             cfg.SWING_ACCELERATION)
        t0 = time.perf_counter()
        self.dxl.write_position(SWING_ID, front_raw)
        elapsed = timeout
        while time.perf_counter() - t0 < timeout:
            if abs(self.dxl.read_position(SWING_ID) - front_raw) <= tol:
                elapsed = time.perf_counter() - t0
                break
            time.sleep(0.001)
        self.swing_deg = cfg.SWING_FRONT_DEG
        self.last_swing_sec = elapsed

        #  대기각으로 천천히 복귀
        time.sleep(0.15)
        self.dxl.set_profile(SWING_ID, cfg.READY_DXL_VELOCITY,
                             cfg.READY_DXL_ACCELERATION)
        self.dxl.write_position(SWING_ID, rest_raw)
        self.swing_deg = cfg.SWING_REST_DEG
        return elapsed

    def _swing_return(self) -> None:
        time.sleep(cfg.SWING_DURATION_SEC + cfg.SWING_RETURN_HOLD_SEC)
        try:
            v = (cfg.SWING_RETURN_VELOCITY if cfg.SWING_RETURN_VELOCITY > 0
                 else cfg.SWING_VELOCITY)
            self.dxl.set_profile(SWING_ID, v, cfg.SWING_ACCELERATION)
            self.dxl.write_position(
                SWING_ID, self._wrap_safe(SWING_ID,
                                          self._swing_raw(cfg.SWING_REST_DEG)))
            self.swing_deg = cfg.SWING_REST_DEG
        except DynamixelError as e:
            print("[백엔드] 스윙 복귀 실패: %s" % e)
            if self.recover(SWING_ID):
                try:
                    self.dxl.write_position(
                        SWING_ID,
                        self._wrap_safe(SWING_ID,
                                        self._swing_raw(cfg.SWING_REST_DEG)))
                    self.swing_deg = cfg.SWING_REST_DEG
                except DynamixelError:
                    pass

    # ---------------------------------------------------------------- 안전
    def emergency_stop(self) -> None:
        if self.enable_linear:
            try:
                self.linear.emergency_stop()
            except Exception as e:
                print("[백엔드] 리니어 비상정지 실패: %s" % e)
        if self.enable_tilt or self.enable_swing:
            try:
                #  토크를 끊는 대신 그 자리에 정지시킨다.  갑자기 풀면
                #  막대가 낙하한다.
                for i in self.dxl.ids:
                    self.dxl.write_position(i, self.dxl.read_position(i))
            except DynamixelError as e:
                print("[백엔드] 다이나믹셀 정지 실패: %s" % e)
        print("[백엔드] 비상정지")

    def torque_off(self) -> None:
        """토크 해제. 막대가 내려앉으므로 받친 뒤에 부를 것."""
        try:
            self.dxl.torque_all(False)
        except Exception as e:
            print("[백엔드] 토크 해제 실패: %s" % e)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print("[백엔드] %s" % msg)

    def status_text(self) -> str:
        return ("%s | theta %+.1f°  리니어 %.2f%s"
                % (self.axes_text(), self.theta_deg, self.linear_cmd,
                   "  (이동 중)" if getattr(self.linear, "moving", False)
                   else ""))


# ============================================================================
# 자체 시험
# ============================================================================

def _self_test() -> None:
    import home_pose
    print("=" * 70)
    print(" 실기 백엔드 자체 시험 (양쪽 모두 시뮬레이션)")
    print("=" * 70)

    if home_pose.load() is None:
        raw = {1: 2048, 2: 2048, 5: 2048, 10: 2048}
        home_pose.save(raw, note="자체 시험용 임시값")
        print("기본 자세가 없어 임시값을 저장했습니다: %s" % raw)

    be = RealBackend(simulate_dxl=True, simulate_linear=True)
    be.open()
    try:
        print("\n[1] 축 전부 켜고 자세 지령")
        be.set_pose(-50.0, 40.0)
        print("   %s" % be.status_text())
        print("   틸트 raw 목표: %s" % be._tilt_targets(-50.0))

        print("\n[2] 리니어만")
        be.enable_linear, be.enable_tilt, be.enable_swing = True, False, False
        th_before = be.theta_deg
        be.set_pose(-20.0, 80.0)
        print("   %s" % be.status_text())
        print("   theta 는 그대로여야 한다: %+.1f -> %+.1f"
              % (th_before, be.theta_deg))

        print("\n[3] 팔만")
        be.enable_linear, be.enable_tilt = False, True
        lin_before = be.linear_cmd
        be.set_pose(-60.0, 10.0)
        print("   %s" % be.status_text())
        print("   리니어 지령은 그대로여야 한다: %.2f -> %.2f"
              % (lin_before, be.linear_cmd))

        print("\n[4] 스윙만")
        be.enable_tilt, be.enable_swing = False, True
        t = be.swing()
        print("   스윙 지령 %.1f ms 만에 반환 (복귀는 별도 스레드)" % (t * 1000))
        time.sleep(cfg.SWING_DURATION_SEC + cfg.SWING_RETURN_HOLD_SEC + 0.1)
        print("   복귀 후 스윙각 %+.1f°" % be.swing_deg)

        print("\n[5] 스윙 꺼짐")
        be.enable_swing = False
        print("   swing() 반환 %.1f ms (지령 없음)" % (be.swing() * 1000))
    finally:
        be.close()
    print("\n모든 시험 통과")


if __name__ == "__main__":
    _self_test()
