#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamixel_link.py
=================
다이나믹셀 최소 통신 래퍼.

기본 자세를 잡는 데 필요한 것만 담았다.  모션 제어용 고속 경로(GroupSyncWrite
등)는 실기 백엔드를 붙일 때 따로 만든다.  여기서 하는 일은 넷뿐이다.

    1. 핑 — 어떤 ID 가 살아 있는가
    2. 토크 on/off — 손으로 자세를 잡으려면 꺼야 한다
    3. 현재 위치 읽기 — 자세를 잡은 뒤 그 raw 를 기억한다
    4. 조금씩 움직이기 — 손으로는 어려운 미세 조정

확장 위치 제어 모드
-------------------
이 로봇은 전 축이 **확장 위치 제어(모드 4)** 여야 한다.  기본 위치 제어(3)는
0~4095 로 감기기 때문에, 원점 부근에서 목표가 반대쪽으로 튈 수 있다.
모드 변경은 토크가 꺼져 있을 때만 먹는다.

시뮬레이터
----------
하드웨어 없이 절차를 확인할 수 있도록 SimLink 를 함께 둔다.
``--simulate`` 로 도구를 돌리면 이쪽이 쓰인다.
"""

from __future__ import annotations

import random
import threading
import time

try:
    from dynamixel_sdk import (COMM_SUCCESS, PacketHandler, PortHandler)
    HAS_SDK = True
    SDK_ERROR = None
except Exception as _e:                        # SDK 가 없어도 import 는 되게
    HAS_SDK = False
    SDK_ERROR = "%s: %s" % (type(_e).__name__, _e)
    COMM_SUCCESS = 0


PROTOCOL = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

MODE_EXTENDED_POSITION = 4
RAW_PER_REV = 4096


def raw_to_deg(raw: float) -> float:
    return float(raw) * 360.0 / RAW_PER_REV


def deg_to_raw(deg: float) -> int:
    return int(round(float(deg) * RAW_PER_REV / 360.0))


def nearest_equivalent(target: int, current: int) -> int:
    """current 에서 가장 가까운, target 과 각도가 같은 위치.

    확장 위치 제어에서는 4096 만큼 떨어진 값이 물리적으로 같은 각도다.
    누적값이 어긋나 있으면 그대로 보낼 때 모터가 한 바퀴를 통째로 돈다.
    4096 의 배수를 더하거나 빼서 이동량이 반 바퀴를 넘지 않게 만든다.
    """
    t, c = int(target), int(current)
    turns = round((c - t) / float(RAW_PER_REV))
    return t + int(turns) * RAW_PER_REV


class DynamixelError(RuntimeError):
    pass


# ============================================================================
# 실기
# ============================================================================

class DynamixelLink:
    """dynamixel_sdk 를 감싼 최소 인터페이스."""

    def __init__(self, port: str = "COM3", baudrate: int = 1000000,
                 ids=(1, 2, 5, 10)):
        self.port_name = port
        self.baudrate = int(baudrate)
        self.ids = tuple(int(i) for i in ids)
        self.port = None
        self.packet = None
        self._lock = threading.RLock()
        self.simulated = False

    # ------------------------------------------------------------ 수명주기
    def open(self) -> None:
        if not HAS_SDK:
            raise DynamixelError(
                "dynamixel_sdk 를 불러올 수 없습니다.\n%s\n\n"
                "설치:  pip install dynamixel-sdk" % SDK_ERROR)
        self.port = PortHandler(self.port_name)
        self.packet = PacketHandler(PROTOCOL)
        if not self.port.openPort():
            raise DynamixelError("포트를 열 수 없습니다: %s" % self.port_name)
        if not self.port.setBaudRate(self.baudrate):
            self.port.closePort()
            raise DynamixelError("보레이트 설정 실패: %d" % self.baudrate)

    def close(self) -> None:
        with self._lock:
            if self.port is not None:
                try:
                    self.port.closePort()
                finally:
                    self.port = None

    # ---------------------------------------------------------------- 통신
    #  ---- 재시도 ----
    #  스윙 순간의 큰 전류로 버스가 흔들리면 응답이 한 번씩 사라진다
    #  ("There is no status packet!").  한 번 실패했다고 포기하지 않는다.
    retries = 2
    retry_delay = 0.004

    def _try(self, fn, what: str):
        last = None
        for n in range(self.retries + 1):
            try:
                return fn()
            except DynamixelError as e:
                last = e
                if n < self.retries:
                    time.sleep(self.retry_delay)
        raise last if last else DynamixelError(what)

    def _check(self, dxl_id: int, comm: int, err: int, what: str) -> None:
        if comm != COMM_SUCCESS:
            raise DynamixelError("ID %d %s 실패: %s"
                                 % (dxl_id, what,
                                    self.packet.getTxRxResult(comm)))
        if err != 0:
            raise DynamixelError("ID %d %s 하드웨어 오류: %s"
                                 % (dxl_id, what,
                                    self.packet.getRxPacketError(err)))

    def ping(self, dxl_id: int):
        with self._lock:
            model, comm, err = self.packet.ping(self.port, dxl_id)
        if comm != COMM_SUCCESS or err != 0:
            return None
        return int(model)

    def ping_all(self) -> dict:
        return {i: self.ping(i) for i in self.ids}

    def read_position(self, dxl_id: int) -> int:
        def once():
            with self._lock:
                val, comm, err = self.packet.read4ByteTxRx(
                    self.port, dxl_id, ADDR_PRESENT_POSITION)
            self._check(dxl_id, comm, err, "위치 읽기")
            v = int(val)
            return v - (1 << 32) if v >= (1 << 31) else v   # 부호 있는 32비트
        return self._try(once, "위치 읽기")

    def read_all(self) -> dict:
        out = {}
        for i in self.ids:
            try:
                out[i] = self.read_position(i)
            except DynamixelError:
                out[i] = None
        return out

    def torque(self, dxl_id: int, on: bool) -> None:
        def once():
            with self._lock:
                comm, err = self.packet.write1ByteTxRx(
                    self.port, dxl_id, ADDR_TORQUE_ENABLE, 1 if on else 0)
            self._check(dxl_id, comm, err, "토크 설정")
        self._try(once, "토크 설정")

    def is_torque_on(self, dxl_id: int) -> bool:
        def once():
            with self._lock:
                val, comm, err = self.packet.read1ByteTxRx(
                    self.port, dxl_id, ADDR_TORQUE_ENABLE)
            self._check(dxl_id, comm, err, "토크 읽기")
            return bool(val)
        return self._try(once, "토크 읽기")

    def torque_all(self, on: bool) -> None:
        for i in self.ids:
            try:
                self.torque(i, on)
            except DynamixelError as e:
                print("[DXL] %s" % e)

    def read_operating_mode(self, dxl_id: int) -> int:
        with self._lock:
            val, comm, err = self.packet.read1ByteTxRx(
                self.port, dxl_id, ADDR_OPERATING_MODE)
        self._check(dxl_id, comm, err, "모드 읽기")
        return int(val)

    def ensure_extended_mode(self, dxl_id: int) -> bool:
        """확장 위치 제어(4)로 맞춘다. 바꿨으면 True.

        모드 변경은 토크가 꺼져 있을 때만 먹는다.
        """
        if self.read_operating_mode(dxl_id) == MODE_EXTENDED_POSITION:
            return False
        self.torque(dxl_id, False)
        with self._lock:
            comm, err = self.packet.write1ByteTxRx(
                self.port, dxl_id, ADDR_OPERATING_MODE,
                MODE_EXTENDED_POSITION)
        self._check(dxl_id, comm, err, "모드 변경")
        return True

    def set_profile(self, dxl_id: int, velocity: int, acceleration: int) -> None:
        def once():
            with self._lock:
                comm, err = self.packet.write4ByteTxRx(
                    self.port, dxl_id, ADDR_PROFILE_ACCELERATION,
                    int(acceleration))
                self._check(dxl_id, comm, err, "가속도 설정")
                comm, err = self.packet.write4ByteTxRx(
                    self.port, dxl_id, ADDR_PROFILE_VELOCITY, int(velocity))
            self._check(dxl_id, comm, err, "속도 설정")
        self._try(once, "프로파일 설정")

    def write_position(self, dxl_id: int, raw: int) -> None:
        v = int(raw)
        if v < 0:
            v += (1 << 32)

        def once():
            with self._lock:
                comm, err = self.packet.write4ByteTxRx(
                    self.port, dxl_id, ADDR_GOAL_POSITION, v)
            self._check(dxl_id, comm, err, "목표 위치 쓰기")
        self._try(once, "목표 위치 쓰기")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.close()


# ============================================================================
# 시뮬레이터
# ============================================================================

class SimLink(DynamixelLink):
    """하드웨어 없이 절차만 확인한다.

    모터를 갈아끼운 상황을 흉내내려고, 시작 raw 값을 ID 마다 제각각으로
    둔다.  기존 postures.json 과 맞지 않는 것이 정상이다.
    """

    def __init__(self, port="SIM", baudrate=1000000, ids=(1, 2, 5, 10)):
        super().__init__(port, baudrate, ids)
        self.simulated = True
        self._pos = {i: random.randint(-2000, 3900) for i in ids}
        self._goal = dict(self._pos)
        self._torque = {i: False for i in ids}
        self._mode = {i: 3 for i in ids}          # 일부러 기본 모드로 시작
        self._t = time.perf_counter()

    def open(self):
        print("[SIM] 가상 다이나믹셀 열림  초기 raw = %s" % self._pos)

    def close(self):
        print("[SIM] 가상 다이나믹셀 닫힘")

    def _step(self):
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        for i in self.ids:
            if self._torque[i]:
                d = self._goal[i] - self._pos[i]
                step = max(1, int(3000 * dt))
                self._pos[i] += max(-step, min(step, d))
            else:
                #  토크가 꺼져 있으면 중력으로 조금씩 흘러내린다
                if i in (1, 2):
                    self._pos[i] += random.choice((0, 0, 1, -1))

    def ping(self, dxl_id):
        return 1020 if dxl_id in self.ids else None

    def read_position(self, dxl_id):
        self._step()
        return int(self._pos[dxl_id]) + random.choice((-1, 0, 1))

    def torque(self, dxl_id, on):
        self._step()
        if on:
            self._goal[dxl_id] = self._pos[dxl_id]
        self._torque[dxl_id] = bool(on)

    def is_torque_on(self, dxl_id):
        return bool(self._torque[dxl_id])

    def read_operating_mode(self, dxl_id):
        return self._mode[dxl_id]

    def ensure_extended_mode(self, dxl_id):
        if self._mode[dxl_id] == MODE_EXTENDED_POSITION:
            return False
        self._torque[dxl_id] = False
        self._mode[dxl_id] = MODE_EXTENDED_POSITION
        return True

    def set_profile(self, dxl_id, velocity, acceleration):
        pass

    def write_position(self, dxl_id, raw):
        self._goal[dxl_id] = int(raw)


def make_link(port="COM3", baudrate=1000000, ids=(1, 2, 5, 10),
              simulate: bool = False):
    return (SimLink(port, baudrate, ids) if simulate
            else DynamixelLink(port, baudrate, ids))


if __name__ == "__main__":
    print("dynamixel_sdk: %s" % ("있음" if HAS_SDK else "없음 -> %s" % SDK_ERROR))
    link = make_link(simulate=True)
    link.open()
    print("핑:", link.ping_all())
    for i in link.ids:
        changed = link.ensure_extended_mode(i)
        print("  ID %2d 확장 위치 모드 %s" % (i, "로 변경" if changed else " (이미)"))
    print("현재 raw:", link.read_all())
    link.close()
