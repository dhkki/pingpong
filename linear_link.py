#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_link.py
==============
리니어축 링크 — LinearMotorNative.dll (Ajinextek AXL) 래퍼.

이 축의 특수 사정
-----------------
1. **위치 피드백이 없다.**  엔코더가 배선되어 있지 않아 실제 위치 읽기가
   항상 0 을 돌려준다.  그래서 지령값을 소프트웨어로 추적한다.
2. **이동 명령이 블로킹이다.**  DLL 이 AxmMovePos 를 쓰기 때문에 이동이
   끝날 때까지 반환하지 않고, 그동안 내부 뮤텍스를 쥐고 있다.  호출자가
   막히지 않도록 **별도 스레드**에서 돌린다.
   DLL 을 AxmMoveStartPos 로 고쳤다면 LM_StartMoveAbsolute 가 export 되고,
   이 래퍼가 자동으로 그쪽을 쓴다.
3. **한 번 출발하면 중간에 못 멈춘다.**  이동 중에 새 목표가 들어오면
   슬롯 하나에 보관했다가 끝나는 즉시 이어서 보낸다.  중간 목표는 버린다.
4. **LM_HomeStart 에 버그가 있다.**  원점복귀 대신 비상정지를 호출하는
   DLL 이 있으므로, 이 래퍼는 홈잉을 자동으로 부르지 않는다.

단위
----
지령 단위다.  1 단위 = 약 10.37mm (실측).  안전 범위 0~110 단위.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
import time


class LinearError(RuntimeError):
    pass


def profile_progress(u: float) -> float:
    """이동 시간의 u(0~1) 지점에서 지나온 거리의 비율.

    가감속이 대칭인 삼각 프로파일 기준이라 앞뒤가 느리고 가운데가 빠르다.
    사다리꼴 구간에서도 실제와 크게 다르지 않아, 화면 표시용으로 충분하다.
    """
    u = max(0.0, min(1.0, float(u)))
    return 2.0 * u * u if u < 0.5 else 1.0 - 2.0 * (1.0 - u) ** 2


def move_time_sec(distance: float, vmax: float, accel: float) -> float:
    """사다리꼴/삼각 프로파일 이동 시간(s)."""
    d = abs(float(distance))
    if d <= 1e-9 or vmax <= 0 or accel <= 0:
        return 0.0
    if d <= vmax * vmax / accel:
        return 2.0 * math.sqrt(d / accel)
    return d / vmax + vmax / accel


# ============================================================================
# 실기
# ============================================================================

class LinearLink:
    """LinearMotorNative.dll 을 감싸고 지령 위치를 추적한다."""

    def __init__(self, dll_path, mot_path, *, axis: int = 0, irq: int = 7,
                 velocity: float = 400.0, acceleration: float = 1600.0,
                 start_cmd: float = 0.0, min_cmd: float = 0.0,
                 max_cmd: float = 110.0):
        self.dll_path = os.path.abspath(str(dll_path))
        self.mot_path = os.path.abspath(str(mot_path))
        self.axis, self.irq = int(axis), int(irq)
        self.velocity = float(velocity)
        self.acceleration = float(acceleration)
        self.min_cmd, self.max_cmd = float(min_cmd), float(max_cmd)

        self._dll = None
        self._vendor = []
        self._nonblocking = False        # 패치된 DLL 이면 True
        self.opened = False
        self.simulated = False

        self._cmd = float(start_cmd)     # 추적 중인 현재 위치
        self._target = float(start_cmd)
        self._pending = None             # 이동 중에 들어온 최신 목표
        self._moving = False
        self._seg_from = float(start_cmd)   # 지금 조각의 출발 지령
        self._seg_to = float(start_cmd)     # 지금 조각의 도착 지령
        self._seg_t0 = 0.0                  # 조각 시작 시각
        self._seg_dur = 0.0                 # 조각 예상 소요 시간
        self._move_done_at = 0.0
        self._lock = threading.RLock()
        self._thread = None
        self.deferred = 0                # 이동 중이라 미뤄진 지령 수
        self.last_error = None

    # --------------------------------------------------------------- 준비
    def open(self) -> None:
        if os.name != "nt":
            raise LinearError("LinearMotorNative.dll 은 Windows 에서만 "
                              "로드할 수 있습니다.")
        for p in (self.dll_path, self.mot_path):
            if not os.path.isfile(p):
                raise LinearError("파일이 없습니다: %s" % p)
        d = os.path.dirname(self.dll_path)
        deps = (os.path.join(d, "EzBasicAxl.dll"), os.path.join(d, "AXL.dll"))
        missing = [p for p in deps if not os.path.isfile(p)]
        if missing:
            raise LinearError("Ajin 런타임 DLL 이 없습니다:\n" + "\n".join(missing))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
        try:
            #  AXL.dll 이 EzBasicAxl.dll 을 참조하므로 순서를 지킨다
            self._vendor = [ctypes.WinDLL(deps[0]), ctypes.WinDLL(deps[1])]
            self._dll = ctypes.WinDLL(self.dll_path)
        except OSError as e:
            raise LinearError(
                "Ajin 네이티브 라이브러리를 불러오지 못했습니다. 모든 DLL 이 "
                "x64 여야 하고, Ajin 모션 드라이버와 MSVC 2005 x64 런타임이 "
                "필요할 수 있습니다.\n경로: %s\n원인: %s" % (d, e))
        self._bind()
        self._check("LM_Initialize",
                    self._dll.LM_Initialize(os.fsencode(self.mot_path),
                                            self.axis, self.irq))
        self.opened = True

    def _bind(self) -> None:
        c_int, c_long, c_ulong, c_double = (ctypes.c_int, ctypes.c_long,
                                            ctypes.c_ulong, ctypes.c_double)
        d = self._dll
        d.LM_Initialize.argtypes = [ctypes.c_char_p, c_long, c_long]
        d.LM_Initialize.restype = c_int
        for name in ("LM_ServoOn", "LM_ServoOff", "LM_EmergencyStop",
                     "LM_ZeroPositionHere", "LM_Close"):
            fn = getattr(d, name, None)
            if fn is not None:
                fn.argtypes = []
                fn.restype = c_int
        for name in ("LM_MoveAbsolute", "LM_StartMoveAbsolute"):
            fn = getattr(d, name, None)
            if fn is not None:
                fn.argtypes = [c_double] * 4
                fn.restype = c_int
        self._nonblocking = hasattr(d, "LM_StartMoveAbsolute")
        d.LM_Stop.argtypes = [c_double]
        d.LM_Stop.restype = c_int
        for name, arg in (("LM_GetServoAlarm", ctypes.POINTER(c_ulong)),
                          ("LM_GetCommandPosition", ctypes.POINTER(c_double)),
                          ("LM_IsMoving", ctypes.POINTER(c_int))):
            fn = getattr(d, name, None)
            if fn is not None:
                fn.argtypes = [arg]
                fn.restype = c_int
        if hasattr(d, "LM_GetLimits"):
            d.LM_GetLimits.argtypes = [ctypes.POINTER(c_ulong),
                                       ctypes.POINTER(c_ulong)]
            d.LM_GetLimits.restype = c_int

    @staticmethod
    def _check(what: str, code: int) -> None:
        if code != 0:
            raise LinearError("%s 실패: code=%d (0x%08X)"
                              % (what, code, code & 0xFFFFFFFF))

    def servo_on(self) -> None:
        self._check("LM_ServoOn", self._dll.LM_ServoOn())

    def servo_off(self) -> None:
        try:
            self._check("LM_ServoOff", self._dll.LM_ServoOff())
        except LinearError as e:
            print("[리니어] %s" % e)

    def close(self) -> None:
        self.wait(timeout=2.0)
        if self._dll is not None and self.opened:
            try:
                self.servo_off()
                self._dll.LM_Close()
            except Exception:
                pass
        self.opened = False

    # --------------------------------------------------------------- 상태
    @property
    def position(self) -> float:
        """추적 중인 지령 위치(단위)."""
        with self._lock:
            return self._cmd

    @property
    def moving(self) -> bool:
        with self._lock:
            return self._moving

    def zero_here(self, value: float = 0.0) -> None:
        """현재 물리 위치를 value 로 선언한다 (추적값 재동기)."""
        with self._lock:
            self._cmd = self._target = float(value)
        try:
            self._dll.LM_ZeroPositionHere()
        except Exception:
            pass

    def servo_alarm(self) -> int:
        v = ctypes.c_ulong()
        try:
            self._dll.LM_GetServoAlarm(ctypes.byref(v))
        except Exception:
            return 0
        return int(v.value)

    # --------------------------------------------------------------- 이동
    def move_to(self, cmd: float, *, velocity=None, acceleration=None,
                allow_outside: bool = False) -> None:
        """절대 지령. 논블로킹.

        이동 중이면 최신 목표만 보관했다가 끝나는 즉시 이어 보낸다.

        allow_outside=True 면 랠리용 범위(min_cmd~max_cmd) 밖도 허용한다.
        시작·종료 위치가 그 범위 밖에 있을 때 쓴다.
        """
        c = (float(cmd) if allow_outside
             else max(self.min_cmd, min(self.max_cmd, float(cmd))))
        v = self.velocity if velocity is None else float(velocity)
        a = self.acceleration if acceleration is None else float(acceleration)
        with self._lock:
            if self._moving:
                self._pending = (c, v, a)
                self.deferred += 1
                return
            self._target = c
            self._moving = True
        self._thread = threading.Thread(target=self._run, args=(c, v, a),
                                        daemon=True, name="linear")
        self._thread.start()

    def _begin_segment(self, cmd, v, a) -> None:
        """지금 실행할 조각의 시작 상태를 기록한다 (실시간 위치 추정용)."""
        with self._lock:
            self._seg_from = self._cmd
            self._seg_to = float(cmd)
            self._seg_t0 = time.perf_counter()
            self._seg_dur = move_time_sec(self._seg_to - self._seg_from, v, a)

    def estimated_position(self) -> float:
        """지금 이 순간의 **추정 실제 위치**.

        엔코더가 없으므로, 조각의 경과 시간과 속도 프로파일로 어디쯤인지
        계산한다.  이동 중이 아니면 지령값 그대로다.
        """
        with self._lock:
            if not self._moving or self._seg_dur <= 1e-6:
                return self._cmd
            u = (time.perf_counter() - self._seg_t0) / self._seg_dur
            if u >= 1.0:
                return self._seg_to
            return self._seg_from + (self._seg_to - self._seg_from) \
                * profile_progress(u)

    def _run(self, cmd, v, a) -> None:
        while True:
            self._begin_segment(cmd, v, a)
            try:
                fn = (self._dll.LM_StartMoveAbsolute if self._nonblocking
                      else self._dll.LM_MoveAbsolute)
                self._check("LM_MoveAbsolute", fn(float(cmd), v, a, a))
                if self._nonblocking:
                    #  논블로킹 DLL 이면 우리가 시간만큼 기다려 준다
                    time.sleep(move_time_sec(cmd - self.position, v, a))
            except LinearError as e:
                self.last_error = str(e)
                print("[리니어] %s" % e)
            with self._lock:
                self._cmd = cmd
                self._move_done_at = time.perf_counter()
                nxt = self._pending
                self._pending = None
                if nxt is None:
                    self._moving = False
                    return
                cmd, v, a = nxt
                self._target = cmd

    def wait(self, timeout: float = 5.0) -> bool:
        t0 = time.perf_counter()
        while self.moving and time.perf_counter() - t0 < timeout:
            time.sleep(0.002)
        return not self.moving

    def stop(self, deceleration=None) -> None:
        try:
            self._dll.LM_Stop(float(self.acceleration if deceleration is None
                                    else deceleration))
        except Exception as e:
            print("[리니어] 정지 실패: %s" % e)

    def emergency_stop(self) -> None:
        try:
            self._dll.LM_EmergencyStop()
        except Exception as e:
            print("[리니어] 비상정지 실패: %s" % e)
        with self._lock:
            self._pending = None

    def eta_sec(self, cmd: float) -> float:
        return move_time_sec(float(cmd) - self.position,
                             self.velocity, self.acceleration)


# ============================================================================
# 시뮬레이터
# ============================================================================

class SimLinear(LinearLink):
    """하드웨어 없이 이동 시간만 흉내낸다."""

    def __init__(self, *, velocity=400.0, acceleration=1600.0,
                 start_cmd=0.0, min_cmd=0.0, max_cmd=110.0, realtime=True):
        object.__init__(self)
        self.dll_path = self.mot_path = "SIM"
        self.axis, self.irq = 0, 0
        self.velocity, self.acceleration = float(velocity), float(acceleration)
        self.min_cmd, self.max_cmd = float(min_cmd), float(max_cmd)
        self.opened = False
        self.simulated = True
        self.realtime = bool(realtime)
        self._nonblocking = True
        self._cmd = self._target = float(start_cmd)
        self._pending = None
        self._moving = False
        self._move_done_at = 0.0
        self._seg_from = self._seg_to = float(start_cmd)
        self._seg_t0 = self._seg_dur = 0.0
        self._lock = threading.RLock()
        self._thread = None
        self.deferred = 0
        self.last_error = None

    def open(self):
        self.opened = True
        print("[SIM] 가상 리니어축 열림 (지령 %.2f)" % self._cmd)

    def close(self):
        self.wait(timeout=2.0)
        self.opened = False
        print("[SIM] 가상 리니어축 닫힘")

    def servo_on(self):
        pass

    def servo_off(self):
        pass

    def servo_alarm(self):
        return 0

    def zero_here(self, value=0.0):
        with self._lock:
            self._cmd = self._target = float(value)

    def _run(self, cmd, v, a):
        while True:
            self._begin_segment(cmd, v, a)
            if self.realtime:
                time.sleep(move_time_sec(cmd - self._cmd, v, a))
            with self._lock:
                self._cmd = cmd
                self._move_done_at = time.perf_counter()
                nxt = self._pending
                self._pending = None
                if nxt is None:
                    self._moving = False
                    return
                cmd, v, a = nxt
                self._target = cmd

    def stop(self, deceleration=None):
        pass

    def emergency_stop(self):
        with self._lock:
            self._pending = None


def find_linear_files(root=None, max_depth: int = 5):
    """LinearMotorNative.dll 과 .mot 파일을 찾아 (dll, mot) 로 돌려준다.

    프로젝트 폴더 아래를 훑는다.  예선 프로젝트에서 옮겨 온 위치가
    조금 달라도 찾을 수 있게 하려는 것이다.  못 찾으면 (None, None).
    """
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    base_depth = root.rstrip(os.sep).count(os.sep)
    dll = mot = None
    for dirpath, dirs, files in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - base_depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs
                   if d not in (".git", "__pycache__", "obj", ".vs")]
        if dll is None and "LinearMotorNative.dll" in files:
            #  같은 폴더에 벤더 DLL 이 함께 있어야 로드된다
            if "AXL.dll" in files and "EzBasicAxl.dll" in files:
                dll = os.path.join(dirpath, "LinearMotorNative.dll")
                here = [f for f in files if f.lower().endswith(".mot")]
                if here:
                    mot = os.path.join(dirpath, here[0])
        if mot is None:
            for f in files:
                if f.lower().endswith(".mot"):
                    mot = os.path.join(dirpath, f)
                    break
        if dll and mot:
            break
    return dll, mot


def make_linear(dll_path=None, mot_path=None, *, simulate=False, **kw):
    if simulate or not dll_path:
        return SimLinear(**{k: v for k, v in kw.items()
                            if k in ("velocity", "acceleration", "start_cmd",
                                     "min_cmd", "max_cmd", "realtime")})
    return LinearLink(dll_path, mot_path,
                      **{k: v for k, v in kw.items() if k != "realtime"})


if __name__ == "__main__":
    ax = make_linear(simulate=True)
    ax.open()
    print("이동 시간 모델 (속도 400, 가속 1600)")
    for mm in (100, 200, 400, 1141):
        u = mm / 10.37
        print("  %5.0f mm -> %6.1f ms" % (mm, move_time_sec(u, 400, 1600) * 1000))
    t0 = time.perf_counter()
    ax.move_to(60)
    ax.move_to(20)                      # 이동 중 새 목표 -> 보관됐다가 이어짐
    ax.wait(timeout=5)
    print("최종 위치 %.2f, 미뤄진 지령 %d개, 걸린 시간 %.0f ms"
          % (ax.position, ax.deferred, (time.perf_counter() - t0) * 1000))
    ax.close()
