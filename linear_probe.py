#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_probe.py
===============
리니어 축만 아주 천천히 왕복시켜 눈으로 확인하는 스크립트.

다이나믹셀은 건드리지 않는다.  포트도 열지 않는다.
GUI 없이 콘솔에 시작 / 출발 / 이동중 / 도착 / 종료만 찍는다.

경로
----
    시작 지점(지령 0, 테이블 왼쪽 사이드라인)
      -> 준비운동 오른쪽 끝 (기본 지령 85.6, 힌지 X 약 +125mm)
      -> 다시 시작 지점

아주 느리게 간다.  한 번에 쭉 가지 않고 짧게 여러 조각으로 나눠 보내며,
조각마다 잠시 멈춘다.  중간에 Ctrl+C 를 누르면 그 자리에서 정지한 뒤
시작 지점으로 되돌아온다.

    python linear_probe.py                 # 지령 0 -> 85.6 -> 0
    python linear_probe.py --to 60         # 목표를 직접 지정
    python linear_probe.py --speed 20      # 더 느리게 (기본 25)
    python linear_probe.py --dry           # 하드웨어 없이 절차만 확인

★ 켜기 전에 캐리지(힌지)를 테이블 왼쪽 사이드라인에 두어야 한다.
   DLL 은 프로그램을 켤 때의 자리를 지령 0 으로 잡는다.
"""

from __future__ import annotations

import argparse
import sys
import time

import robot_config as cfg
from linear_link import find_linear_files, make_linear

#  보정 파일(settings/robot_tuning.json)이 설정을 덮어쓴다.
#  본 프로그램과 같은 값을 쓰도록 여기서도 읽어 둔다.
try:
    import robot_tuning as _tuning
    _TUNE = _tuning.load()
    _tuning.apply_to_config(_TUNE)
    _GEO = _tuning.geometry_from(_TUNE)
except Exception as _e:                     # 파일이 없으면 설계값 그대로
    _TUNE, _GEO = None, cfg.ARM


def default_target() -> float:
    """준비운동이 실제로 가는 오른쪽 끝 지령."""
    try:
        from motion_control import MotionWorker, SimBackend
        from robot_kinematics import ArmKinematics
        w = MotionWorker(SimBackend(verbose=False),
                         kinematics=ArmKinematics(_GEO), armed=False,
                         verbose=False)
        return max(q for _, q in w.warmup_path())
    except Exception:
        return float(cfg.ARM.linear_max_cmd)


def hinge_mm(cmd: float) -> float:
    g = _GEO
    return cmd * g.linear_mm_per_unit + g.arm_x_offset_mm


def creep(ax, start: float, goal: float, *, step_cmd: float,
          dwell: float, velocity: float, accel: float) -> None:
    """짧은 조각으로 나눠 아주 천천히 이동한다."""
    d = goal - start
    n = max(1, int(abs(d) / max(0.1, step_cmd)) + 1)
    for i in range(1, n + 1):
        cmd = start + d * i / n
        ax.move_to(cmd, velocity=velocity, acceleration=accel,
                   allow_outside=True)
        ax.wait(timeout=15.0)
        print("  이동중  지령 %6.2f   힌지 X %+7.1f mm   (%d/%d)"
              % (cmd, hinge_mm(cmd), i, n))
        time.sleep(dwell)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="리니어 축만 천천히 왕복")
    ap.add_argument("--to", type=float, default=None,
                    help="목표 지령 (기본: 준비운동 오른쪽 끝)")
    ap.add_argument("--speed", type=float, default=25.0,
                    help="이동 속도 (기본 25, 랠리용은 400)")
    ap.add_argument("--accel", type=float, default=100.0)
    ap.add_argument("--step", type=float, default=4.0,
                    help="한 조각의 지령 크기 (작을수록 더 잘게)")
    ap.add_argument("--dwell", type=float, default=0.4,
                    help="조각마다 멈추는 시간(s)")
    ap.add_argument("--dry", action="store_true",
                    help="하드웨어 없이 절차만 확인")
    args = ap.parse_args(argv)

    goal = default_target() if args.to is None else float(args.to)
    #  ★ 어떤 경우에도 랠리 상한을 넘지 않는다 ★
    hard = float(_GEO.linear_max_cmd)
    if goal > hard:
        print("  목표 %.2f 가 상한 %.2f 를 넘어 잘랐습니다." % (goal, hard))
        goal = hard
    start = float(cfg.LINEAR_START_CMD_AT_LEFT_END)

    print("=" * 62)
    print(" 리니어 축 확인 — 다이나믹셀은 건드리지 않습니다")
    print("=" * 62)
    print("  시작 지령 %6.2f   힌지 X %+7.1f mm" % (start, hinge_mm(start)))
    print("  목표 지령 %6.2f   힌지 X %+7.1f mm" % (goal, hinge_mm(goal)))
    print("  이동량   %6.2f 단위 = %.0f mm"
          % (goal - start, (goal - start) * cfg.ARM.linear_mm_per_unit))
    print("  속도 %.0f / 가속 %.0f  (랠리용은 %.0f / %.0f)"
          % (args.speed, args.accel, cfg.LINEAR_VELOCITY,
             cfg.LINEAR_ACCELERATION))
    print("  조각 %.1f 단위마다 %.1f 초 정지" % (args.step, args.dwell))
    print()
    print("  ★ 캐리지(힌지)가 테이블 왼쪽 사이드라인에 있어야 합니다.")
    try:
        input("  준비되면 Enter, 취소하려면 Ctrl+C : ")
    except (KeyboardInterrupt, EOFError):
        print("\n취소했습니다.")
        return 1

    dll, mot = cfg.LINEAR_DLL_PATH, cfg.LINEAR_MOT_PATH
    if dll is None or mot is None:
        fd, fm = find_linear_files()
        dll, mot = dll or fd, mot or fm
    if args.dry or not dll:
        if not args.dry:
            print("  리니어 DLL 을 찾지 못해 가상으로 돌립니다.")
        ax = make_linear(simulate=True, start_cmd=start,
                         min_cmd=-9999, max_cmd=hard)
    else:
        print("  DLL: %s" % dll)
        ax = make_linear(dll, mot, start_cmd=start,
                         min_cmd=-9999, max_cmd=hard)

    print("\n시작")
    ax.open()
    try:
        ax.servo_on()
    except Exception as e:
        print("  서보 ON 실패: %s" % e)

    try:
        print("출발  ->  오른쪽 (지령 %.2f)" % goal)
        creep(ax, start, goal, step_cmd=args.step, dwell=args.dwell,
              velocity=args.speed, accel=args.accel)
        print("도착  지령 %.2f   힌지 X %+.1f mm" % (goal, hinge_mm(goal)))
        time.sleep(1.0)

        print("\n출발  ->  시작 지점 (지령 %.2f)" % start)
        creep(ax, goal, start, step_cmd=args.step, dwell=args.dwell,
              velocity=args.speed, accel=args.accel)
        print("도착  지령 %.2f   힌지 X %+.1f mm" % (start, hinge_mm(start)))

    except KeyboardInterrupt:
        print("\n  중단 요청 — 시작 지점으로 되돌립니다.")
        try:
            ax.wait(timeout=15.0)
            creep(ax, ax.position, start, step_cmd=args.step,
                  dwell=args.dwell, velocity=args.speed, accel=args.accel)
        except Exception as e:
            print("  복귀 실패: %s" % e)
    except Exception as e:
        print("  오류: %s" % e)
    finally:
        try:
            ax.wait(timeout=15.0)
            ax.close()
        except Exception:
            pass
        print("\n종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
