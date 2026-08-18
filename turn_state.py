#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
turn_state.py
=============
지금 공을 칠 차례가 사람인지 로봇인지 판정한다.

원리
----
네트를 기준선으로 두고 공이 어느 코트에 있는지만 본다.

    공이 사람 코트(Y > 기준선)  ->  다음에 칠 사람은 **사람**
    공이 로봇 코트(Y < 기준선)  ->  다음에 칠 사람은 **로봇**

서브는 사람이 넣으므로 시작 상태는 사람이다.  공이 안 보이는 동안에는
마지막 상태를 그대로 유지한다.  경계에서 덜덜 떨지 않도록 히스테리시스를
두어, 기준선을 일정 거리 이상 넘어야 상태가 바뀐다.

왜 필요한가
-----------
1. 화면에 지금 차례를 보여 주기 위해.
2. 로봇이 친 뒤 나가는 공이 예측기에 들어가 트랙을 만드는 것을 막기 위해.
   로봇이 스윙하면 차단이 걸리고, 공이 사람 코트로 넘어가면 풀린다.
"""

from __future__ import annotations

import time

HUMAN = "human"
ROBOT = "robot"

#  기준선을 이만큼 확실히 넘어야 상태를 바꾼다(m).  경계 잡음 방지.
HYSTERESIS_M = 0.10


class TurnState:

    def __init__(self, line_m: float, *, verbose: bool = True):
        self.line_m = float(line_m)
        self.verbose = bool(verbose)
        self.who = HUMAN            # 서브는 사람부터
        self.changed_at = time.perf_counter()
        self.last_y = None
        self.last_seen = 0.0

        #  로봇이 친 뒤 공이 넘어갈 때까지 예측기 입력을 막는다
        self.blocked = False
        self._blocked_at = 0.0

    # ------------------------------------------------------------------
    def set_line(self, line_m: float) -> None:
        self.line_m = float(line_m)

    def note_swing(self) -> None:
        """로봇이 스윙했다. 공이 넘어갈 때까지 입력을 막는다."""
        self.blocked = True
        self._blocked_at = time.perf_counter()
        self._set(ROBOT, "스윙")      # 아직 로봇 코트에 있다

    def update(self, y_m: float | None, now: float | None = None) -> str:
        """공의 진행축 위치로 차례를 갱신한다. 현재 차례를 돌려준다."""
        now = time.perf_counter() if now is None else now
        if y_m is None:
            return self.who
        self.last_y = float(y_m)
        self.last_seen = now
        if y_m > self.line_m + HYSTERESIS_M:
            self._set(HUMAN, "사람 코트")
            self.blocked = False          # 넘어갔으니 차단 해제
        elif y_m < self.line_m - HYSTERESIS_M:
            self._set(ROBOT, "로봇 코트")
        return self.who

    def tick(self, now: float | None = None) -> None:
        """공이 안 잡히는 동안의 안전망.

        차단이 걸린 뒤 공이 오래 안 보이면, 넘어간 것으로 보고 푼다.
        (라켓에 가렸거나 시야 밖으로 나간 경우)
        """
        import robot_config as cfg
        if not self.blocked:
            return
        now = time.perf_counter() if now is None else now
        if now - max(self.last_seen, self._blocked_at) \
                > cfg.BLOCK_RELEASE_ON_LOST_SEC:
            self.blocked = False
            self._set(HUMAN, "공을 놓쳐 해제")

    def should_feed(self) -> bool:
        """예측기에 샘플을 넣어도 되는가."""
        import robot_config as cfg
        if not cfg.BLOCK_UNTIL_CROSSED:
            return True
        return not self.blocked

    # ------------------------------------------------------------------
    def _set(self, who: str, why: str) -> None:
        if who == self.who:
            return
        self.who = who
        self.changed_at = time.perf_counter()
        if self.verbose:
            print("[차례] %s (%s)" % ("사람" if who == HUMAN else "로봇", why))

    def label(self) -> str:
        return "사람 차례" if self.who == HUMAN else "로봇 차례"


if __name__ == "__main__":
    import robot_config as cfg
    t = TurnState(1.37)
    print("시작:", t.label())
    for y in (2.4, 2.0, 1.5, 1.2, 0.6, 0.2):
        t.update(y)
        print("  공 Y %.2f -> %s  (입력 %s)"
              % (y, t.label(), "허용" if t.should_feed() else "차단"))
    print("\n로봇 스윙")
    t.note_swing()
    for y in (0.4, 0.9, 1.3, 1.6, 2.1):
        t.update(y)
        print("  공 Y %.2f -> %s  (입력 %s)"
              % (y, t.label(), "허용" if t.should_feed() else "차단"))
