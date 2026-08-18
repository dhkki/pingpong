#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
home_pose.py
============
기본 자세(원점 raw) 저장소.

무엇을 저장하는가
-----------------
각 모터의 **기준 자세에서의 raw 값**이다.  이후 모든 지령은 이 값을 원점으로
삼은 절대 위치로 계산된다.

    ID 1, 2  막대가 수직(theta = 0)일 때의 raw.  두 모터는 마주보아 있어
             theta 를 만들 때 서로 반대로 움직인다.
    ID 5     라켓 면이 기준 각도일 때의 raw.
    ID 10    스윙 각도 0, 즉 라켓이 라켓 평면에 오는 순간의 raw.

왜 다시 잡아야 하는가
---------------------
모터를 교체하면 내부 카운터의 원점이 완전히 달라진다.  이전 값으로 지령을
보내면 팔이 엉뚱한 자세로 움직이므로, 교체 후에는 반드시 다시 잡아야 한다.
(교체하지 않았더라도 하드웨어 리셋이나 모드 변경 뒤에는 확인하는 편이 좋다.)

옛 형식도 읽는다
----------------
예선 프로젝트의 postures.json 형식({"기본자세": {"positions": {...}}})도
그대로 읽을 수 있다.  새로 저장할 때는 이 파일의 형식으로 쓴다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from dynamixel_link import RAW_PER_REV, raw_to_deg

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "settings")
HOME_PATH = os.path.join(SETTINGS_DIR, "home_pose.json")

#  옛 형식도 찾아본다
LEGACY_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "postures.json"),
    os.path.join(SETTINGS_DIR, "postures.json"),
)

DEFAULT_IDS = (1, 2, 5, 10)
ROLE = {1: "틸트 A", 2: "틸트 B", 5: "라켓 면", 10: "스윙(원판)"}

#  두 틸트 모터는 마주보아 있다.  기준 자세에서 둘의 raw 합/차가 매번
#  같아야 할 이유는 없지만, 한쪽만 크게 튀면 잘못 잡았을 가능성이 높다.
PAIR_IDS = (1, 2)


# ============================================================================
# 저장 / 복원
# ============================================================================

def save(raw: dict, *, path: str = HOME_PATH, port: str = "",
         baudrate: int = 0, note: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "_comment": ("기준 자세(막대 수직 · 라켓 면 기준 · 스윙 각도 0)에서 "
                     "읽은 raw 값. 모터를 교체하면 다시 잡아야 한다."),
        "raw": {str(int(k)): int(v) for k, v in raw.items()},
        "roles": {str(k): ROLE.get(k, "") for k in raw},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "port": port,
        "baudrate": int(baudrate),
        "note": note,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load(path: str = HOME_PATH) -> dict | None:
    """{id: raw} 또는 None. 새 형식을 먼저 보고, 없으면 옛 형식을 찾는다."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {int(k): int(v) for k, v in d["raw"].items()}
        except Exception as e:
            print("[기본자세] 파일을 읽지 못했습니다: %s" % e)
    for p in LEGACY_PATHS:
        legacy = _load_legacy(p)
        if legacy:
            print("[기본자세] 옛 형식에서 읽었습니다: %s" % p)
            return legacy
    return None


def _load_legacy(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if "raw" in d:
        try:
            return {int(k): int(v) for k, v in d["raw"].items()}
        except Exception:
            return None
    for _, body in d.items():                   # {"기본자세": {"positions": ...}}
        if isinstance(body, dict) and "positions" in body:
            try:
                return {int(k): int(v) for k, v in body["positions"].items()}
            except Exception:
                return None
    return None


def info(path: str = HOME_PATH) -> str:
    if not os.path.exists(path):
        return "저장된 기본 자세가 없습니다."
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return "읽기 실패: %s" % e
    raw = {int(k): int(v) for k, v in d.get("raw", {}).items()}
    return ("저장 시각 %s   포트 %s\n  %s"
            % (d.get("saved_at", "?"), d.get("port", "?"),
               "   ".join("ID%d %d" % (k, raw[k]) for k in sorted(raw))))


# ============================================================================
# 검증
# ============================================================================

def validate(raw: dict, ids=DEFAULT_IDS, *, previous: dict | None = None):
    """저장 전에 확인할 것들. 경고 문자열 목록을 돌려준다."""
    warn = []
    missing = [i for i in ids if i not in raw or raw[i] is None]
    if missing:
        warn.append("응답이 없는 모터가 있습니다: %s"
                    % ", ".join("ID %d" % i for i in missing))
    for i, v in raw.items():
        if v is None:
            continue
        if abs(int(v)) > 6 * RAW_PER_REV:
            warn.append("ID %d 의 raw 가 비정상적으로 큽니다 (%d = %.0f회전). "
                        "확장 위치 모드에서 여러 바퀴 감긴 상태일 수 있습니다."
                        % (i, v, v / RAW_PER_REV))
    if previous:
        for i in ids:
            if i in raw and i in previous and raw[i] is not None:
                d = abs(int(raw[i]) - int(previous[i]))
                if d > RAW_PER_REV // 4:        # 90도 이상 차이
                    warn.append("ID %d 가 이전 저장값과 %d raw (%.0f°) "
                                "다릅니다. 모터 교체 후라면 정상입니다."
                                % (i, d, raw_to_deg(d)))
    return warn


def describe(raw: dict) -> str:
    lines = []
    for i in sorted(raw):
        v = raw[i]
        lines.append("  ID %-3d %-12s raw %+7s   (%s)"
                     % (i, ROLE.get(i, ""),
                        "----" if v is None else "%d" % v,
                        "응답 없음" if v is None
                        else "%.1f°" % raw_to_deg(v)))
    return "\n".join(lines)


if __name__ == "__main__":
    print(info())
    r = load()
    if r:
        print("\n불러온 기본 자세")
        print(describe(r))
        w = validate(r)
        print("\n" + ("\n".join("! " + x for x in w) if w
                      else "[OK] 뚜렷한 문제는 없습니다."))
