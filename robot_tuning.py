#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_tuning.py
===============
실기에서 만지게 되는 값만 따로 모아 파일로 저장/복원한다.

robot_config.py 는 "설계값"이고, 여기 있는 것은 "현장에서 맞추는 값"이다.
UI 의 [보정] 탭에서 바꾼 뒤 저장하면 settings/robot_tuning.json 에 남고,
다음 실행에 자동으로 반영된다.

부호 두 개가 특히 중요하다.  둘 중 하나만 뒤집혀도 로봇이 공과 반대쪽으로
간다.  실기에서 확인한 뒤 토글하면 된다.
"""

from __future__ import annotations

import dataclasses
import json
import os

import robot_config as cfg


SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "settings")
TUNING_PATH = os.path.join(SETTINGS_DIR, "robot_tuning.json")


#  키 -> (설명, 기본값)
FIELDS = {
    "arm_x_offset_mm": ("theta=0, 지령 0 에서 막대 끝(손목 축)의 X(mm)",
                        cfg.ARM.arm_x_offset_mm),
    "paddle_offset_mm": ("손목 축 -> 라켓 중심, 레일 방향 거리(mm)",
                         cfg.ARM.paddle_offset_mm),
    "paddle_perp_mm":   ("손목 축 -> 라켓 중심, 막대에 수직인 성분(mm)",
                         cfg.ARM.paddle_perp_mm),
    "hinge_z_mm":       ("힌지(모터 1·2) 높이, 상판 기준(mm)",
                         cfg.ARM.hinge_z_mm),
    "radius_mm":        ("막대 길이, 힌지 -> 손목(mm)", cfg.ARM.radius_mm),
    "arm_x_sign":      ("theta 음수(우측)가 X 의 + 이면 +1, 아니면 -1",
                        cfg.ARM.arm_x_sign),
    "linear_sign":     ("리니어 지령 + 가 X 의 + 이면 +1, 아니면 -1", 1.0),
    "id5_sign":        ("손목(ID5) 회전 방향. 반대로 조립했으면 -1",
                        cfg.ID5_SIGN),
    "hit_x_min_mm":    ("타격 폭 창 하한(mm)", cfg.HIT_X_MIN_MM),
    "hit_x_max_mm":    ("타격 폭 창 상한(mm)", cfg.HIT_X_MAX_MM),
    "swing_duration_sec": ("대기각 -> 타격각 소요 시간(s)",
                           cfg.SWING_DURATION_SEC),
    "swing_latency_sec":  ("지령 -> 실제 기동 지연(s)",
                           cfg.SWING_COMMAND_LATENCY_SEC),
    "ready_theta_deg": ("대기 자세 theta(deg)", cfg.READY_THETA_DEG),
    "swing_rest_deg":  ("스윙 대기각(후방, deg)", cfg.SWING_REST_DEG),
    "swing_front_deg": ("스윙 타격각(전방, deg)", cfg.SWING_FRONT_DEG),
    "decide_line_cm":  ("결정선. 끝선에서 이 거리를 넘으면 목적지 확정(cm)",
                        cfg.DECIDE_LINE_CM),
    "aim_z_offset_mm": ("조준 Z 보정(mm). 라켓이 낮게 가면 올린다",
                        cfg.AIM_Z_OFFSET_MM),
    "aim_x_offset_mm": ("조준 X 보정(mm)", cfg.AIM_X_OFFSET_MM),
    "hit_z_margin_mm": ("타격 높이 창의 블레이드 여유(mm)",
                        cfg.ARM.blade_radius_mm * cfg.ARM.blade_usable_frac),
}

#  리니어 배율의 크기(부호는 linear_sign 이 따로 관리)
LINEAR_MM_PER_UNIT_ABS = abs(cfg.ARM.linear_mm_per_unit)


def _num(v):
    """None 은 '자동'을 뜻하므로 그대로 둔다."""
    return None if v is None else float(v)


def defaults() -> dict:
    return {k: _num(v) for k, (_, v) in FIELDS.items()}


def load(path: str = TUNING_PATH) -> dict:
    """저장된 값을 읽는다. 없으면 기본값."""
    d = defaults()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k in d:
                if k in saved:
                    d[k] = _num(saved[k])
        except Exception as e:
            print("[보정] 파일을 읽지 못해 기본값을 씁니다: %s" % e)
    return d


def save(values: dict, path: str = TUNING_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {"_comment": {k: v[0] for k, v in FIELDS.items()}}
    out.update({k: _num(values[k]) for k in FIELDS if k in values})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def geometry_from(values: dict):
    """보정값을 반영한 ArmGeometry 를 만든다."""
    sign = 1.0 if float(values.get("linear_sign", 1.0)) >= 0 else -1.0
    return dataclasses.replace(
        cfg.ARM,
        arm_x_offset_mm=float(values["arm_x_offset_mm"]
                              if values["arm_x_offset_mm"] is not None
                              else cfg.ARM.arm_x_offset_mm),
        paddle_offset_mm=float(values.get("paddle_offset_mm",
                                          cfg.ARM.paddle_offset_mm)),
        paddle_perp_mm=float(values.get("paddle_perp_mm",
                                        cfg.ARM.paddle_perp_mm)),
        hinge_z_mm=float(values.get("hinge_z_mm", cfg.ARM.hinge_z_mm)),
        radius_mm=float(values.get("radius_mm", cfg.ARM.radius_mm)),
        arm_x_sign=1.0 if float(values["arm_x_sign"]) >= 0 else -1.0,
        linear_mm_per_unit=LINEAR_MM_PER_UNIT_ABS * sign,
    )


def apply_to_config(values: dict) -> None:
    """모듈 전역 설정에 반영한다 (스윙 시간, 타격 창, 손목 방향 등)."""
    cfg.ID5_SIGN = 1 if float(values.get("id5_sign", cfg.ID5_SIGN)) >= 0 else -1
    cfg.HIT_X_MIN_MM = _num(values["hit_x_min_mm"])
    cfg.HIT_X_MAX_MM = _num(values["hit_x_max_mm"])
    cfg.SWING_DURATION_SEC = float(values["swing_duration_sec"])
    cfg.SWING_COMMAND_LATENCY_SEC = float(values["swing_latency_sec"])
    cfg.READY_THETA_DEG = float(values["ready_theta_deg"])
    if "swing_rest_deg" in values:
        cfg.SWING_REST_DEG = float(values["swing_rest_deg"])
    if "swing_front_deg" in values:
        cfg.SWING_FRONT_DEG = float(values["swing_front_deg"])
    if "decide_line_cm" in values:
        cfg.DECIDE_LINE_CM = float(values["decide_line_cm"])
    if "aim_z_offset_mm" in values:
        cfg.AIM_Z_OFFSET_MM = float(values["aim_z_offset_mm"])
    if "aim_x_offset_mm" in values:
        cfg.AIM_X_OFFSET_MM = float(values["aim_x_offset_mm"])
    if "hit_z_margin_mm" in values:
        cfg.HIT_Z_BLADE_MARGIN_MM = float(values["hit_z_margin_mm"])


def describe(values: dict) -> str:
    def _s(v):
        return "자동" if v is None else "%+.0f" % v
    return ("X 오프셋 %+.1f/%+.1fmm  팔 %+.0f  리니어 %+.0f  손목 %+.0f  "
            "폭 창 %s~%s  스윙 %.0fms(+%.0fms)  "
            "스윙각 %+.0f~%+.0f°"
            % (values["arm_x_offset_mm"],
               values.get("paddle_offset_mm", cfg.ARM.paddle_offset_mm),
               values["arm_x_sign"],
               values["linear_sign"], values.get("id5_sign", cfg.ID5_SIGN),
               _s(values["hit_x_min_mm"]), _s(values["hit_x_max_mm"]),
               values["swing_duration_sec"] * 1000,
               values["swing_latency_sec"] * 1000,
               values.get("swing_rest_deg", cfg.SWING_REST_DEG),
               values.get("swing_front_deg", cfg.SWING_FRONT_DEG)))


if __name__ == "__main__":
    v = load()
    print("현재 보정값")
    for k, (desc, _) in FIELDS.items():
        val = v[k]
        txt = "      자동" if val is None else "%10.3f" % val
        print("  %-20s %s   %s" % (k, txt, desc))
    print("\n" + describe(v))
    g = geometry_from(v)
    print("\n반영된 기구: x_offset %+.1f, x_sign %+.0f, mm/unit %+.2f"
          % (g.arm_x_offset_mm, g.arm_x_sign, g.linear_mm_per_unit))
