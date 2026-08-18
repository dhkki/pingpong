#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_profile.py
==============
지금 화면에 맞춰 놓은 설정을 **한 파일로** 저장하고 다시 불러온다.

무엇이 담기나
-------------
    검출   카메라별 HSV 범위, 크기 조건, 배경 차분 · ROI 제한 · 노출
    예측   타겟 Y, 횡 기울기 배수, 반발계수, 수평 감쇠, 회귀 구간
    로봇   보정값 전부(오프셋 · 부호 · 타격 창 · 스윙 각/시간), 움직일 축

무엇이 안 담기나
----------------
    카메라 캘리브레이션(intrinsics/extrinsics), 기본 자세 raw.
    이 둘은 기구와 설치에 묶인 값이라 따로 관리한다.

파일은 settings/profiles/ 아래에 JSON 으로 쌓인다.  현장에서 조명이나
공이 바뀔 때마다 이름을 달리해 저장해 두고 골라 쓰면 된다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "settings")
PROFILE_DIR = os.path.join(SETTINGS_DIR, "profiles")
LAST_PATH = os.path.join(SETTINGS_DIR, "last_profile.json")

VERSION = 1

#  예측기에서 저장할 항목
PRED_KEYS = ("restitution", "friction", "line_window_m", "min_points",
             "gap_sec", "auto_clear_sec", "bounce_z", "bounce_z_slack",
             "bounce_min_fall", "bounce_min_rise", "bounce_min_after",
             "bounce_min_before", "dir_rev_m", "pred_max_t")


def default_name() -> str:
    return "profile_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# 모으기
# ============================================================================

def collect(app) -> dict:
    """앱의 현재 상태를 사전으로 모은다."""
    import robot_config as cfg

    pred = app.fit.pred
    be = app.worker.backend if app.worker is not None else None

    data = {
        "_version": VERSION,
        "_saved_at": datetime.now().isoformat(timespec="seconds"),
        "detector": {
            "cam0": dict(app.params[0]),
            "cam1": dict(app.params[1]),
        },
        "view": {
            "auto_exposure": bool(app.var_autoexp.get()),
            "exposure": float(app.var_exp.get()),
            "show_roi": bool(app.var_showroi.get()),
            "show_cam": bool(app.var_showcam.get()),
        },
        "predictor": {
            "target_y": float(pred.target_y),
            "direction": int(pred.direction),
            "lat_gain": float(pred.gain),
            "cfg": {k: pred.cfg[k] for k in PRED_KEYS if k in pred.cfg},
        },
        "tuning": dict(app.tune),
        "robot": {
            "swing_velocity": int(cfg.SWING_VELOCITY),
            "swing_acceleration": int(cfg.SWING_ACCELERATION),
            "enable_linear": bool(be.enable_linear) if be else True,
            "enable_tilt": bool(be.enable_tilt) if be else True,
            "enable_swing": bool(be.enable_swing) if be else True,
        },
    }
    return data


# ============================================================================
# 적용하기
# ============================================================================

def apply(app, data: dict) -> list:
    """사전을 앱에 반영한다. 처리하지 못한 항목의 설명을 돌려준다."""
    import robot_config as cfg
    import robot_tuning as tuning

    skipped = []

    #  ---- 검출 ----
    det = data.get("detector") or {}
    for i, key in ((0, "cam0"), (1, "cam1")):
        src = det.get(key)
        if not src:
            continue
        app.params[i].update(src)
        for k, v in src.items():
            try:
                app.detector[i].set(k, v)
            except Exception:
                pass
    #  슬라이더/체크박스 표시 갱신
    try:
        src = app.params[0 if app.slider_target == 2 else app.slider_target]
        for k, var in app.sliders.items():
            if k in src:
                var.set(float(src[k]))
        app.var_mode.set(str(src.get("mode", "both")))
        app.var_bg.set(bool(src.get("use_bgsub", False)))
        app.var_async.set(bool(src.get("compensate_async", False)))
        app.var_roimask.set(bool(src.get("use_roi_mask", True)))
        app.var_roimargin.set(float(src.get("roi_mask_margin_px", 20)))
        app.on_opt()
        if hasattr(app, "_rebuild_roi_masks"):
            app._rebuild_roi_masks()
    except Exception as e:
        skipped.append("검출 표시 갱신: %s" % e)

    #  ---- 화면/노출 ----
    view = data.get("view") or {}
    try:
        if "auto_exposure" in view:
            app.var_autoexp.set(bool(view["auto_exposure"]))
        if "exposure" in view:
            app.var_exp.set(float(view["exposure"]))
        app.on_exposure()
        if "show_roi" in view:
            app.var_showroi.set(bool(view["show_roi"]))
        if "show_cam" in view:
            app.var_showcam.set(bool(view["show_cam"]))
    except Exception as e:
        skipped.append("노출/표시: %s" % e)

    #  ---- 예측 ----
    pr = data.get("predictor") or {}
    try:
        pred = app.fit.pred
        if "target_y" in pr:
            pred.set_target_y(float(pr["target_y"]))
            app.var_target_y.set(float(pr["target_y"]))
        if "direction" in pr:
            pred.direction = int(pr["direction"])
        if "lat_gain" in pr:
            pred.gain = float(pr["lat_gain"])
            app.lbl_gain.config(text="%.2f" % pred.gain)
        for k, v in (pr.get("cfg") or {}).items():
            pred.cfg[k] = v
        if "restitution" in (pr.get("cfg") or {}):
            app.var_restitution.set(float(pr["cfg"]["restitution"]))
        if "friction" in (pr.get("cfg") or {}):
            app.var_friction.set(float(pr["cfg"]["friction"]))
        if "dir_rev_m" in (pr.get("cfg") or {}):
            app.var_dir_rev.set(float(pr["cfg"]["dir_rev_m"]) * 100.0)
    except Exception as e:
        skipped.append("예측: %s" % e)

    #  ---- 보정값 ----
    tun = data.get("tuning") or {}
    if tun:
        try:
            base = tuning.defaults()
            base.update({k: (None if v is None else float(v))
                         for k, v in tun.items() if k in base})
            app.tune = base
            #  UI 위젯에 반영한 뒤 앱의 적용 경로를 그대로 태운다
            app.tune_vars["arm_x_offset_mm"].set(
                "%.1f" % base["arm_x_offset_mm"])
            app.var_armsign.set(base["arm_x_sign"] < 0)
            app.var_linsign.set(base["linear_sign"] < 0)
            app.var_id5sign.set(base.get("id5_sign", cfg.ID5_SIGN) < 0)
            app.var_hit_auto.set(base["hit_x_min_mm"] is None)
            if base["hit_x_min_mm"] is not None:
                app.var_hit_lo.set(base["hit_x_min_mm"])
            if base["hit_x_max_mm"] is not None:
                app.var_hit_hi.set(base["hit_x_max_mm"])
            app.var_swing_ms.set(base["swing_duration_sec"] * 1000)
            app.var_lat_ms.set(base["swing_latency_sec"] * 1000)
            app.var_ready_th.set(base["ready_theta_deg"])
            app.var_sw_rest.set(base.get("swing_rest_deg", cfg.SWING_REST_DEG))
            app.var_sw_front.set(base.get("swing_front_deg",
                                          cfg.SWING_FRONT_DEG))
            app.apply_swing_span()
            app.apply_tuning()
        except Exception as e:
            skipped.append("보정값: %s" % e)

    #  ---- 로봇 ----
    rb = data.get("robot") or {}
    try:
        if "swing_velocity" in rb:
            app.var_sw_vel.set(int(rb["swing_velocity"]))
        if "swing_acceleration" in rb:
            app.var_sw_acc.set(int(rb["swing_acceleration"]))
        app.apply_swing_profile()
        if app.worker is not None:
            app.var_ax_linear.set(bool(rb.get("enable_linear", True)))
            app.var_ax_tilt.set(bool(rb.get("enable_tilt", True)))
            app.var_ax_swing.set(bool(rb.get("enable_swing", True)))
            app.apply_axes()
    except Exception as e:
        skipped.append("로봇: %s" % e)

    return skipped


# ============================================================================
# 파일
# ============================================================================

def save(data: dict, path: str | None = None) -> str:
    if path is None:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        path = os.path.join(PROFILE_DIR, default_name())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    #  마지막으로 쓴 파일을 기억해 다음 실행에 자동으로 불러온다
    try:
        with open(LAST_PATH, "w", encoding="utf-8") as f:
            json.dump({"path": os.path.abspath(path)}, f, ensure_ascii=False)
    except Exception:
        pass
    return path


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def last_path() -> str | None:
    if not os.path.exists(LAST_PATH):
        return None
    try:
        with open(LAST_PATH, "r", encoding="utf-8") as f:
            p = json.load(f).get("path")
        return p if p and os.path.exists(p) else None
    except Exception:
        return None


def summary(data: dict) -> str:
    d0 = (data.get("detector") or {}).get("cam0", {})
    d1 = (data.get("detector") or {}).get("cam1", {})
    t = data.get("tuning") or {}
    return ("저장 %s\n"
            "  cam0  H %s~%s  S %s+  V %s+\n"
            "  cam1  H %s~%s  S %s+  V %s+\n"
            "  스윙 %.0f ms   타격 폭 X %.0f~%.0f mm"
            % (data.get("_saved_at", "?"),
               d0.get("h_lo"), d0.get("h_hi"), d0.get("s_lo"), d0.get("v_lo"),
               d1.get("h_lo"), d1.get("h_hi"), d1.get("s_lo"), d1.get("v_lo"),
               float(t.get("swing_duration_sec", 0)) * 1000,
               (t.get("hit_x_min_mm") if t.get("hit_x_min_mm") is not None
                else float("nan")),
               (t.get("hit_x_max_mm") if t.get("hit_x_max_mm") is not None
                else float("nan"))))
