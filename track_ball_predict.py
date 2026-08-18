#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_ball_predict.py
=====================
기존 track_ball.py 를 **한 줄도 고치지 않고** 궤적 예측 창을 하나 더 띄우는
실행 래퍼.

  track_ball.TrackerApp 을 상속해
    - 매 프레임 처리 후, 새로 확정된 궤적 샘플만 예측기에 넘기고
    - 예측 창을 갱신한다
  두 가지만 덧붙인다. 기존 파이프라인(검출·삼각측량·칼만·3D 뷰)에는
  아무 영향도 주지 않는다.

실행
----
  python track_ball_predict.py                 # 원본과 동일한 인자 사용 가능
  python track_ball_predict.py --sim
  python track_ball_predict.py --cam0 0 --cam1 1

세 파일을 같은 폴더에 두면 된다:
  track_ball.py / trajectory_predictor.py / track_ball_predict.py

예측 창 조작
------------
  타겟 Y 슬라이더 : 통과 시각·위치를 계산할 라인
  횡(X) 기울기 배수 : 실제 낙하점이 예측보다 안쪽이면 1보다 작게
  방향 반전 : Y 감소 방향 <-> 증가 방향
  [ , ]  = 배수 조정      c = 지우기
"""

import sys
import tkinter as tk
from tkinter import messagebox

import track_ball as tb
from trajectory_predictor import PredictorWindow


# 예측기 설정을 여기서 덮어쓸 수 있다 (None 이면 기본값).
# 예) PREDICTOR_CFG = {"direction": -1, "target_y": 0.0, "lat_gain": 0.9}
PREDICTOR_CFG = None


class PredictiveTrackerApp(tb.TrackerApp):
    """원본 앱 + 예측 창. 원본 메서드는 super() 로 그대로 실행한다."""

    def __init__(self, args):
        self.predictor = None          # super().__init__ 이 _tick 을 부르므로 먼저
        self._fed_t = None
        super().__init__(args)
        self.predictor = PredictorWindow(self, self.geo, PREDICTOR_CFG,
                                         record_dir=tb.RECORD_DIR)
        self._status("궤적 예측 창을 함께 실행합니다. "
                     "(창을 닫아도 본 추적에는 영향 없음)")

    # -------------------------------------------------- 프레임 처리 후 후킹
    def _process(self):
        super()._process()
        pw = getattr(self, "predictor", None)
        if pw is None or not pw.alive:
            return
        # trail 에는 '실측이 확정된' (t, pos) 만 쌓인다. 새 샘플만 넘긴다.
        if self.trail:
            t, p = self.trail[-1]
            if self._fed_t is None or t > self._fed_t + 1e-9:
                self._fed_t = t
                pw.push(t, p)
        pw.render()

    # ------------------------------------------------------ 상태 동기화만
    def clear_trail(self):
        super().clear_trail()
        pw = getattr(self, "predictor", None)
        if pw is not None and pw.alive:
            pw.pred.clear()
        self._fed_t = None

    def on_close(self):
        pw = getattr(self, "predictor", None)
        if pw is not None and pw.alive:
            pw.close()
        super().on_close()


def main(argv=None):
    args = tb.parse_args(argv)
    tb.print_env_banner()
    try:
        app = PredictiveTrackerApp(args)
    except FileNotFoundError as e:
        print("[!] 설정 파일이 없습니다:\n%s" % e)
        try:
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("설정 없음",
                                 "%s\n\nsetup_extrinsics.py 를 먼저 실행하세요."
                                 % e)
            r.destroy()
        except Exception:
            pass
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())