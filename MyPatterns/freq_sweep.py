#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
freq_sweep.py — 调制频率扫描实验
=================================
发射一个焦点并做幅度调制,频率从 10 Hz 逐步升到 1000 Hz,每个频率持续 2 秒,
让你对比手掌体感随频率的变化。

频率序列:
  10, 20, 30, 40   (细扫低频段:感受"拍打"区)
  50, 100, 150, 200, ... 1000   (50 起步,每次 +50)

用法:
  py -3.13 MyPatterns/freq_sweep.py --port COM7
  py -3.13 MyPatterns/freq_sweep.py --port COM7 --dist 0.12 --duty 0.7 --hold 2
每换一个频率终端会打印提示;全部结束自动关发射。
中途 Ctrl+C 可随时中止(会自动停止发射)。
"""

import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

import numpy as np

from my_patterns import Board, N_EMITTERS, phases_focus


def build_freq_list():
    """10,20,30,40,然后 50,100,150,...,1000"""
    freqs = [10, 20, 30, 40]
    freqs += list(range(50, 1001, 50))
    return freqs


def modulated_hold(board, phases, freq, duty, hold_s):
    """单频调制发射 hold_s 秒:预装图案/关闭两图,循环 commit 互换。"""
    board.send(phases)                      # 图案入前台
    board.send(np.full(N_EMITTERS, np.nan)) # 关闭图入前台(此刻静默)
    t_on = duty / freq
    t_off = (1.0 - duty) / freq
    t_end = time.perf_counter() + hold_s
    while time.perf_counter() < t_end:
        board.send_commit()                 # 开
        target = time.perf_counter() + t_on
        while time.perf_counter() < target:
            pass
        board.send_commit()                 # 关
        target = time.perf_counter() + t_off
        while time.perf_counter() < target:
            pass
    board.off()


def main():
    p = argparse.ArgumentParser(description="调制频率扫描:10→40 细扫,50→1000 步进50,每档2秒")
    p.add_argument("--port", default="COM7")
    p.add_argument("--dist", type=float, default=0.12, help="焦点高度 [m]")
    p.add_argument("--duty", type=float, default=0.7, help="占空比 0.1~0.9")
    p.add_argument("--hold", type=float, default=2.0, help="每个频率持续 [s]")
    p.add_argument("--gap", type=float, default=1.0, help="两档之间静音间隔 [s]")
    args = p.parse_args()

    freqs = build_freq_list()
    print(f"频率扫描序列({len(freqs)} 档): {freqs[0]}→{freqs[-1]} Hz, "
          f"每档 {args.hold}s,焦点 {args.dist*100:.0f}cm,占空比 {args.duty:.0%}")
    print("手掌放在阵面上方约 %.0f cm,感受每个频率的质感差异(嗡/拍/尖)\n" % (args.dist * 100))

    phases = phases_focus((0.0, args.dist, 0.0))
    with Board(args.port) as b:
        try:
            for i, f in enumerate(freqs, 1):
                print(f"[{i:2d}/{len(freqs)}] >>> {f:4d} Hz  ({args.hold:.0f} 秒)  记住这个感觉!", flush=True)
                modulated_hold(b, phases, f, args.duty, args.hold)
                if i < len(freqs):
                    time.sleep(args.gap)
            print("\n扫描完成,发射已关闭")
        except KeyboardInterrupt:
            print("\n用户中止")
        finally:
            b.off()


if __name__ == "__main__":
    main()
