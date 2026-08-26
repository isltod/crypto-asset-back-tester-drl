"""
flare.backtest.test_sol_tuning

SOLUSDT에 대한 손절폭(SL) [4%, 6%, 8%, 10%] 최적화 백테스트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multicoin_swing import run_symbol_swing


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    
    print("=" * 85)
    print("🔬 [솔라나(SOLUSDT) 변동성 맞춤 손절폭(SL) 정밀 튜닝 (2021~2024, 4개년)]")
    print("=" * 85)
    
    header = "{:<20} | {:<8} | {:<8} | {:<14} | {:<8} | {:<10}"
    row = "{:<20} | {:>6}회 | {:>7.1f}% | {:>13.2f}% | {:>8.2f} | {:>9.2f}%"
    print(header.format("솔라나 SL 손절폭", "거래수", "승률", "1배수 누적수익률", "손익비(PF)", "최대낙폭(MDD)"))
    print("-" * 85)
    
    for sl in [4.0, 6.0, 8.0, 10.0]:
        res = run_symbol_swing("SOLUSDT", data_dir, threshold=-0.00010, sl_pct=sl)
        print(row.format(f"SL -{sl:.1f}%", res["trades"], res["win_rate"], res["cum_ret"], res["pf"], res["mdd"]))
    print("=" * 85)


if __name__ == "__main__":
    main()
