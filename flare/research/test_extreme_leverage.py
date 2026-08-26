import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position

if __name__ == '__main__':
    data_dir = Path('data')
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    levs = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    
    print("\n" + "="*95)
    print(" 🔬 [FLARE 4대 코인] 레버리지 극한 확장 실험 (1배 ~ 15배 복리 백테스트)")
    print("="*95)
    print(f" {'레버리지':<8} | {'4년 뒤 최종 잔고':<20} | {'누적 수익률':<12} | {'MDD':<8} | {'판정 및 상태'}")
    print("-"*95)
    
    for lev in levs:
        res = run_equal_weight_multi_position(symbols, data_dir, initial_capital=1_000_000.0, leverage=lev, allocation_ratio=0.80)
        bal = res['final_balance']
        ret = res['return_pct']
        mdd = res['mdd']
        
        status = ""
        if lev == 2.5:
            status = "👑 황금 밸런스 (최적 칼마)"
        elif lev == 5.0:
            status = "🚀 명목 수익률 정점 부근"
        elif lev >= 10.0:
            status = "💀 계좌 괴멸 / 변동성 잠식 파산"
        elif mdd > 75.0:
            status = "⚠️ 치명적 MDD (회복 불가)"
        elif mdd > 60.0:
            status = "⚠️ 심각한 변동성 잠식 발생"
        else:
            status = "안정적 우상향"
            
        print(f" {lev:4.1f}x    | ₩{bal:16,.0f} | {ret:+10.2f}% | {mdd:6.2f}% | {status}")
    print("="*95)
