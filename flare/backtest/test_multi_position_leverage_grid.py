"""
flare.backtest.test_multi_position_leverage_grid

4대 정예 코인 (BTC, ETH, SOL, XRP) 1/4 균등 분할 동시 운용에 대한
레버리지 수준별 [1.0x, 2.0x, 3.0x, 4.0x, 5.0x] 4개년 실전 복리 백테스트 성적표 (2021~2024)
- 초기 자본: 100만 원
- 투입 비중: 슬롯당 (총자산 / 4) * 80% (20% 현금 버퍼)
- 룰: SL -4.0% (SOL은 SL -6.0%) / No TP / 24시간 만기 종가 청산
- 수수료/슬리피지 실시간 100% 차감
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    print("=" * 125)
    print("🔬 [4대 정예 코인 1/4 분할 동시 운용] 레버리지 수준별(1x~5x) 실전 복리 백테스트 성적표 (2021~2024, 4개년)")
    print("   • 대상: BTC, ETH, SOL, XRP | 조건: 슬롯당 (자산/4)*80% 투입 (20% 현금 버퍼) | No TP | 24시간 만기 청산")
    print("=" * 125)
    
    header = "{:<12} | {:<8} | {:<18} | {:<16} | {:<14} | {:<20}"
    row = "{:<12} | {:>6}회 | {:>16} | {:>14.2f}% | {:>12.2f}% | {:<20}"
    print(header.format("레버리지 수준", "총 거래수", "4년 뒤 최종 잔고", "실제 복리수익률", "최대낙폭(MDD)", "평가 및 권장 여부"))
    print("-" * 125)
    
    for lev in [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        res = run_equal_weight_multi_position(symbols, data_dir, initial_capital=1_000_000.0, leverage=lev, allocation_ratio=0.80)
        bal_str = f"₩{res['final_balance']:,.0f}"
        
        eval_str = ""
        if lev == 1.0:
            eval_str = "주식급 극강 안정성 🛡️"
        elif lev == 2.0:
            eval_str = "안정적 3배 증식 💎"
        elif lev == 2.5:
            eval_str = "최적 밸런스 스윗스팟 👑"
        elif lev == 3.0:
            eval_str = "강력한 고수익 🚀"
        elif lev == 4.0:
            eval_str = "원금 6.5배 돌파! 🔥"
        elif lev == 5.0:
            eval_str = "변동성 드래그 시작 ⚠️"
            
        print(row.format(f"{lev:.1f}x", len(res["trades_df"]), bal_str, res["return_pct"], res["mdd"], eval_str))
    print("=" * 125)


if __name__ == "__main__":
    main()
