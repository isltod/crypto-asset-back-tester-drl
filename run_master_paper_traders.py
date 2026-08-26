"""
run_master_paper_traders.py
- 3대 실시간 페이퍼 트레이딩 모델 통합 오케스트레이터 마스터 러너
  1) [🌟 RADE 표준] STANDARD_GOLDEN (3.0x 레버리지, MDD 16%, CASH 모드)
  2) [💥 작은 몬스터] MONSTER_MINI (TF 4% x MR 12% x 80% 숏, 100.0x 레버리지, 수익률 22.8배)
  3) [🏰 8:2 앙상블] ENSEMBLE_82 (RADE 표준 80% + FLARE 5x 20% 분기 리밸런싱)
- 매 정시(00:05) Cron에서 이 스크립트 하나만 실행하면 3개 모델이 1초 만에 격리 실행됩니다.
"""

import sys
import logging
from datetime import datetime, timezone
import pandas as pd

# 콘솔 UTF-8 출력 강제 (Windows & Linux 공통)
sys.stdout.reconfigure(encoding="utf-8")

from rade.live.paper_trader import PaperTrader
from flare.live.ensemble_paper_trader import EnsemblePaperTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MasterRunner")


def run_all():
    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now.astimezone(timezone(pd.Timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    print(f"\n{'='*70}")
    print(f" 🚀 [3대 모델 통합 페이퍼 트레이딩 사이클 가동] - {kst_now}")
    print(f"{'='*70}")

    # 1. [🌟 RADE 공식 표준 모델]
    try:
        print("\n[1/3] 🌟 RADE 공식 표준 모델 (STANDARD_GOLDEN) 실행 중...")
        trader_std = PaperTrader(
            symbol="BTCUSDT",
            initial_capital=10000.0,
            preset_name="STANDARD_GOLDEN",
            instance_id="standard"
        )
        trader_std.execute_cycle()
        print("  └─► ✅ RADE 표준 모델 사이클 완료")
    except Exception as e:
        logger.error(f"RADE 표준 실행 실패: {e}")

    # 2. [💥 작은 몬스터 실전 공격형 모델]
    try:
        print("\n[2/3] 💥 작은 몬스터 실전 공격형 (MONSTER_MINI) 실행 중...")
        trader_mini = PaperTrader(
            symbol="BTCUSDT",
            initial_capital=10000.0,
            preset_name="MONSTER_MINI",
            instance_id="monster_mini"
        )
        trader_mini.execute_cycle()
        print("  └─► ✅ 작은 몬스터 모델 사이클 완료")
    except Exception as e:
        logger.error(f"작은 몬스터 실행 실패: {e}")

    # 3. [🏰 8:2 앙상블 리밸런싱 포트폴리오]
    try:
        print("\n[3/3] 🏰 8:2 앙상블 포트폴리오 (RADE 80% + FLARE 20%) 실행 중...")
        ensemble = EnsemblePaperTrader(
            initial_capital=10000.0,
            rade_ratio=0.80,
            flare_ratio=0.20,
            instance_id="ensemble_82"
        )
        ensemble.execute_cycle()
        print("  └─► ✅ 8:2 앙상블 포트폴리오 사이클 완료")
    except Exception as e:
        logger.error(f"8:2 앙상블 실행 실패: {e}")

    print(f"\n{'='*70}")
    print(f" ✨ 3대 모델 실시간 페이퍼 트레이딩 사이클 성공적으로 완료!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_all()
