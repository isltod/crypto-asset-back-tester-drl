"""
RADE Walk-Forward Optimization (전진 분석 및 과적합 제거 모듈)
- 6개월 In-Sample(IS) 파라미터 최적화 -> 1개월 Out-of-Sample(OOS) 실전 검증
- 1개월 단위로 슬라이딩하며 OOS 거래를 연결하여 과적합 없는 진짜 기대 성능 산출
"""
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from rade.backtest.simulator import BacktestSimulator
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine


class WalkForwardOptimizer:
    """Walk-Forward 분석 및 파라미터 최적화기"""

    def __init__(
        self,
        train_window_bars: int = 4320,  # 6개월 (약 4,320시간)
        test_window_bars: int = 720,    # 1개월 (약 720시간)
        step_bars: int = 720,           # 1개월 전진
    ):
        self.train_window_bars = train_window_bars
        self.test_window_bars = test_window_bars
        self.step_bars = step_bars

    def run_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 구간에 걸쳐 슬라이딩 윈도우 Walk-Forward 최적화 실행
        """
        total_bars = len(df)
        all_oos_trades: List[Dict[str, Any]] = []
        window_reports: List[Dict[str, Any]] = []

        # 탐색할 주요 파라미터 그리드 (실행 속도와 다양성 균형)
        param_grid = {
            "mr_sl_atr": [1.0, 1.2, 1.5],
            "mr_rsi_low": [30.0, 35.0],
            "tf_sl_atr": [1.2, 1.5, 2.0],
            "tf_trail_atr": [2.0, 2.5, 3.0],
            "tf_adx_thresh": [20.0, 25.0],
        }

        # 파라미터 조합 생성
        keys, values = zip(*param_grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"[Walk-Forward] 총 {len(combos)}개 파라미터 조합 탐색")

        start_idx = 0
        window_id = 1

        while (start_idx + self.train_window_bars + self.test_window_bars) <= total_bars:
            train_end = start_idx + self.train_window_bars
            test_end = train_end + self.test_window_bars

            df_train = df.iloc[start_idx:train_end].reset_index(drop=True)
            df_test = df.iloc[train_end:test_end].reset_index(drop=True)

            train_start_date = str(df_train['datetime'].iloc[0])[:10]
            train_end_date = str(df_train['datetime'].iloc[-1])[:10]
            test_start_date = str(df_test['datetime'].iloc[0])[:10]
            test_end_date = str(df_test['datetime'].iloc[-1])[:10]

            print(f"\n--- [Window {window_id}] Train: {train_start_date}~{train_end_date} | Test: {test_start_date}~{test_end_date} ---")

            # 1. In-Sample 그리드 서치
            best_score = -999.0
            best_params = combos[0]

            for params in combos:
                sim = BacktestSimulator(initial_capital=10000.0)
                sim.mean_revert_engine.sl_atr_multiplier = params['mr_sl_atr']
                sim.mean_revert_engine.rsi_oversold = params['mr_rsi_low']
                sim.mean_revert_engine.rsi_overbought = 100.0 - params['mr_rsi_low']

                sim.trend_engine.sl_atr_multiplier = params['tf_sl_atr']
                sim.trend_engine.trailing_atr_multiplier = params['tf_trail_atr']
                sim.trend_engine.adx_threshold = params['tf_adx_thresh']

                res = sim.run(df_train)
                # 목적 함수: Sharpe Ratio + Profit Factor 가중
                score = res['sharpe_ratio'] + (res['profit_factor'] if res['profit_factor'] < 5 else 5.0)

                if score > best_score:
                    best_score = score
                    best_params = params

            print(f"  * 최적 파라미터 도출: {best_params} (IS Score: {best_score:.2f})")

            # 2. Out-of-Sample 테스트 적용
            oos_sim = BacktestSimulator(initial_capital=10000.0)
            oos_sim.mean_revert_engine.sl_atr_multiplier = best_params['mr_sl_atr']
            oos_sim.mean_revert_engine.rsi_oversold = best_params['mr_rsi_low']
            oos_sim.mean_revert_engine.rsi_overbought = 100.0 - best_params['mr_rsi_low']

            oos_sim.trend_engine.sl_atr_multiplier = best_params['tf_sl_atr']
            oos_sim.trend_engine.trailing_atr_multiplier = best_params['tf_trail_atr']
            oos_sim.trend_engine.adx_threshold = best_params['tf_adx_thresh']

            oos_res = oos_sim.run(df_test)
            print(f"  * OOS 검증 결과: 수익률 {oos_res['total_return_pct']:+.2f}%, MDD {oos_res['mdd_pct']:.2f}%, 승률 {oos_res['win_rate_pct']:.1f}%")

            if not oos_res['trades_df'].empty:
                all_oos_trades.extend(oos_res['trades_df'].to_dict(orient="records"))

            window_reports.append({
                "window": window_id,
                "train_period": f"{train_start_date}~{train_end_date}",
                "test_period": f"{test_start_date}~{test_end_date}",
                "best_params": best_params,
                "oos_return_pct": oos_res['total_return_pct'],
                "oos_mdd_pct": oos_res['mdd_pct'],
                "oos_trades": oos_res['total_trades'],
                "oos_win_rate": oos_res['win_rate_pct'],
            })

            start_idx += self.step_bars
            window_id += 1

        # 3. 전체 Out-of-Sample 성과 누적 계산
        df_all_oos = pd.DataFrame(all_oos_trades)
        return {
            "window_reports": window_reports,
            "all_oos_trades": df_all_oos,
        }
