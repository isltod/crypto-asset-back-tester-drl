"""
flare.backtest.engine

5분봉 기반 정밀 트리플 배리어(TP/SL/Timeout) 백테스트 엔진
- 지정가 익절 (Maker TP)
- 서버 시장가 손절 (Taker SL)
- 타임아웃 만기 청산 (Timeout Exit)
- 동적 리스크 관리 (매 N시간마다 손절선 바짝 조이기 지원)
- 수수료(Maker 0.02%, Taker 0.05%) 및 슬리피지(0.02%) 정밀 반영
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    return_pct: float       # 순수익률 (수수료/슬리피지 차감 후 %)
    exit_reason: str        # "TP", "SL", "TIMEOUT", "DYNAMIC_SL"
    hold_bars: int          # 보유 봉 수 (5분봉 단위)
    mfe_pct: float          # 보유 기간 중 최대 유리 진폭 (%)
    mae_pct: float          # 보유 기간 중 최대 불리 진폭 (%)


class TripleBarrierEngine:
    def __init__(
        self,
        fee_maker_pct: float = 0.02,    # 지정가 TP 수수료 (0.02%)
        fee_taker_pct: float = 0.05,    # 시장가 진입/손절 수수료 (0.05%)
        slippage_pct: float = 0.02      # 시장가 체결 시 슬리피지 (0.02%)
    ):
        self.fee_maker = fee_maker_pct
        self.fee_taker = fee_taker_pct
        self.slippage = slippage_pct

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,             # True/False 진입 시그널
        tp_pct: float = 1.5,            # 목표 익절선 (+%)
        sl_pct: float = 1.0,            # 방어 손절선 (-%)
        max_horizon_bars: int = 48,     # 최대 보유 봉 수 (48봉 = 4시간, 288봉 = 24시간)
        dynamic_check_interval: int = 0,# 동적 리스크 점검 주기 (0이면 정적 고정, 48이면 매 4시간마다 점검)
        dynamic_eval_func: Optional[Callable[[pd.DataFrame, int, float, float], tuple[bool, float]]] = None
        # dynamic_eval_func(df, current_idx, entry_price, current_sl) -> (should_tighten, new_sl)
    ) -> tuple[pd.DataFrame, dict]:
        """
        단일 포지션(No Overlapping) 원칙으로 시계열을 순회하며 백테스트를 수행합니다.
        """
        trades: list[TradeRecord] = []
        n_bars = len(df)
        
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        datetimes = df["datetime"].values
        sig_arr = signals.values
        
        i = 0
        while i < n_bars:
            if not sig_arr[i]:
                i += 1
                continue
                
            # 포지션 진입 (신호 발생 봉의 종가 체결, Taker 수수료 + 슬리피지 적용)
            entry_idx = i
            entry_time = pd.Timestamp(datetimes[entry_idx])
            raw_entry_price = closes[entry_idx]
            entry_price = raw_entry_price * (1.0 + self.slippage / 100.0) # 슬리피지
            entry_fee = self.fee_taker
            
            curr_tp_price = entry_price * (1.0 + tp_pct / 100.0)
            curr_sl_price = entry_price * (1.0 - sl_pct / 100.0)
            
            exit_idx = entry_idx
            exit_price = entry_price
            exit_reason = "TIMEOUT"
            exit_fee = self.fee_taker
            
            max_p = entry_price
            min_p = entry_price
            
            # 미래 캔들 순회
            for step in range(1, max_horizon_bars + 1):
                curr_idx = entry_idx + step
                if curr_idx >= n_bars:
                    break
                    
                h = highs[curr_idx]
                l = lows[curr_idx]
                c = closes[curr_idx]
                
                max_p = max(max_p, h)
                min_p = min(min_p, l)
                
                # 1. 동적 리스크 관리 (매 N봉마다 손절선 조이기 점검)
                if dynamic_check_interval > 0 and dynamic_eval_func is not None:
                    if step % dynamic_check_interval == 0:
                        should_tighten, new_sl = dynamic_eval_func(df, curr_idx, entry_price, curr_sl_price)
                        if should_tighten:
                            curr_sl_price = max(curr_sl_price, new_sl)
                
                # 2. 손절 체크 (Low가 손절선 터치 시 시장가 즉시 탈출)
                if l <= curr_sl_price:
                    exit_idx = curr_idx
                    exit_price = curr_sl_price * (1.0 - self.slippage / 100.0)
                    exit_reason = "SL"
                    exit_fee = self.fee_taker
                    break
                    
                # 3. 익절 체크 (High가 익절선 터치 시 지정가 체결)
                if h >= curr_tp_price:
                    exit_idx = curr_idx
                    exit_price = curr_tp_price
                    exit_reason = "TP"
                    exit_fee = self.fee_maker
                    break
                    
                # 4. 시간 만료 (Timeout)
                if step == max_horizon_bars:
                    exit_idx = curr_idx
                    exit_price = c * (1.0 - self.slippage / 100.0)
                    exit_reason = "TIMEOUT"
                    exit_fee = self.fee_taker
                    break
            
            # 거래 수익률 계산 (수수료 차감)
            raw_ret_pct = (exit_price - entry_price) / entry_price * 100.0
            net_ret_pct = raw_ret_pct - (entry_fee + exit_fee)
            
            mfe_pct = (max_p - entry_price) / entry_price * 100.0
            mae_pct = (entry_price - min_p) / entry_price * 100.0
            
            trade = TradeRecord(
                entry_time=entry_time,
                exit_time=pd.Timestamp(datetimes[exit_idx]),
                direction="LONG",
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=net_ret_pct,
                exit_reason=exit_reason,
                hold_bars=exit_idx - entry_idx,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct
            )
            trades.append(trade)
            
            # 다음 진입은 현재 포지션 청산 이후부터 탐색 (중첩 진입 방지)
            i = exit_idx + 1
            
        # 통계 요약
        trade_df = pd.DataFrame([t.__dict__ for t in trades])
        metrics = self._calculate_metrics(trade_df)
        return trade_df, metrics

    def _calculate_metrics(self, trade_df: pd.DataFrame) -> dict:
        if len(trade_df) == 0:
            return {"total_trades": 0, "cumulative_return": 0.0, "win_rate": 0.0}
            
        rets = trade_df["return_pct"]
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        
        cum_ret = rets.sum()
        win_rate = len(wins) / len(rets) * 100.0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss
        
        # MDD 계산 (복리 누적 기준)
        equity_curve = (1.0 + rets / 100.0).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100.0
        mdd = abs(drawdown.min())
        
        # Sharpe
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(len(rets))
        
        return {
            "total_trades": len(trade_df),
            "win_rate": win_rate,
            "cumulative_return": cum_ret,
            "profit_factor": profit_factor,
            "mdd": mdd,
            "sharpe_ratio": sharpe,
            "avg_return_per_trade": rets.mean(),
            "tp_count": np.sum(trade_df["exit_reason"] == "TP"),
            "sl_count": np.sum(trade_df["exit_reason"] == "SL"),
            "timeout_count": np.sum(trade_df["exit_reason"] == "TIMEOUT")
        }
