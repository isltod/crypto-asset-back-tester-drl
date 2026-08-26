"""
flare.backtest.plot_leverage_equity_curves

4대 정예 코인 (BTC, ETH, SOL, XRP) 1/4 균등 분할 동시 운용에 대한
레버리지 수준별 (1.0x, 2.0x, 2.5x, 3.0x, 4.0x) 4개년(2021~2024) 실시간 자산 곡선 & MDD 비교 차트 생성기
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 한글 폰트 설정 (Windows 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multicoin_unified_account import load_coin_events


@dataclass
class PositionSlot:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    margin_cost: float
    leverage: float
    sl_price: float
    max_bars: int = 24
    bars_held: int = 0


def get_timeline_equity(symbols: List[str], data_dir: Path, leverage: float):
    n_slots = len(symbols)
    slot_weight = 1.0 / n_slots
    dfs = [load_coin_events(sym, data_dir) for sym in symbols]
    coin_dict = {df["symbol"].iloc[0]: df.set_index("datetime") for df in dfs}
    master_timeline = pd.date_range("2021-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="1h", tz="UTC")
    
    cash = 1_000_000.0
    active_positions: Dict[str, PositionSlot] = {}
    equity_records = []
    
    for current_time in master_timeline:
        closed_symbols = []
        for sym, pos in list(active_positions.items()):
            sym_data = coin_dict[sym]
            if current_time in sym_data.index:
                row = sym_data.loc[current_time]
                pos.bars_held += 1
                exit_price = None
                if row["low"] <= pos.sl_price:
                    exit_price = pos.sl_price * 0.9998
                elif pos.bars_held >= pos.max_bars:
                    exit_price = row["close"] * 0.9998
                if exit_price is not None:
                    raw_pnl = (exit_price - pos.entry_price) * pos.position_size
                    exit_fee = (exit_price * pos.position_size) * 0.0005
                    net_trade_pnl = raw_pnl - exit_fee
                    cash += pos.margin_cost + net_trade_pnl
                    closed_symbols.append(sym)
        for sym in closed_symbols:
            del active_positions[sym]
            
        current_margin = sum(p.margin_cost for p in active_positions.values())
        total_equity = cash + current_margin
        
        for sym in symbols:
            if sym not in active_positions:
                sym_data = coin_dict[sym]
                if current_time in sym_data.index:
                    row = sym_data.loc[current_time]
                    if row["sig_swing"]:
                        trade_margin = (total_equity * slot_weight) * 0.80
                        if cash >= trade_margin:
                            c = row["close"]
                            entry_p = c * 1.0002
                            fee = (entry_p * (trade_margin * leverage / entry_p)) * 0.0005
                            usable_margin = trade_margin - fee
                            pos_size = (usable_margin * leverage) / entry_p
                            sl_rate = 0.06 if sym == "SOLUSDT" else 0.04
                            active_positions[sym] = PositionSlot(
                                symbol=sym,
                                entry_time=current_time,
                                entry_price=entry_p,
                                position_size=pos_size,
                                margin_cost=usable_margin,
                                leverage=leverage,
                                sl_price=entry_p * (1.0 - sl_rate),
                                max_bars=24
                            )
                            cash -= trade_margin
                            
        current_margin = sum(p.margin_cost for p in active_positions.values())
        equity_records.append({"datetime": current_time, "equity": cash + current_margin})
        
    df_eq = pd.DataFrame(equity_records)
    df_eq["peak"] = df_eq["equity"].cummax()
    df_eq["drawdown"] = (df_eq["equity"] - df_eq["peak"]) / df_eq["peak"] * 100.0
    return df_eq


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    # 저장 경로: artifact 폴더 및 flare/study 폴더
    artifact_dir = Path(r"C:\Users\wolf\.gemini\antigravity-ide\brain\58868cca-8072-4efc-a4d1-d120d04069e8")
    study_dir = Path(__file__).resolve().parent.parent / "study"
    
    leverage_list = [
        (1.0, "1.0x (1배)", "#4CAF50", 1.8),
        (2.0, "2.0x (2배)", "#2196F3", 2.0),
        (2.5, "2.5x (황금비율)", "#9C27B0", 2.4),
        (3.0, "3.0x (3배)", "#FF9800", 2.2),
        (4.0, "4.0x (4배)", "#E91E63", 2.2),
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor("#0F172A")
    ax1.set_facecolor("#1E293B")
    ax2.set_facecolor("#1E293B")
    
    print("[*] 레버리지별 자산 곡선 데이터 시뮬레이션 계산 중...")
    
    for lev, label, color, lw in leverage_list:
        print(f"    • 레버리지 {lev:.1f}x 계산 중...")
        df_eq = get_timeline_equity(symbols, data_dir, lev)
        final_val = df_eq["equity"].iloc[-1]
        mdd_val = abs(df_eq["drawdown"].min())
        legend_label = f"{label} ➔ 최종 ₩{final_val:,.0f} (+{(final_val-1e6)/1e6*100:.1f}%) | MDD {mdd_val:.1f}%"
        
        ax1.plot(df_eq["datetime"], df_eq["equity"] / 10000, label=legend_label, color=color, linewidth=lw)
        ax2.plot(df_eq["datetime"], df_eq["drawdown"], label=f"{label} DD", color=color, linewidth=1.2, alpha=0.8)
        
    # 상단 자산 곡선 서식
    ax1.set_title("FLARE Multi-Asset Swing Strategy: 4-Year Compound Equity Growth (2021~2024)\n[4대 코인(BTC, ETH, SOL, XRP) 1/4 균등 분할 동시 운용]", fontsize=15, color="#F8FAFC", pad=15, fontweight="bold")
    ax1.set_ylabel("계좌 총 자산 (만원)", fontsize=12, color="#E2E8F0")
    ax1.tick_params(colors="#94A3B8", labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")
    ax1.axhline(100, color="#94A3B8", linestyle=":", alpha=0.5, label="원금 (100만원)")
    ax1.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=10, framealpha=0.9)
    
    # 하단 Drawdown 서식
    ax2.set_title("Drawdown Analysis (낙폭 %)", fontsize=12, color="#F8FAFC", pad=10, fontweight="bold")
    ax2.set_ylabel("낙폭 (%)", fontsize=11, color="#E2E8F0")
    ax2.set_xlabel("시간 (2021 ~ 2024)", fontsize=12, color="#E2E8F0")
    ax2.tick_params(colors="#94A3B8", labelsize=10)
    ax2.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    plt.tight_layout()
    
    out_artifact = artifact_dir / "equity_curves_leverage_grid.png"
    out_study = study_dir / "equity_curves_leverage_grid.png"
    plt.savefig(out_artifact, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(out_study, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    
    print(f"[*] ✅ 고해상도 자산 곡선 그래프 생성 완료:")
    print(f"    - Artifact: {out_artifact}")
    print(f"    - Study   : {out_study}")


if __name__ == "__main__":
    main()
