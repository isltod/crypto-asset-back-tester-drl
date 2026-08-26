"""
flare.backtest.save_high_res_chart

FLARE 4개년(2021~2024) 레버리지별 실전 복리 자산 곡선 및 낙폭 비교 차트를
초고해상도(300 DPI, 4K급 화질)로 flare/study 폴더에 저장하는 전용 스크립트
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

# 한글 폰트 설정
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
    study_dir = Path(__file__).resolve().parent.parent / "study"
    artifact_dir = Path(r"C:\Users\wolf\.gemini\antigravity-ide\brain\58868cca-8072-4efc-a4d1-d120d04069e8")
    
    leverage_list = [
        (1.0, "1.0x (1배)", "#10B981", 2.2),
        (2.0, "2.0x (2배)", "#3B82F6", 2.4),
        (2.5, "2.5x (황금비율)", "#A855F7", 2.8),
        (3.0, "3.0x (3배)", "#F59E0B", 2.5),
        (4.0, "4.0x (4배)", "#EC4899", 2.4),
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [3.2, 1.2]}, sharex=True)
    fig.patch.set_facecolor("#0B1120")
    ax1.set_facecolor("#1E293B")
    ax2.set_facecolor("#1E293B")
    
    print("[*] 고해상도 차트 생성용 데이터 시뮬레이션 계산 중...")
    
    for lev, label, color, lw in leverage_list:
        df_eq = get_timeline_equity(symbols, data_dir, lev)
        final_val = df_eq["equity"].iloc[-1]
        mdd_val = abs(df_eq["drawdown"].min())
        legend_label = f"{label:12} | 최종: ₩{final_val:9,.0f} (+{(final_val-1e6)/1e6*100:6.1f}%) | MDD: {mdd_val:4.1f}%"
        
        ax1.plot(df_eq["datetime"], df_eq["equity"] / 10000, label=legend_label, color=color, linewidth=lw)
        ax2.plot(df_eq["datetime"], df_eq["drawdown"], label=f"{label}", color=color, linewidth=1.4, alpha=0.85)
        
    # 상단 자산 곡선
    ax1.set_title("FLARE Multi-Asset Swing Strategy: 4-Year Compound Equity Growth (2021~2024)\n[4대 코인(BTC, ETH, SOL, XRP) 1/4 균등 분할 동시 운용 실전 복리 곡선]", fontsize=16, color="#F8FAFC", pad=18, fontweight="bold")
    ax1.set_ylabel("계좌 총 자산 (단위: 만 원)", fontsize=13, color="#E2E8F0", labelpad=10)
    ax1.tick_params(colors="#94A3B8", labelsize=11)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")
    ax1.axhline(100, color="#EF4444", linestyle=":", alpha=0.6, linewidth=1.5, label="초기 원금 (100만 원)")
    ax1.legend(loc="upper left", facecolor="#0B1120", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=11, framealpha=0.95, borderpad=1.0)
    
    # 하단 Drawdown
    ax2.set_title("Drawdown Analysis (실시간 계좌 낙폭 %)", fontsize=13, color="#F8FAFC", pad=12, fontweight="bold")
    ax2.set_ylabel("낙폭 (%)", fontsize=12, color="#E2E8F0", labelpad=10)
    ax2.set_xlabel("운용 기간 (2021년 ~ 2024년)", fontsize=13, color="#E2E8F0", labelpad=10)
    ax2.tick_params(colors="#94A3B8", labelsize=11)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    
    # 300 DPI 초고해상도로 저장
    out_file1 = study_dir / "4개년_레버리지별_자산곡선_비교.png"
    out_file2 = study_dir / "equity_curves_leverage_grid.png"
    out_artifact = artifact_dir / "equity_curves_leverage_grid.png"
    
    plt.savefig(out_file1, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(out_file2, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(out_artifact, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    
    print(f"[*] 🌟 300 DPI 초고해상도 이미지 저장 완료:")
    print(f"    1. {out_file1}")
    print(f"    2. {out_file2}")


if __name__ == "__main__":
    main()
