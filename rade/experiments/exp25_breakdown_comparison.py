"""
[실험 25] 1위(조합 F) vs 2위(조합 C) 연도별(2021~2024) 및 국면별 정밀 분해 분석
- 조합 C: TH=0.45 (보수적 HMM), ATR 상한=4.5x, 타임스탑=24h
- 조합 F: TH=0.35 (적극적 HMM), ATR 상한=4.5x, 타임스탑=24h
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.experiments.exp16_3state_hmm import Regime3StateManager, HMM3StateSimulator


class CustomHMM3Simulator(HMM3StateSimulator):
    def __init__(self, trailing_atr_mult=4.5, mr_max_holding_bars=24, **kwargs):
        super().__init__(**kwargs)
        self.tf_engine.trailing_atr_multiplier = trailing_atr_mult
        self.tf_engine.max_trailing_atr = trailing_atr_mult
        self.mr_engine.max_holding_bars = mr_max_holding_bars


def analyze_model_breakdown(df_ind: pd.DataFrame, th: float, name: str):
    mgr = Regime3StateManager(hmm_window=720, retrain_interval=168, trans_threshold=th)
    df_proc = mgr.calculate_regimes(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    sim = CustomHMM3Simulator(trailing_atr_mult=4.5, mr_max_holding_bars=24, initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0, bear_mode="CASH")
    res = sim.run(test_df)

    trades_df = res["trades_df"]
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_time"])
    trades_df["year"] = trades_df["entry_dt"].dt.year
    trades_df["holding_hours"] = (trades_df["exit_dt"] - trades_df["entry_dt"]).dt.total_seconds() / 3600.0

    print("\n" + "=" * 95)
    print(f"      [{name}] 4개년 정밀 분해 성과 보고서")
    print("=" * 95)
    print(f"* 4개년 종합: 총수익률 {res['total_return_pct']:+.2f}% | MDD {res['mdd_pct']:.2f}% | 최종자산 ${res['final_equity']:,.2f} | 거래 {len(trades_df)}회 | PF {res['profit_factor']:.2f}")
    print("-" * 95)

    # 1. 연도별 성과
    print(f"\n[1] 연도별 성과 (Yearly Breakdown):")
    print(f"{'연도':<6} | {'수익($)':<12} | {'거래수':<8} | {'승률':<8} | {'PF':<6} | {'평균보유':<8} | {'시장 환경 및 특징'}")
    print("-" * 95)

    yearly_data = []
    for yr in [2021, 2022, 2023, 2024]:
        sub = trades_df[trades_df["year"] == yr]
        cnt = len(sub)
        if cnt > 0:
            pnl_sum = sub["pnl"].sum()
            wins = sub[sub["pnl"] > 0]
            wr = len(wins) / cnt * 100.0
            gp = wins["pnl"].sum()
            gl = abs(sub[sub["pnl"] < 0]["pnl"].sum())
            pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
            avg_h = sub["holding_hours"].mean()
        else:
            pnl_sum = 0.0
            wr = 0.0
            pf = 0.0
            avg_h = 0.0

        charac = ""
        if yr == 2021: charac = "대세 상승 불장 & 급락 랠리"
        elif yr == 2022: charac = "크립토 윈터 (69k -> 15k 대폭락)"
        elif yr == 2023: charac = "바닥 탈출 및 지루한 박스권 횡보"
        elif yr == 2024: charac = "반감기 ETF 승인 신고가 불장"

        print(f"{yr:<6} | {pnl_sum:+10.2f}$ | {cnt:5d}회 | {wr:6.1f}% | {pf:4.2f} | {avg_h:6.1f}h | {charac}")
        yearly_data.append({"year": yr, "pnl": pnl_sum, "cnt": cnt, "wr": wr, "pf": pf, "avg_h": avg_h})

    # 2. 엔진 및 포지션 방향별 성과
    print(f"\n[2] 전략 엔진 및 포지션 방향별 성과 (Engine & Side Breakdown):")
    print(f"{'엔진/방향':<25} | {'수익($)':<12} | {'거래수':<8} | {'승률':<8} | {'PF':<6} | {'평균보유':<8} | {'비중(%)'}")
    print("-" * 95)

    eng_groups = trades_df.groupby(["engine", "side"])
    for (eng, s), grp in eng_groups:
        cnt = len(grp)
        pnl_sum = grp["pnl"].sum()
        wins = grp[grp["pnl"] > 0]
        wr = len(wins) / cnt * 100.0
        gp = wins["pnl"].sum()
        gl = abs(grp[grp["pnl"] < 0]["pnl"].sum())
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        avg_h = grp["holding_hours"].mean()
        ratio = cnt / len(trades_df) * 100.0
        label = f"{eng} ({s})"
        print(f"{label:<25} | {pnl_sum:+10.2f}$ | {cnt:5d}회 | {wr:6.1f}% | {pf:4.2f} | {avg_h:6.1f}h | {ratio:5.1f}%")

    return {
        "name": name,
        "total_return": res["total_return_pct"],
        "mdd": res["mdd_pct"],
        "pf": res["profit_factor"],
        "trades": len(trades_df),
        "yearly": yearly_data,
    }


def run_experiment_25():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_all = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_all)

    # 조합 C (TH=0.45)
    res_c = analyze_model_breakdown(df_ind, th=0.45, name="조합 C (TH=0.45 보수형, ATR=4.5x, TS=24h)")

    # 조합 F (TH=0.35)
    res_f = analyze_model_breakdown(df_ind, th=0.35, name="조합 F (TH=0.35 적극형, ATR=4.5x, TS=24h)")

    print("\n" + "=" * 95)
    print("      [최종 1:1 비교 요약표] 조합 C (TH=0.45) vs 조합 F (TH=0.35)")
    print("=" * 95)
    print(f"{'구분':<20} | {'조합 C (보수적 TH=0.45)':<32} | {'조합 F (적극적 TH=0.35)':<32}")
    print("-" * 95)
    print(f"{'4개년 총수익률':<20} | {res_c['total_return']:+10.2f}%                     | {res_f['total_return']:+10.2f}%")
    print(f"{'최대 낙폭 (MDD)':<20} | {res_c['mdd']:8.2f}%                       | {res_f['mdd']:8.2f}%")
    print(f"{'손익비 (PF)':<20} | {res_c['pf']:8.2f}                         | {res_f['pf']:8.2f}")
    print(f"{'총 거래 횟수':<20} | {res_c['trades']:5d}회 (연 {res_c['trades']/3.92:.1f}회)            | {res_f['trades']:5d}회 (연 {res_f['trades']/3.92:.1f}회)")
    print("-" * 95)
    for i, yr in enumerate([2021, 2022, 2023, 2024]):
        c_p = res_c["yearly"][i]["pnl"]
        f_p = res_f["yearly"][i]["pnl"]
        c_cnt = res_c["yearly"][i]["cnt"]
        f_cnt = res_f["yearly"][i]["cnt"]
        print(f"{yr}년 수익 (거래수) | {c_p:+8.2f}$ ({c_cnt}회)                  | {f_p:+8.2f}$ ({f_cnt}회)")
    print("=" * 95)


if __name__ == "__main__":
    run_experiment_25()
