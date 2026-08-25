"""
flare.backtest.run_backtest

FLARE 4대 모드(Sniper-Pure, Sniper-ML, Swing-Pure, Swing-Dynamic)
실전 수수료/슬리피지 반영 통합 백테스트 실행 및 4열 대조 성과 분석
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features
from flare.data.labeling import create_asymmetric_labels
from flare.backtest.engine import TripleBarrierEngine


def train_lgbm_model(df: pd.DataFrame, feature_cols: list[str], horizon_bars: int = 48):
    """4시간 Sniper LightGBM 모델을 학습하고 Out-of-Fold 예측 확률을 반환합니다."""
    print(f"[*] 4시간 Sniper LightGBM 게이트 모델 학습 중...")
    df_labeled, label_col = create_asymmetric_labels(df.copy(), horizon_bars=horizon_bars, min_mfe_pct=1.0, ratio_threshold=1.3)
    valid_mask = df_labeled[label_col].notnull() & (df_labeled.index >= 8640) & (df_labeled.index < len(df_labeled) - horizon_bars)
    
    X = df_labeled.loc[valid_mask, feature_cols].reset_index(drop=True)
    y = df_labeled.loc[valid_mask, label_col].astype(int).reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = np.zeros((len(df), 3))
    valid_indices = df_labeled[valid_mask].index.values
    
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        clean_train_idx = train_idx[:-horizon_bars]
        X_train, y_train = X.iloc[clean_train_idx], y.iloc[clean_train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        model = lgb.train(
            lgb_params, train_data, num_boost_round=500,
            valid_sets=[train_data, val_data], callbacks=callbacks
        )
        
        val_orig_idx = valid_indices[val_idx]
        oof_probs[val_orig_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
    return oof_probs


def run_all_4modes_backtest():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    print(f"[*] 5분봉 데이터 로드: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    df, feature_cols = generate_all_features(df)
    
    # 4h LightGBM OOF 예측 확률 산출
    oof_probs = train_lgbm_model(df, feature_cols, horizon_bars=48)
    df["ml_prob_long"] = oof_probs[:, 1]
    df["ml_prob_short"] = oof_probs[:, 2]
    
    # 웜업 구간 제외
    eval_df = df.iloc[8640:].reset_index(drop=True)
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    # =========================================================================
    # [Mode 1.1] FLARE-Sniper-Pure (순수 룰 기반: 펀딩과열 + 청산꼬리)
    # =========================================================================
    sig_sniper_pure = (
        (eval_df["feat_funding_rsi_30d"] <= 0.10) & 
        (eval_df["feat_is_lower_wick_spike"] == 1.0)
    )
    trades_1_1, m_1_1 = engine.run_backtest(
        eval_df, sig_sniper_pure, tp_pct=1.4, sl_pct=1.0, max_horizon_bars=48
    )
    
    # =========================================================================
    # [Mode 1.2] FLARE-Sniper-ML (룰 + 4h ML 게이트: Prob >= 0.45)
    # =========================================================================
    sig_sniper_ml = (
        (eval_df["feat_funding_rsi_30d"] <= 0.10) & 
        (eval_df["feat_is_lower_wick_spike"] == 1.0) &
        (eval_df["ml_prob_long"] >= 0.45)
    )
    trades_1_2, m_1_2 = engine.run_backtest(
        eval_df, sig_sniper_ml, tp_pct=1.4, sl_pct=1.0, max_horizon_bars=48
    )
    
    # =========================================================================
    # [Mode 2.1] FLARE-Swing-Pure (8시간 펀딩비 하위 5% 정적 스윙)
    # =========================================================================
    # 8시간 정산 시점 (00:00, 08:00, 16:00 UTC)
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing_pure = (
        is_settle_bar & is_settle_hour & 
        (eval_df["feat_funding_rsi_30d"] <= 0.05)
    )
    trades_2_1, m_2_1 = engine.run_backtest(
        eval_df, sig_swing_pure, tp_pct=3.0, sl_pct=3.0, max_horizon_bars=288
    )
    
    # =========================================================================
    # [Mode 2.2] FLARE-Swing-Dynamic (8h 스윙 + 매 4h ML 점검 동적 SL 조이기)
    # =========================================================================
    def dynamic_sl_evaluation(data: pd.DataFrame, curr_idx: int, entry_p: float, curr_sl_p: float):
        # 현재 시점의 ML 숏 하방 위험도 체크
        prob_short = data.loc[curr_idx, "ml_prob_short"]
        curr_price = data.loc[curr_idx, "close"]
        
        # 숏 위험 확률 40% 이상 감지 시
        if prob_short >= 0.40:
            # 현재 가격이 진입가보다 위에 있으면 본전(0.0%)으로 SL 당김
            if curr_price >= entry_p:
                new_sl = entry_p * 1.0005 # 본전 + 수수료 방어
                return True, new_sl
            else:
                # 현재 가격이 마이너스면 -1.0% 선으로 손절폭 타이트하게 조임
                new_sl = entry_p * 0.9900
                return True, new_sl
        return False, curr_sl_p

    trades_2_2, m_2_2 = engine.run_backtest(
        eval_df, sig_swing_pure, tp_pct=3.0, sl_pct=3.0, max_horizon_bars=288,
        dynamic_check_interval=48, dynamic_eval_func=dynamic_sl_evaluation
    )
    
    # =========================================================================
    # 4대 모드 종합 비교 출력
    # =========================================================================
    results = [
        {"모드": "Mode 1.1 (Sniper-Pure)", "Horizon": "4시간", **m_1_1},
        {"모드": "Mode 1.2 (Sniper-ML)", "Horizon": "4시간", **m_1_2},
        {"모드": "Mode 2.1 (Swing-Pure)", "Horizon": "24시간", **m_2_1},
        {"모드": "Mode 2.2 (Swing-Dynamic)", "Horizon": "24시간", **m_2_2}
    ]
    res_df = pd.DataFrame(results)
    
    print("\n" + "=" * 115)
    print("🏆 [FLARE Phase 3] 실전 수수료/슬리피지 반영 4대 모드 통합 백테스트 성과 보고서 (2022~2024)")
    print("=" * 115)
    
    header_fmt = "{:<25} | {:<6} | {:<6} | {:<8} | {:<12} | {:<7} | {:<8} | {:<8} | {:<15}"
    row_fmt = "{:<25} | {:<6} | {:>6} | {:>7.1f}% | {:>11.2f}% | {:>7.2f} | {:>7.2f}% | {:>8.2f} | TP:{:<2} SL:{:<2} TO:{:<2}"
    
    print(header_fmt.format("모드 명칭", "Horizon", "거래수", "승률", "총 누적수익률", "손익비", "최대낙폭(MDD)", "샤프지수", "청산사유 분포"))
    print("-" * 115)
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["모드"],
            r["Horizon"],
            f"{r['total_trades']}회",
            r["win_rate"],
            r["cumulative_return"],
            r["profit_factor"],
            r["mdd"],
            r["sharpe_ratio"],
            r["tp_count"],
            r["sl_count"],
            r["timeout_count"]
        ))
    print("=" * 115)


if __name__ == "__main__":
    run_all_4modes_backtest()
