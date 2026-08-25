"""
flare.models.train_lgbm

LightGBM을 활용하여:
1) Mode 0: FLARE-UltraSniper (2시간 Horizon 비대칭 국면)
2) Mode 1: FLARE-Sniper (4시간 Horizon 비대칭 국면)
3) Mode 2: FLARE-Swing (24시간 Horizon 비대칭 국면)
3가지 서로 다른 타임 호라이즌에서의 예측 정확도, F1-Score, 고확신 정밀도 및 피처 기여도를 1:1 정밀 대조하는 모듈
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features
from flare.data.labeling import create_asymmetric_labels


def load_dataset():
    """5분봉 캔들과 펀딩비를 결합하고 피처를 생성합니다."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    print(f"[*] 5분봉 캔들 데이터 로드 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"[*] 펀딩비 데이터 매핑 중...")
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    print(f"[*] 27종 통합 피처 생성 중...")
    df, feature_cols = generate_all_features(df)
    
    return df, feature_cols


def train_purged_timeseries_lgbm(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    horizon_bars: int,
    min_mfe_pct: float,
    mode_name: str
):
    """Purged TimeSeries Cross-Validation을 적용하여 LightGBM 분류기를 학습합니다."""
    df_labeled, label_col = create_asymmetric_labels(
        df.copy(), 
        horizon_bars=horizon_bars, 
        min_mfe_pct=min_mfe_pct, 
        ratio_threshold=1.3
    )
    
    valid_mask = df_labeled[label_col].notnull() & (df_labeled.index >= 8640) & (df_labeled.index < len(df_labeled) - horizon_bars)
    data = df_labeled[valid_mask].reset_index(drop=True)
    
    X = data[feature_cols]
    y = data[label_col].astype(int)
    
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(data), 3))
    feature_importances = np.zeros(len(feature_cols))
    
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
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )
        
        val_preds_prob = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds_prob
        feature_importances += model.feature_importance(importance_type="gain") / 5.0
        
    eval_mask = np.sum(oof_preds, axis=1) > 0
    y_true_eval = y[eval_mask]
    y_pred_eval = np.argmax(oof_preds[eval_mask], axis=1)
    
    overall_acc = accuracy_score(y_true_eval, y_pred_eval)
    overall_f1 = f1_score(y_true_eval, y_pred_eval, average="macro")
    
    long_probs = oof_preds[eval_mask, 1]
    short_probs = oof_preds[eval_mask, 2]
    
    high_conf_long = (long_probs >= 0.45)
    high_conf_short = (short_probs >= 0.45)
    
    long_prec = np.mean(y_true_eval[high_conf_long] == 1) * 100 if np.sum(high_conf_long) > 0 else 0
    short_prec = np.mean(y_true_eval[high_conf_short] == 2) * 100 if np.sum(high_conf_short) > 0 else 0
    
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": feature_importances
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    
    return {
        "mode": mode_name,
        "horizon": f"{horizon_bars*5//60}시간",
        "c0_ratio": np.mean(y==0)*100,
        "c1_ratio": np.mean(y==1)*100,
        "accuracy": overall_acc * 100,
        "f1": overall_f1,
        "long_precision": long_prec,
        "short_precision": short_prec,
        "long_signals": np.sum(high_conf_long),
        "short_signals": np.sum(high_conf_short),
        "top_feature": fi_df.iloc[0]["feature"],
        "top_feature_gain": fi_df.iloc[0]["gain"]
    }


def main():
    df, feature_cols = load_dataset()
    
    print("=" * 110)
    print("[FLARE] 시간 지평(2시간 vs 4시간 vs 24시간)별 LightGBM 예측력 정밀 대조 실험")
    print("=" * 110)
    
    # 1. 2시간 Horizon (24봉, 최소진폭 0.75%)
    print("[1/3] 2시간 Horizon 모델 학습 중...")
    res_2h = train_purged_timeseries_lgbm(
        df, feature_cols, horizon_bars=24, min_mfe_pct=0.75, mode_name="2시간 Horizon (Ultra-Sniper)"
    )
    
    # 2. 4시간 Horizon (48봉, 최소진폭 1.0%)
    print("[2/3] 4시간 Horizon 모델 학습 중...")
    res_4h = train_purged_timeseries_lgbm(
        df, feature_cols, horizon_bars=48, min_mfe_pct=1.0, mode_name="4시간 Horizon (Sniper)"
    )
    
    # 3. 24시간 Horizon (288봉, 최소진폭 2.0%)
    print("[3/3] 24시간 Horizon 모델 학습 중...")
    res_24h = train_purged_timeseries_lgbm(
        df, feature_cols, horizon_bars=288, min_mfe_pct=2.0, mode_name="24시간 Horizon (Swing)"
    )
    
    summary_df = pd.DataFrame([res_2h, res_4h, res_24h])
    
    print("=" * 110)
    print("📊 [비교 결과] Horizon별 머신러닝 예측 성과 대조표:")
    print("=" * 110)
    
    header_fmt = "{:<26} | {:<6} | {:<8} | {:<9} | {:<8} | {:<12} | {:<12} | {:<10}"
    row_fmt = "{:<26} | {:<6} | {:>7.1f}% | {:>8.2f}% | {:>8.4f} | {:>11.1f}% | {:>11.1f}% | {:>10}"
    
    print(header_fmt.format("모드 명칭", "Horizon", "횡보비율", "전체정확도", "F1-Score", "Long정밀도(승률)", "Short정밀도", "Long신호수"))
    print("-" * 110)
    
    for _, r in summary_df.iterrows():
        print(row_fmt.format(
            r["mode"],
            r["horizon"],
            r["c0_ratio"],
            r["accuracy"],
            r["f1"],
            r["long_precision"],
            r["short_precision"],
            f"{r['long_signals']:,}회"
        ))
    print("=" * 110)


if __name__ == "__main__":
    main()
