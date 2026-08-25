"""
flare.models.test_feature_pruning

4시간 Sniper 모델에 대한 피처 압축 정밀 대조:
1) [실험 1] 27개 전체 피처 (풀 세트)
2) [실험 2] 상위 8개 피처 (Gain 상위권 엄선)
3) [실험 3] 상위 3개 초핵심 피처 (ATR, Parkinson, Funding_CV)
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


def evaluate_features_set(df: pd.DataFrame, feature_cols: list[str], exp_name: str):
    df_labeled, label_col = create_asymmetric_labels(df.copy(), horizon_bars=48, min_mfe_pct=1.0, ratio_threshold=1.3)
    valid_mask = df_labeled[label_col].notnull() & (df_labeled.index >= 8640) & (df_labeled.index < len(df_labeled) - 48)
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
        clean_train_idx = train_idx[:-48]
        X_train, y_train = X.iloc[clean_train_idx], y.iloc[clean_train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        model = lgb.train(
            lgb_params, train_data, num_boost_round=500,
            valid_sets=[train_data, val_data], callbacks=callbacks
        )
        
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importances += model.feature_importance(importance_type="gain") / 5.0
        
    eval_mask = np.sum(oof_preds, axis=1) > 0
    y_true_eval = y[eval_mask]
    y_pred_eval = np.argmax(oof_preds[eval_mask], axis=1)
    
    acc = accuracy_score(y_true_eval, y_pred_eval)
    f1 = f1_score(y_true_eval, y_pred_eval, average="macro")
    
    long_probs = oof_preds[eval_mask, 1]
    short_probs = oof_preds[eval_mask, 2]
    
    high_conf_long = (long_probs >= 0.45)
    high_conf_short = (short_probs >= 0.45)
    
    long_prec = np.mean(y_true_eval[high_conf_long] == 1) * 100 if np.sum(high_conf_long) > 0 else 0
    short_prec = np.mean(y_true_eval[high_conf_short] == 2) * 100 if np.sum(high_conf_short) > 0 else 0
    
    return {
        "실험명": exp_name,
        "피처수": len(feature_cols),
        "전체정확도": acc * 100,
        "Macro F1": f1,
        "Long정밀도(승률)": long_prec,
        "Short정밀도": short_prec,
        "Long신호수": np.sum(high_conf_long)
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    df, all_features = generate_all_features(df)
    
    # 1. 27개 전체 피처
    print("[1/3] 27개 전체 피처 학습 중...")
    res_all = evaluate_features_set(df, all_features, "1. 27개 전체 피처 (풀 세트)")
    
    # 2. 상위 8개 핵심 피처
    top8_features = [
        "feat_atr_norm",
        "feat_parkinson_vol_2h",
        "feat_funding_cv_30d",
        "feat_funding_rsi_30d",
        "feat_hour_cos",
        "feat_funding_rate",
        "feat_hour_sin",
        "feat_ret_48bar"
    ]
    print("[2/3] 상위 8개 피처 학습 중...")
    res_top8 = evaluate_features_set(df, top8_features, "2. 상위 8개 핵심 피처 (Pruning)")
    
    # 3. 상위 3개 초핵심 피처 (압도적 1, 2, 3위)
    top3_features = [
        "feat_atr_norm",
        "feat_parkinson_vol_2h",
        "feat_funding_cv_30d"
    ]
    print("[3/3] 상위 3개 초핵심 피처 학습 중...")
    res_top3 = evaluate_features_set(df, top3_features, "3. 상위 3개 초핵심 피처 (TOP 3만)")
    
    comp_df = pd.DataFrame([res_all, res_top8, res_top3])
    
    print("=" * 110)
    print("[FLARE] 4시간 Sniper 모델: 피처 수(27개 vs 8개 vs 3개) 압축에 따른 성과 대조 보고서")
    print("=" * 110)
    
    header_fmt = "{:<32} | {:<5} | {:<9} | {:<9} | {:<14} | {:<12} | {:<10}"
    row_fmt = "{:<32} | {:<5} | {:>8.2f}% | {:>9.4f} | {:>13.1f}% | {:>11.1f}% | {:>10}"
    
    print(header_fmt.format("실험 조건", "피처수", "전체정확도", "Macro F1", "Long정밀도(승률)", "Short정밀도", "Long신호수"))
    print("-" * 110)
    for _, r in comp_df.iterrows():
        print(row_fmt.format(
            r["실험명"],
            r["피처수"],
            r["전체정확도"],
            r["Macro F1"],
            r["Long정밀도(승률)"],
            r["Short정밀도"],
            f"{r['Long신호수']:,}회"
        ))
    print("=" * 110)


if __name__ == "__main__":
    main()
