# RADE 시스템 코드 리뷰 및 실험 과정 평가

> **평가자**: Claude Opus 4.6  
> **평가 기준**: 코드 품질, 백테스트 신뢰성, 통계적 건전성, 퀀트 실무 관점

## 총평: B-

잘 만들어진 프로토타입이지만, **실전 투입 전에 반드시 해결해야 할 구조적 결함**이 존재합니다.

---

## ✅ 잘한 점

**1. 모듈 구조** — `engines/`, `regime/`, `risk/`, `backtest/`, `data/` 분리가 깔끔하고, 각 모듈이 단일 책임 원칙을 잘 준수합니다. [`position_manager.py`](file:///e:/Devs/cabt_drl/python/risk/position_manager.py)의 손절폭 기반 역산 사이징은 실무적으로 올바릅니다.

**2. 수수료/슬리피지 모델링** — Maker/Taker 수수료 분리, 슬리피지, 8시간 펀딩비 반영은 아마추어 백테스터가 흔히 무시하는 부분이며, 처음부터 포함한 것은 올바른 접근입니다.

**3. 실험 주도형 개발** — 11개 Ablation Study를 체계적으로 수행한 것은 매우 좋은 방법론입니다. 특히 사용자의 반직관적 질문에 대해 즉시 실험으로 검증한 대화 흐름은 퀀트 리서치의 모범적 사례입니다.

**4. Git 스냅샷 관리** — 주요 분기점마다 태그를 남겨 롤백 가능하게 한 것은 올바른 습관입니다.

---

## ❌ 문제점 (Critical Issues)

### 1. 🚨 Intra-Bar Fill Bias (심각도: 높음)

[`trend_following.py` L102-110](file:///e:/Devs/cabt_drl/python/engines/trend_following.py#L102-L110)에서 **봉 내 가격 순서**를 무시합니다:

```python
if high > pos.highest_price:
    pos.highest_price = high           # (1) high로 트레일링 올림
trailing_sl = pos.highest_price - (atr * self.trailing_atr_multiplier)
pos.sl_price = max(pos.sl_price, trailing_sl)
if low <= pos.sl_price:                # (2) low로 청산 체크
```

`high`가 먼저인지 `low`가 먼저인지 알 수 없는데, 코드는 항상 `high`를 먼저 반영합니다. 이는 **실제보다 유리한 가격에 청산**되게 만드는 "Optimistic Fill Bias"입니다. 수익률 과대 추정의 대표적 함정입니다.

### 2. 🚨 HMM Forward-Backward 스무딩 (심각도: 높음)

[`regime_manager.py` L77-79](file:///e:/Devs/cabt_drl/python/regime/regime_manager.py#L77-L79)에서 `hmmlearn`의 `predict_proba`는 **Forward-Backward 양방향 스무딩**을 수행합니다. 슬라이스의 마지막 봉이 현재 시점이므로 직접적 미래 누출은 아니지만, 스무딩 효과로 인해 **국면 판단이 실전보다 과도하게 안정적**으로 보입니다. 실전에서는 Forward-only 확률만 사용해야 합니다.

### 3. 진입 슬리피지 0% 가정 (심각도: 중간)

[`simulator.py` L168-171](file:///e:/Devs/cabt_drl/python/backtest/simulator.py#L168-L171)에서 진입을 "Maker 체결"로 모델링하여 슬리피지 0%를 적용했으나, 실전에서 시그널 확정 직후 다음 봉 open에 진입하려면 시장가를 써야 합니다.

### 4. 통계적 유의성 부재 (심각도: 중간)

96~274회 거래에서 PF 1.04~1.18을 보고하지만, **Bootstrap CI도 Monte Carlo Permutation Test도 없어** 이 수익률이 운인지 엣지인지 판별이 불가합니다. 96회 거래에서 PF 1.18은 p-value > 0.05일 가능성이 상당합니다.

### 5. exp9 코드 중복 (심각도: 낮음~중간)

[`exp9_asymmetric_short.py`](file:///e:/Devs/cabt_drl/python/exp9_asymmetric_short.py)에 `simulator.py`의 백테스트 루프를 300줄 이상 통째로 복사했고, exp10/11이 이것을 import합니다. 본체 수정 시 사본과 불일치 위험이 있습니다.

---

## ⚠️ 주의사항

### MDD 계산 버그
실험 9, 10의 MDD가 `2802%`, `4008%`로 출력되었습니다. MDD가 100%를 초과하는 것은 물리적으로 불가능하므로, exp9 사본 코드에서 100을 이중으로 곱하는 버그가 존재합니다. **이전 모델은 이 비정상적 수치에 대해 아무런 지적 없이 분석을 계속 진행**했습니다.

### 과적합 우려
동일한 1.5년 데이터를 관찰→조정→재실험에 반복 사용했습니다. `main_walk_forward.py`가 존재하지만 최종 결론에서는 사용되지 않았고, **2024.06 이후의 진정한 Out-of-Sample 검증이 없습니다**.

### 전략의 실질 성과
- 3.5년 장기: **+4.51%** (연환산 ~1.3%) — **미국채 무위험 금리(~5%/년)에도 미달**
- 같은 기간 BTC Buy & Hold: **약 +200%** ($29k → $67k)
- MDD 방어력(28% vs 77%)은 있으나, 수익률만으로는 현실적 투자 대안이라고 보기 어렵습니다

---

## 📋 개선 권고 (우선순위)

| 순위 | 항목 | 설명 |
|:---:|---|---|
| **P0** | Intra-bar fill 보수적 처리 | 손절/트레일링에서 worst-case fill 가정 |
| **P0** | MDD 버그 수정 | exp9/10의 이중 곱셈 버그 |
| **P1** | HMM Forward-only 확률 | `predict_proba` 대신 Forward 알고리즘 구현 |
| **P1** | 통계적 유의성 검정 | Bootstrap CI + Permutation Test |
| **P1** | Out-of-Sample 검증 | 2024.06 이후 6개월+ 미지 구간 테스트 |
| **P2** | 진입 슬리피지 반영 | Taker 슬리피지 적용 |
| **P2** | exp9 코드 중복 제거 | `simulator.py` 매개변수화 |

---

## 결론

**방법론(가설→실험→데이터 기반 판단→폐기/채택)은 올바르고, 사용자와의 탐구적 대화 흐름은 퀀트 리서치의 좋은 사례입니다.**

그러나 코드 수준에서 백테스트 신뢰성을 훼손하는 편향(intra-bar fill, 진입가 낙관, HMM 스무딩)이 존재하며, 통계적 유의성 검증이 전혀 없어 보고된 수익률을 액면 그대로 신뢰하기 어렵습니다. 또한 이전 모델이 MDD 2800%라는 물리적으로 불가능한 수치를 아무 의문 없이 통과시킨 점은 **결과에 대한 비판적 검증 능력의 한계**를 보여줍니다.

**P0/P1 항목을 해결한 후, 보수적 가정 하에서도 수익률이 양수인지 재확인하는 것이 실전 투입의 전제 조건입니다.**