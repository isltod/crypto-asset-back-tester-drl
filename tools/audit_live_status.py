"""
tools/audit_live_status.py
- 페이퍼 트레이딩 3대 모델 실시간 상태 및 데이터 무결성 전수 감사 도구
- 실행: .venv/Scripts/python -X utf8 tools/audit_live_status.py
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

sys.stdout.reconfigure(encoding="utf-8")

from rade.live.auditor import LiveAuditor


def main():
    auditor = LiveAuditor(PROJECT_ROOT)
    full_res = auditor.run_full_audit()

    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now.astimezone(timezone(pd.Timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")

    print("\n" + "=" * 75)
    print(f" 🛡️ [RADE / FLARE 페이퍼 트레이딩 시스템 일일 무결성 감사 보고서]")
    print(f" • 감사 일시: {kst_now}")
    print("=" * 75)

    instances_map = {
        "standard": "🌟 RADE 공식 표준 (STANDARD_GOLDEN)",
        "monster_mini": "💥 작은 몬스터 실전 공격형 (MONSTER_MINI)",
        "ensemble_82": "🏰 8:2 앙상블 리밸런싱 포트폴리오 (ENSEMBLE_82)",
    }

    for inst_id, title in instances_map.items():
        res = full_res["results"].get(inst_id, {})
        status_icon = "✅ 정상 (Pass)" if res.get("is_clean") else "⚠️ 이상 감지 (Warning)"
        print(f"\n[{title}]")
        print(f" • 감사 결과: {status_icon}")
        print(f" • 현재 평가자본: ${res.get('equity', 0):,.2f}")
        print(f" • 누적 실현손익: ${res.get('accum_net_pnl', 0):+,.2f} (완료 거래: {res.get('trade_count', 0)}회)")
        print(f" • 세부 요약: {res.get('summary', '정보 없음')}")

        if res.get("issues"):
            print(" • 🚨 발견된 이슈 목록:")
            for issue in res["issues"]:
                print(f"   - {issue}")

    print("\n" + "=" * 75)
    if full_res["all_clean"]:
        print(" ✨ [전체 결과] 3대 모델 회계 항등식 및 데이터 무결성 100% 정상 통과!")
    else:
        print(" ⚠️ [전체 결과] 일부 모델에서 회계/데이터 불일치가 발견되었습니다. 위 세부 목록을 확인하세요.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
