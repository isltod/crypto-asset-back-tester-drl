# pulse/check_status.py
import sys
import os
import sqlite3
from datetime import datetime

# Windows 콘솔 UTF-8 출력 보장
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "pulse_events.db")

def check_pulse_health():
    print("="*75)
    print(" [PULSE] 수집 시스템 건강 상태 및 데이터베이스 점검")
    print("="*75)

    if not os.path.exists(DB_PATH):
        print(f"[!] 오류: DB 파일이 존재하지 않습니다: {DB_PATH}")
        return

    db_size_kb = os.path.getsize(DB_PATH) / 1024.0

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. 전체 통계
        cursor.execute("SELECT COUNT(*) FROM raw_events")
        total_cnt = cursor.fetchone()[0]

        cursor.execute("""
            SELECT event_type, COUNT(*), MIN(collected_at), MAX(collected_at)
            FROM raw_events
            GROUP BY event_type
        """)
        type_stats = cursor.fetchall()

        print(f"■ DB 파일 위치 : {DB_PATH}")
        print(f"■ DB 파일 용량 : {db_size_kb:.2f} KB")
        print(f"■ 총 수집 건수 : {total_cnt}건\n")

        print("■ 타입별 수집 현황:")
        for row in type_stats:
            print(f"  - [{row[0]:<7}] {row[1]:>4}건  | 최초: {row[2]}  ~  최근: {row[3]}")

        # 2. 소스별 통계
        cursor.execute("""
            SELECT source, event_type, COUNT(*) as cnt, MAX(collected_at) as latest
            FROM raw_events
            GROUP BY source, event_type
            ORDER BY event_type, cnt DESC
        """)
        source_stats = cursor.fetchall()

        print("\n■ 출처(Source)별 수집 현황:")
        for row in source_stats:
            print(f"  - {row['source']:<18} ({row['event_type']}) : {row['cnt']:>3}건 (최근 수집: {row['latest']})")

        # 3. 최근 수집된 5개 이벤트
        cursor.execute("""
            SELECT event_type, source, title, btc_price, collected_at
            FROM raw_events
            ORDER BY collected_at DESC
            LIMIT 5
        """)
        recent_events = cursor.fetchall()

        print("\n■ 가장 최근에 수집된 이벤트 Top 5:")
        for i, row in enumerate(recent_events, 1):
            price_str = f"${row['btc_price']:,.1f}" if row['btc_price'] else "N/A"
            print(f"  {i}. [{row['collected_at']}] [{row['event_type']}] [{row['source']}] (BTC: {price_str})")
            print(f"     -> {row['title'][:70]}...")

    print("="*75)
    print("[✓] 시스템 진단 완료: DB 정상 기록 중\n")

if __name__ == '__main__':
    check_pulse_health()
