# pulse/data/db.py
import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

class PulseDB:
    """
    PULSE 비정형 이벤트 및 시세 매칭 데이터를 관리하는 경량 SQLite 매니저
    - event_type: 'ARTICLE' (RSS 완성형 기사) vs 'FLASH' (텔레그램 초고속 속보)
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "pulse_events.db")
        
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,       -- 'ARTICLE' or 'FLASH'
                    source TEXT NOT NULL,           -- 'CoinDesk', 'TreeNews', etc.
                    guid TEXT UNIQUE NOT NULL,      -- 중복 방지 고유 키
                    title TEXT NOT NULL,            -- 기사 제목 또는 속보 원문
                    link TEXT,                      -- 기사 URL 또는 채널 링크
                    summary TEXT,                   -- 본문 요약 또는 부가 내용
                    published_at TEXT,              -- 원본 발행 시각
                    collected_at TEXT NOT NULL,     -- 수집 시각 (UTC YYYY-MM-DD HH:MM:SS)
                    btc_price REAL,                 -- 당시 바이낸스 BTC 실시간 가격
                    raw_payload TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON raw_events(event_type);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_collected_at ON raw_events(collected_at);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON raw_events(source);
            """)
            conn.commit()

    def save_event(
        self,
        event_type: str,
        source: str,
        guid: str,
        title: str,
        link: str = "",
        summary: str = "",
        published_at: str = "",
        btc_price: Optional[float] = None,
        raw_payload: str = ""
    ) -> bool:
        """
        새로운 이벤트를 DB에 저장. 중복(guid 기준)인 경우 False 반환
        """
        now_iso = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO raw_events (
                        event_type, source, guid, title, link, summary, published_at, collected_at, btc_price, raw_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (event_type, source, guid, title, link, summary, published_at, now_iso, btc_price, raw_payload))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # 이미 수집된 중복 이벤트
                return False

    def get_event_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_events")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_events WHERE event_type = 'ARTICLE'")
            articles = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_events WHERE event_type = 'FLASH'")
            flashes = cursor.fetchone()[0]
            
            return {
                "total": total,
                "articles": articles,
                "flashes": flashes
            }
