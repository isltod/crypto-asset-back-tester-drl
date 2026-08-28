# pulse/run_collector.py
import sys
import os
import time
import asyncio
from datetime import datetime

# Windows 콘솔 UTF-8 출력 보장
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 프로젝트 루트 경로 등록
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pulse.data.db import PulseDB
from pulse.collectors.market_logger import MarketPriceFetcher
from pulse.collectors.rss_collector import RSSCollector
from pulse.collectors.telegram_collector import TelegramPublicCollector

RSS_CONFIG = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "Decrypt", "url": "https://decrypt.co/feed"},
    {"name": "TheBlock", "url": "https://www.theblock.co/rss.xml"},
]

TELEGRAM_CONFIG = [
    {"name": "CoinTelegraph_TG", "handle": "cointelegraph"},
    {"name": "WatcherGuru", "handle": "WatcherGuru"},
    {"name": "WuBlockchain", "handle": "wublockchainenglish"},
    {"name": "WhaleAlert", "handle": "whale_alert_io"},
]

class PulseMasterCollector:
    def __init__(self):
        self.db = PulseDB()
        self.market_fetcher = MarketPriceFetcher(symbol="BTCUSDT")
        
        # 1. RSS 완성형 기사 수집기 (ARTICLE)
        self.rss_collectors = [RSSCollector(f["name"], f["url"]) for f in RSS_CONFIG]
        
        # 2. 텔레그램 초고속 속보 수집기 (FLASH)
        self.tg_collectors = [TelegramPublicCollector(t["name"], t["handle"]) for t in TELEGRAM_CONFIG]
        
        self.is_running = False

    def collect_once(self) -> int:
        """
        1회 수집 사이클 실행 (신규 기사/속보 개수 반환)
        """
        current_btc_price = self.market_fetcher.fetch_current_price()
        new_events_count = 0

        # A. RSS 기사 수집 (ARTICLE)
        for col in self.rss_collectors:
            events = col.fetch_feed()
            for ev in events:
                is_new = self.db.save_event(
                    event_type=ev['event_type'],
                    source=ev['source'],
                    guid=ev['guid'],
                    title=ev['title'],
                    link=ev['link'],
                    summary=ev['summary'],
                    published_at=ev['published_at'],
                    btc_price=current_btc_price
                )
                if is_new:
                    new_events_count += 1
                    time_str = datetime.now().strftime('%H:%M:%S')
                    price_str = f"${current_btc_price:,.1f}" if current_btc_price else "N/A"
                    print(f"[{time_str}] [ARTICLE] [{ev['source']}] [BTC: {price_str}] {ev['title'][:65]}...")

        # B. 텔레그램 속보 수집 (FLASH)
        for col in self.tg_collectors:
            events = col.fetch_feed()
            for ev in events:
                is_new = self.db.save_event(
                    event_type=ev['event_type'],
                    source=ev['source'],
                    guid=ev['guid'],
                    title=ev['title'],
                    link=ev['link'],
                    summary=ev['summary'],
                    published_at=ev['published_at'],
                    btc_price=current_btc_price
                )
                if is_new:
                    new_events_count += 1
                    time_str = datetime.now().strftime('%H:%M:%S')
                    price_str = f"${current_btc_price:,.1f}" if current_btc_price else "N/A"
                    print(f"[{time_str}] [FLASH]   [{ev['source']}] [BTC: {price_str}] {ev['title'][:65]}...")

        return new_events_count

    async def run_forever(self, poll_interval_sec: int = 30):
        self.is_running = True
        stats = self.db.get_event_stats()
        print("="*80)
        print(" [PULSE] 하이브리드 Raw 데이터 수집 데몬이 시작되었습니다.")
        print(f" - 감시 RSS 피드(ARTICLE): {len(self.rss_collectors)}개")
        print(f" - 감시 텔레그램(FLASH):   {len(self.tg_collectors)}개")
        print(f" - 현재 DB 통계: 총 {stats['total']}건 (기사: {stats['articles']}건 | 속보: {stats['flashes']}건)")
        print("="*80)

        # 초기 1회 실행
        initial_new = self.collect_once()
        stats = self.db.get_event_stats()
        print(f"\n[*] 초기 피드 스캔 완료: +{initial_new}건 추가됨 (총 {stats['total']}건: 기사 {stats['articles']} / 속보 {stats['flashes']})\n")

        while self.is_running:
            try:
                await asyncio.sleep(poll_interval_sec)
                self.collect_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[!] 수집 루프 예외 발생: {e}")
                await asyncio.sleep(5)

        print("\n[PULSE] 수집 데몬이 정상 종료되었습니다.")

def main():
    collector = PulseMasterCollector()
    
    # 단발성 테스트 모드 (--once)
    if "--once" in sys.argv:
        print("[*] 하이브리드 1회 수집 테스트 실행 중...")
        new_cnt = collector.collect_once()
        stats = collector.db.get_event_stats()
        print(f"[*] 테스트 완료: 신규 등록 {new_cnt}건")
        print(f"    -> 총 DB 통계: 총 {stats['total']}건 (기사 ARTICLE: {stats['articles']}건 | 속보 FLASH: {stats['flashes']}건)")
        return

    # 24시간 백그라운드 모드
    try:
        asyncio.run(collector.run_forever(poll_interval_sec=30))
    except KeyboardInterrupt:
        print("\n[PULSE] 사용자에 의해 중단되었습니다.")

if __name__ == '__main__':
    main()
