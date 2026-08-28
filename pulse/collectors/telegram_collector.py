# pulse/collectors/telegram_collector.py
import urllib.request
import re
import html
from datetime import datetime
from typing import List, Dict, Any, Optional

class TelegramPublicCollector:
    """
    공개 텔레그램 속보 채널(t.me/s/{channel})의 웹 피드를 인증 키 없이 실시간 파싱하는 초경량 수집기 (FLASH 타입)
    """
    def __init__(self, name: str, channel_handle: str):
        self.name = name
        self.channel_handle = channel_handle.replace("@", "")
        self.url = f"https://t.me/s/{self.channel_handle}"

    def fetch_feed(self) -> List[Dict[str, Any]]:
        events = []
        try:
            req = urllib.request.Request(
                self.url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html_text = resp.read().decode('utf-8', errors='ignore')

            # 메시지 파싱: data-post, message_text, time
            msgs = re.findall(
                r'data-post="([^"]+)".*?<div class="tgme_widget_message_text[^>]*>(.*?)</div>.*?<time[^>]*datetime="([^"]+)"',
                html_text,
                re.DOTALL
            )

            for post_id, raw_msg, pub_time in msgs:
                clean_text = re.sub(r'<br\s*/?>', '\n', raw_msg)
                clean_text = re.sub(r'<[^>]+>', '', clean_text)
                clean_text = html.unescape(clean_text).strip()

                if not clean_text:
                    continue

                events.append({
                    'event_type': 'FLASH',
                    'source': self.name,
                    'guid': f"TG_{post_id.strip()}",
                    'title': clean_text[:200],  # 앞부분을 타이틀로 활용
                    'link': f"https://t.me/{post_id.strip()}",
                    'summary': clean_text,
                    'published_at': pub_time.strip()
                })

        except Exception:
            pass

        return events
