"""
텔레그램 봇 알림 발송 모듈 (Telegram Notifier)
- .env에 TELEGRAM_BOT_TOKEN 및 TELEGRAM_CHAT_ID 설정 시 텔레그램 메시지 발송
- 미설정 시 콘솔 로깅으로 Fallback 작동
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.is_active = bool(self.bot_token and self.chat_id)

    def send_message(self, text: str):
        """텔레그램 메시지 발송 (Markdown 지원)"""
        # 콘솔 출력
        print(f"\n[NOTIFIER]\n{text}\n")

        if not self.is_active:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"[Notifier Error] 텔레그램 발송 실패: {resp.text}")
        except Exception as e:
            print(f"[Notifier Exception] 텔레그램 통신 에러: {e}")
