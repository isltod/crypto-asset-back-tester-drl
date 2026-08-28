# pulse/collectors/market_logger.py
import urllib.request
import json
from typing import Optional

class MarketPriceFetcher:
    """
    바이낸스 공개 REST API를 통해 현재 BTC/USDT 실시간 가격을 경량 조회
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}"
        self.last_price: Optional[float] = None

    def fetch_current_price(self) -> Optional[float]:
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                self.last_price = float(data['price'])
                return self.last_price
        except Exception:
            return self.last_price
