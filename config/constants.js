const path = require('path');
require('dotenv').config();

module.exports = {
  PORT: process.env.PORT || 5000,
  BINANCE_FUTURES_API_URL: process.env.BINANCE_FUTURES_API_URL || 'https://fapi.binance.com',
  DATA_DIR: path.join(__dirname, '../data'),
  
  // Binance 선물 Klines 지원 타임프레임 목록
  VALID_INTERVALS: ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'],
  
  // 바이낸스 Klines API 1회 최대 개수 제한
  MAX_KLINES_LIMIT: 1500,
};
