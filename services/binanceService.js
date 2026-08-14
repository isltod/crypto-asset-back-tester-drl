const axios = require('axios');
const { BINANCE_FUTURES_API_URL, MAX_KLINES_LIMIT, VALID_INTERVALS } = require('../config/constants');

/**
 * 바이낸스 선물 Klines(캔들) 데이터를 수집하는 서비스
 */
class BinanceService {
  /**
   * 바이낸스 Klines 수집 (페이지네이션 자동 처리)
   * @param {Object} params
   * @param {string} params.symbol 예: 'BTCUSDT'
   * @param {string} params.interval 예: '1h', '5m', '1d'
   * @param {number} [params.startTime] 시작 타임스탬프 (ms)
   * @param {number} [params.endTime] 종료 타임스탬프 (ms)
   * @param {number} [params.limit] 최대 가져올 개수 (기본 전체 수집)
   */
  async fetchKlines({ symbol = 'BTCUSDT', interval = '1h', startTime, endTime, limit }) {
    if (!VALID_INTERVALS.includes(interval)) {
      throw new Error(`지원하지 않는 타임프레임입니다: ${interval}`);
    }

    const formattedSymbol = symbol.toUpperCase();
    let currentStartTime = startTime ? Number(startTime) : undefined;
    const finalEndTime = endTime ? Number(endTime) : Date.now();

    let allKlines = [];
    let hasMore = true;

    while (hasMore) {
      const fetchLimit = limit && (limit - allKlines.length < MAX_KLINES_LIMIT)
        ? (limit - allKlines.length)
        : MAX_KLINES_LIMIT;

      if (fetchLimit <= 0) break;

      const queryParams = {
        symbol: formattedSymbol,
        interval,
        limit: fetchLimit,
      };

      if (currentStartTime) {
        queryParams.startTime = currentStartTime;
      }
      if (finalEndTime) {
        queryParams.endTime = finalEndTime;
      }

      try {
        const response = await axios.get(`${BINANCE_FUTURES_API_URL}/fapi/v1/klines`, {
          params: queryParams,
          timeout: 10000,
        });

        const rawData = response.data;
        if (!Array.isArray(rawData) || rawData.length === 0) {
          hasMore = false;
          break;
        }

        const parsedData = rawData.map(item => ({
          timestamp: item[0],
          datetime: new Date(item[0]).toISOString(),
          open: parseFloat(item[1]),
          high: parseFloat(item[2]),
          low: parseFloat(item[3]),
          close: parseFloat(item[4]),
          volume: parseFloat(item[5]),
          closeTime: item[6],
          quoteVolume: parseFloat(item[7]),
          trades: item[8],
        }));

        allKlines.push(...parsedData);

        // 페이지네이션 업데이트: 마지막 캔들의 openTime + 1ms
        const lastCandleTime = rawData[rawData.length - 1][0];
        currentStartTime = lastCandleTime + 1;

        // 반환된 개수가 요청 limit보다 적거나, 지정된 전체 limit에 도달했거나, endTime에 도달한 경우 종료
        if (rawData.length < fetchLimit || (limit && allKlines.length >= limit) || currentStartTime >= finalEndTime) {
          hasMore = false;
        }

        // 바이낸스 Rate Limit 예방을 위한 미세 딜레이 (50ms)
        if (hasMore) {
          await new Promise(resolve => setTimeout(resolve, 50));
        }

      } catch (error) {
        const errorMsg = error.response?.data?.msg || error.message;
        throw new Error(`바이낸스 API 데이터 수집 실패: ${errorMsg}`);
      }
    }

    return {
      symbol: formattedSymbol,
      interval,
      totalCount: allKlines.length,
      startTime: allKlines.length > 0 ? allKlines[0].timestamp : null,
      endTime: allKlines.length > 0 ? allKlines[allKlines.length - 1].timestamp : null,
      data: allKlines,
    };
  }
}

module.exports = new BinanceService();
