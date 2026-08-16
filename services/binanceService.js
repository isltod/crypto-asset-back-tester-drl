const axios = require('axios');
// VALID_INTERVALS는 타임프레임 문자열
const { BINANCE_FUTURES_API_URL, MAX_KLINES_LIMIT, VALID_INTERVALS } = require('../config/constants');

/**
 * 바이낸스 선물 Klines(캔들) 데이터를 수집하는 서비스
 */
class BinanceService {
  // 아래 fetchKlines 메서드가 {} 객체를 받아서 처리하는 객체 구조 분해 할당 방식...
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
    // 변수는 let, 상수는 const, var는 가급적 사용하지 말라고...
    let currentStartTime = startTime ? Number(startTime) : undefined;
    const finalEndTime = endTime ? Number(endTime) : Date.now();

    // 받을 ohlcv 빈 배열로, 더 받는지는 일단 true로 해놓고 시작...
    let allKlines = [];
    let hasMore = true;

    while (hasMore) {

      // limit가 있고, 남은 개수가 1500개보다 작으면 그만큼만, 아니면 1500개...
      const fetchLimit = limit && (limit - allKlines.length < MAX_KLINES_LIMIT)
        ? (limit - allKlines.length)
        : MAX_KLINES_LIMIT;

      // 근데 그 값이 0보다 작으면 반복문 끝내고...
      if (fetchLimit <= 0) break;

      // 바이낸스 api 요청할 때 심볼, 타임프레임, 갯수를 params에 담아 보낸다...
      const queryParams = {
        symbol: formattedSymbol,
        interval,
        limit: fetchLimit,
      };

      // 시작 시점이나 종료 시점이 있으면 요청 매개변수에 추가...
      if (currentStartTime) {
        queryParams.startTime = currentStartTime;
      }
      if (finalEndTime) {
        queryParams.endTime = finalEndTime;
      }

      // ohlcv 요청해서 비동기로 받아보고...
      try {
        const response = await axios.get(`${BINANCE_FUTURES_API_URL}/fapi/v1/klines`, {
          params: queryParams,
          timeout: 10000,
        });

        // 받아온 데이터가 배열이 아니거나 빈 배열이면 종료...근데 이럼 오류 아닌가?
        // cats에서 뭔가 데이터 못 받아오면 그냥 대충 받은 데이터만 뿌려서 F5 했었던 문제가 이거인 듯...
        const rawData = response.data;
        if (!Array.isArray(rawData) || rawData.length === 0) {
          hasMore = false;
          break;
        }

        // [{timestapm: 1591..., datetime: "2026...", ...}, {}, ...] 형태로 정리해서 allKlines에 누적...
        const parsedData = rawData.map(item => ({
          timestamp: item[0],
          datetime: new Date(item[0]).toISOString(),
          open: parseFloat(item[1]),
          high: parseFloat(item[2]),
          low: parseFloat(item[3]),
          close: parseFloat(item[4]),
          volume: parseFloat(item[5]),
          // 봉이 닫힌 타임스탬프...
          closeTime: item[6],
          // 거래 가격...
          quoteVolume: parseFloat(item[7]),
          // 체결 건수...
          trades: item[8],
        }));
        allKlines.push(...parsedData);

        // 페이지네이션 업데이트: 마지막 캔들의 openTime + 1ms
        const lastCandleTime = rawData[rawData.length - 1][0];
        currentStartTime = lastCandleTime + 1;

        // 반환된 개수가 요청 limit보다 적거나(근데 이것도 오류긴 한데...),
        // 지정된 전체 limit에 도달했거나, endTime에 도달한 경우 종료
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

    const isPartial = Boolean(limit && allKlines.length < limit);
    const warning = isPartial
      ? `요청 수량(${limit}개)보다 부족한 ${allKlines.length}개만 수집되었습니다. (바이낸스 상장 이전 기간이거나 데이터 없음)`
      : null;

    return {
      symbol: formattedSymbol,
      interval,
      requestedLimit: limit ? Number(limit) : null,
      totalCount: allKlines.length,
      isPartial,
      warning,
      startTime: allKlines.length > 0 ? allKlines[0].timestamp : null,
      endTime: allKlines.length > 0 ? allKlines[allKlines.length - 1].timestamp : null,
      data: allKlines,
    };
  }
}

// 이것도 싱글톤 방식의 내보내기...
module.exports = new BinanceService();
