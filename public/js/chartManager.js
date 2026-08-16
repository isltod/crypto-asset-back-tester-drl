/**
 * TradingView Lightweight Charts 차트 제어 전용 모듈
 */
class ChartManager {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.chart = null;
    this.candlestickSeries = null;
    this.volumeSeries = null;
    this.initChart();
  }

  /**
   * 차트 인스턴스 및 시리즈 초기화
   */
  initChart() {
    if (!this.container || typeof LightweightCharts === 'undefined') {
      console.warn('LightweightCharts 라이브러리가 로드되지 않았거나 컨테이너가 없습니다.');
      return;
    }

    // 기존 차트 제거
    this.container.innerHTML = '';

    // 차트 생성 (다크 테마 및 로컬 타임존 포맷터 적용)
    this.chart = LightweightCharts.createChart(this.container, {
      width: this.container.clientWidth,
      height: 450,
      layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1f293d' },
        horzLines: { color: '#1f293d' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      localization: {
        dateFormat: 'yyyy-MM-dd',
        // 십자선 툴팁의 시간을 로컬 타임존(KST)으로 포맷팅
        timeFormatter: (timestamp) => {
          const date = new Date(timestamp * 1000);
          const pad = (n) => String(n).padStart(2, '0');
          const yyyy = date.getFullYear();
          const mm = pad(date.getMonth() + 1);
          const dd = pad(date.getDate());
          const hh = pad(date.getHours());
          const min = pad(date.getMinutes());
          return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
        },
      },
      rightPriceScale: {
        borderColor: '#2b3648',
      },
      timeScale: {
        borderColor: '#2b3648',
        timeVisible: true,
        secondsVisible: false,
        // 가로 X축 눈금 시간도 로컬 타임존(KST)으로 포맷팅
        tickMarkFormatter: (time) => {
          const date = new Date(time * 1000);
          const pad = (n) => String(n).padStart(2, '0');
          const mm = pad(date.getMonth() + 1);
          const dd = pad(date.getDate());
          const hh = pad(date.getHours());
          const min = pad(date.getMinutes());
          return `${mm}/${dd} ${hh}:${min}`;
        },
      },
    });

    // 1. 캔들스틱 시리즈 추가
    this.candlestickSeries = this.chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // 2. 거래량(Volume) 히스토그램 시리즈 추가 (하단 오버레이)
    this.volumeSeries = this.chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // 별도 하단 스케일
    });

    this.volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8, // 거래량이 차트 하단 20% 영역에만 위치하도록 조정
        bottom: 0,
      },
    });

    // 창 크기 조절 시 자동 리사이징
    window.addEventListener('resize', () => {
      if (this.chart && this.container) {
        this.chart.applyOptions({ width: this.container.clientWidth });
      }
    });
  }

  /**
   * OHLCV 캔들 데이터 차트에 렌더링
   * @param {Array} rawKlines - 백엔드에서 받은 캔들 객체 배열
   */
  renderData(rawKlines) {
    if (!this.chart || !this.candlestickSeries || !Array.isArray(rawKlines) || rawKlines.length === 0) {
      return;
    }

    // TradingView 포맷으로 변환 (time은 초 단위 Unix timestamp)
    const candleData = [];
    const volumeData = [];

    rawKlines.forEach(item => {
      // timestamp가 ms 단위이면 초 단위로 변환
      const timeInSeconds = Math.floor(Number(item.timestamp) / 1000);

      candleData.push({
        time: timeInSeconds,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      });

      // 거래량 색상: 양봉이면 초록, 음봉이면 빨강
      const isUp = item.close >= item.open;
      volumeData.push({
        time: timeInSeconds,
        value: item.volume,
        color: isUp ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      });
    });

    // 시간 오름차순 정렬 보장
    candleData.sort((a, b) => a.time - b.time);
    volumeData.sort((a, b) => a.time - b.time);

    this.candlestickSeries.setData(candleData);
    this.volumeSeries.setData(volumeData);

    // 전체 차트가 한눈에 보이도록 맞춤 조정
    this.chart.timeScale().fitContent();
  }

  /**
   * 매수/매도 포지션 마커(Markers) 표시 (추후 백테스팅 시각화 대비)
   * @param {Array} markers - [{ time: 1234567, position: 'belowBar', color: '#2196F3', shape: 'arrowUp', text: 'BUY' }]
   */
  setMarkers(markers) {
    if (this.candlestickSeries && Array.isArray(markers)) {
      this.candlestickSeries.setMarkers(markers);
    }
  }
}

// 전역 인스턴스용 export
window.ChartManager = ChartManager;
