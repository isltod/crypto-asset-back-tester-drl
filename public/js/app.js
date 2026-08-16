/**
 * 애플리케이션 메인 오케스트레이션 진입점
 */
document.addEventListener('DOMContentLoaded', () => {
  // DOM 엘리먼트
  const fetchForm = document.getElementById('fetchForm');
  const warningToast = document.getElementById('warningToast');
  const warningMsg = document.getElementById('warningMsg');
  const jsonOutput = document.getElementById('jsonOutput');

  const symbolInput = document.getElementById('symbolInput');
  const intervalSelect = document.getElementById('intervalSelect');
  const limitInput = document.getElementById('limitInput');
  const saveSelect = document.getElementById('saveSelect');

  const chartSymbol = document.getElementById('chartSymbol');
  const chartInterval = document.getElementById('chartInterval');
  const chartDateRange = document.getElementById('chartDateRange');
  const chartTotalCount = document.getElementById('chartTotalCount');

  // ==========================================
  // 1. 하위 매니저 모듈 인스턴스 초기화
  // ==========================================
  const chartManager = new ChartManager('chartContainer');

  const datePickerManager = new DatePickerManager({
    startInputId: 'startDateInput',
    endInputId: 'endDateInput',
    intervalSelectId: 'intervalSelect',
    limitInputId: 'limitInput',
  });

  const fileListManager = new FileListManager({
    tableBodyId: 'fileListTable',
    refreshBtnId: 'refreshFilesBtn',
    onViewChart: (fileName) => viewFileChart(fileName),
  });

  // ==========================================
  // 2. 바이낸스 데이터 수집 폼 제출 이벤트
  // ==========================================
  fetchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideWarning();
    jsonOutput.textContent = '데이터 수집 중... 잠시만 기다려주세요.';

    const symbol = symbolInput.value.trim();
    const interval = intervalSelect.value;
    const limit = limitInput.value.trim();
    const save = saveSelect.value;
    const { startTime, endTime } = datePickerManager.getRangeTimestamps();

    const queryParams = new URLSearchParams({ symbol, interval, save });
    if (limit) queryParams.append('limit', limit);
    if (startTime) queryParams.append('startTime', startTime);
    if (endTime) queryParams.append('endTime', endTime);

    try {
      const response = await fetch(`/api/data/fetch?${queryParams.toString()}`);
      const result = await response.json();

      jsonOutput.textContent = JSON.stringify(result, null, 2);

      if (!result.success) {
        alert(`수집 에러: ${result.error}`);
        return;
      }

      // 차트 렌더링 및 헤더 정보 갱신
      if (Array.isArray(result.data) && result.data.length > 0) {
        chartManager.renderData(result.data);
        updateChartHeader(result.meta || { symbol, interval, totalCount: result.data.length, startTime, endTime });
      }

      // 수집 미달 경고 감지 시 토스트 알림
      const meta = result.meta || {};
      if (meta.isPartial || meta.warning) {
        showWarning(meta.warning || `요청한 수량(${meta.requestedLimit}개)보다 적은 ${meta.totalCount}개만 수집되었습니다.`);
      }

      // 파일 목록 새로고침
      if (save === 'true') {
        fileListManager.loadFileList();
      }

    } catch (err) {
      jsonOutput.textContent = `네트워크 오류: ${err.message}`;
    }
  });

  // ==========================================
  // 3. 차트 헤더 및 UI 알림 헬퍼 함수
  // ==========================================
  function updateChartHeader(meta) {
    if (chartSymbol) chartSymbol.textContent = meta.symbol || 'BTCUSDT';
    if (chartInterval) chartInterval.textContent = meta.interval || '1h';
    if (chartTotalCount) chartTotalCount.textContent = `${meta.totalCount ? meta.totalCount.toLocaleString() : 0} 캔들`;

    if (chartDateRange) {
      const startDisplay = DatePickerManager.formatToLocalDisplay(meta.startTime || meta.startDate);
      const endDisplay = DatePickerManager.formatToLocalDisplay(meta.endTime || meta.endDate);

      if (startDisplay && endDisplay) {
        chartDateRange.textContent = `${startDisplay} ~ ${endDisplay}`;
      } else {
        chartDateRange.textContent = '기간 정보 없음';
      }
    }
  }

  function showWarning(message) {
    warningMsg.textContent = message;
    warningToast.classList.add('show');
  }

  function hideWarning() {
    warningToast.classList.remove('show');
    warningMsg.textContent = '';
  }

  // 특정 저장 파일 차트로 불러오기
  async function viewFileChart(fileName) {
    try {
      jsonOutput.textContent = `'${fileName}' 파일 로딩 중...`;
      const response = await fetch(`/api/data/file/${fileName}`);
      const result = await response.json();

      if (result.success && Array.isArray(result.data)) {
        chartManager.renderData(result.data);
        updateChartHeader(result.meta || {});
        jsonOutput.textContent = `'${fileName}' 차트 렌더링 완료 (${result.data.length}개 캔들)`;
      } else {
        alert(`차트 데이터 로드 실패: ${result.error}`);
      }
    } catch (err) {
      alert(`파일 읽기 오류: ${err.message}`);
    }
  }
});
