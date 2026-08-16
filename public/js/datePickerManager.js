/**
 * 날짜/시간 선택기 (Flatpickr) 및 타임프레임 양방향 계산 전용 모듈
 */
class DatePickerManager {
  constructor({ startInputId, endInputId, intervalSelectId, limitInputId }) {
    this.startInput = document.getElementById(startInputId);
    this.endInput = document.getElementById(endInputId);
    this.intervalSelect = document.getElementById(intervalSelectId);
    this.limitInput = document.getElementById(limitInputId);

    this.startPicker = null;
    this.endPicker = null;

    this.init();
  }

  /**
   * 타임프레임별 밀리초(ms) 계산
   */
  getIntervalMs(interval) {
    const unit = interval.slice(-1);
    const value = parseInt(interval.slice(0, -1), 10);
    switch (unit) {
      case 'm': return value * 60 * 1000;
      case 'h': return value * 60 * 60 * 1000;
      case 'd': return value * 24 * 60 * 60 * 1000;
      case 'w': return value * 7 * 24 * 60 * 60 * 1000;
      default: return 60 * 60 * 1000;
    }
  }

  /**
   * Flatpickr 초기화 및 이벤트 리스너 바인딩
   */
  init() {
    if (typeof flatpickr === 'undefined') {
      console.warn('Flatpickr 라이브러리가 로드되지 않았습니다.');
      return;
    }

    const flatpickrConfig = {
      enableTime: true,
      time_24hr: true,
      dateFormat: 'Y-m-d H:i',
      minuteIncrement: 1,
    };

    this.startPicker = flatpickr(this.startInput, {
      ...flatpickrConfig,
      onChange: () => this.updateEndDateFromLimit(),
    });

    this.endPicker = flatpickr(this.endInput, {
      ...flatpickrConfig,
      onChange: () => this.updateLimitFromEndDate(),
    });

    // 타임프레임 및 개수 변경 시 자동 갱신
    this.intervalSelect.addEventListener('change', () => this.updateEndDateFromLimit());
    this.limitInput.addEventListener('input', () => this.updateEndDateFromLimit());

    // 기본값 설정 (최근 100개 캔들 전 ~ 현재)
    this.initDefaultRange();
  }

  /**
   * 시작일시 + (목표개수 * 타임프레임)으로 종료일시 자동 계산
   */
  updateEndDateFromLimit() {
    const selectedStart = this.startPicker.selectedDates[0];
    if (!selectedStart) return;

    const intervalMs = this.getIntervalMs(this.intervalSelect.value);
    const limit = parseInt(this.limitInput.value, 10) || 1;

    const calculatedEndTime = new Date(selectedStart.getTime() + (limit * intervalMs));
    this.endPicker.setDate(calculatedEndTime, false);
  }

  /**
   * (종료일시 - 시작일시) / 타임프레임으로 목표개수 자동 계산
   */
  updateLimitFromEndDate() {
    const selectedStart = this.startPicker.selectedDates[0];
    const selectedEnd = this.endPicker.selectedDates[0];
    if (!selectedStart || !selectedEnd) return;

    const startTime = selectedStart.getTime();
    const endTime = selectedEnd.getTime();
    const intervalMs = this.getIntervalMs(this.intervalSelect.value);

    if (endTime <= startTime) {
      this.limitInput.value = 1;
      this.updateEndDateFromLimit();
      return;
    }

    const calculatedLimit = Math.max(1, Math.round((endTime - startTime) / intervalMs));
    this.limitInput.value = calculatedLimit;
  }

  /**
   * 기본 날짜 범위 초기화
   */
  initDefaultRange() {
    const now = new Date();
    const intervalMs = this.getIntervalMs(this.intervalSelect.value);
    const initialLimit = parseInt(this.limitInput.value, 10) || 100;
    const initialStartTime = new Date(now.getTime() - (initialLimit * intervalMs));

    this.startPicker.setDate(initialStartTime, false);
    this.endPicker.setDate(now, false);
  }

  /**
   * 현재 선택된 시작/종료 일시 타임스탬프(ms) 반환
   */
  getRangeTimestamps() {
    const start = this.startPicker.selectedDates[0];
    const end = this.endPicker.selectedDates[0];
    return {
      startTime: start ? start.getTime() : undefined,
      endTime: end ? end.getTime() : undefined,
    };
  }

  /**
   * 타임스탬프 또는 ISO 문자열을 로컬 시간(YYYY-MM-DD HH:mm)으로 변환
   */
  static formatToLocalDisplay(timestampOrIso) {
    if (!timestampOrIso) return null;
    const date = new Date(timestampOrIso);
    if (isNaN(date.getTime())) return null;

    const pad = (n) => String(n).padStart(2, '0');
    const yyyy = date.getFullYear();
    const mm = pad(date.getMonth() + 1);
    const dd = pad(date.getDate());
    const hh = pad(date.getHours());
    const min = pad(date.getMinutes());

    return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
  }
}

// 전역 내보내기
window.DatePickerManager = DatePickerManager;
