/**
 * 저장된 데이터 파일 목록 조회, 렌더링 및 삭제 전용 모듈
 */
class FileListManager {
  constructor({ tableBodyId, refreshBtnId, onViewChart }) {
    this.tableBody = document.getElementById(tableBodyId);
    this.refreshBtn = document.getElementById(refreshBtnId);
    this.onViewChart = onViewChart; // 차트 보기 클릭 시 실행할 외부 콜백

    this.init();
  }

  init() {
    if (this.refreshBtn) {
      this.refreshBtn.addEventListener('click', () => this.loadFileList());
    }

    // HTML onclick에서 호출할 수 있도록 window 전역 바인딩
    window.deleteFile = (fileName) => this.deleteFile(fileName);
    window.viewFileChart = (fileName) => {
      if (typeof this.onViewChart === 'function') {
        this.onViewChart(fileName);
      }
    };

    // 초기 파일 목록 로드
    this.loadFileList();
  }

  /**
   * 서버에서 저장된 파일 목록 조회 및 테이블 렌더링
   */
  async loadFileList() {
    if (!this.tableBody) return;

    try {
      const response = await fetch('/api/data/list');
      const result = await response.json();

      if (!result.success || !Array.isArray(result.files)) {
        this.tableBody.innerHTML = '<tr><td colspan="5">목록을 불러올 수 없습니다.</td></tr>';
        return;
      }

      if (result.files.length === 0) {
        this.tableBody.innerHTML = '<tr><td colspan="5">저장된 데이터 파일이 없습니다.</td></tr>';
        return;
      }

      this.tableBody.innerHTML = result.files.map(file => {
        const meta = file.meta || {};
        const isPartialBadge = meta.isPartial ? '<span style="color:#f59e0b; font-weight:bold;"> [일부수집]</span>' : '';
        const createdDate = meta.createdAt ? new Date(meta.createdAt).toLocaleString() : '-';

        return `
          <tr>
            <td><strong>${file.fileName}</strong>${isPartialBadge}</td>
            <td>${meta.symbol || '-'} (${meta.interval || '-'})</td>
            <td>${meta.totalCount ? meta.totalCount.toLocaleString() : 0} 캔들</td>
            <td>${createdDate}</td>
            <td style="display: flex; gap: 6px;">
              <button class="btn btn-secondary" onclick="viewFileChart('${file.fileName}')">차트 보기</button>
              <button class="btn btn-danger" onclick="deleteFile('${file.fileName}')">삭제</button>
            </td>
          </tr>
        `;
      }).join('');

    } catch (err) {
      this.tableBody.innerHTML = `<tr><td colspan="5">오류 발생: ${err.message}</td></tr>`;
    }
  }

  /**
   * 특정 데이터 파일 삭제
   */
  async deleteFile(fileName) {
    if (!confirm(`'${fileName}' 파일을 정말 삭제하시겠습니까?`)) return;

    try {
      const response = await fetch(`/api/data/file/${fileName}`, { method: 'DELETE' });
      const result = await response.json();

      if (result.success) {
        this.loadFileList();
      } else {
        alert(`삭제 실패: ${result.error}`);
      }
    } catch (err) {
      alert(`삭제 오류: ${err.message}`);
    }
  }
}

// 전역 내보내기
window.FileListManager = FileListManager;
