document.addEventListener('DOMContentLoaded', () => {
  const fetchForm = document.getElementById('fetchForm');
  const warningToast = document.getElementById('warningToast');
  const warningMsg = document.getElementById('warningMsg');
  const jsonOutput = document.getElementById('jsonOutput');
  const fileListTable = document.getElementById('fileListTable');
  const refreshFilesBtn = document.getElementById('refreshFilesBtn');

  // 저장된 파일 목록 로드
  loadFileList();

  // 데이터 수집 폼 제출 이벤트
  fetchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideWarning();
    jsonOutput.textContent = '데이터 수집 중... 잠시만 기다려주세요.';

    const symbol = document.getElementById('symbolInput').value.trim();
    const interval = document.getElementById('intervalSelect').value;
    const limit = document.getElementById('limitInput').value.trim();
    const save = document.getElementById('saveSelect').value;

    const queryParams = new URLSearchParams({
      symbol,
      interval,
      save,
    });
    if (limit) queryParams.append('limit', limit);

    try {
      const response = await fetch(`/api/data/fetch?${queryParams.toString()}`);
      const result = await response.json();

      jsonOutput.textContent = JSON.stringify(result, null, 2);

      if (!result.success) {
        alert(`수집 에러: ${result.error}`);
        return;
      }

      // [개선안 2] 수집 미달 (isPartial === true 또는 warning 발편) 감지 시 경고 알림 토스트 출력
      const meta = result.meta || {};
      if (meta.isPartial || meta.warning) {
        showWarning(meta.warning || `요청한 수량(${meta.requestedLimit}개)보다 적은 ${meta.totalCount}개만 수집되었습니다.`);
      }

      // 저장된 경우 파일 목록 갱신
      if (save === 'true') {
        loadFileList();
      }

    } catch (err) {
      jsonOutput.textContent = `네트워크 오류: ${err.message}`;
    }
  });

  // 파일 목록 새로고침 버튼
  if (refreshFilesBtn) {
    refreshFilesBtn.addEventListener('click', loadFileList);
  }

  // 저장된 데이터 파일 목록 조회 함수
  async function loadFileList() {
    try {
      const response = await fetch('/api/data/list');
      const result = await response.json();

      if (!result.success || !Array.isArray(result.files)) {
        fileListTable.innerHTML = '<tr><td colspan="5">목록을 불러올 수 없습니다.</td></tr>';
        return;
      }

      if (result.files.length === 0) {
        fileListTable.innerHTML = '<tr><td colspan="5">저장된 데이터 파일이 없습니다.</td></tr>';
        return;
      }

      fileListTable.innerHTML = result.files.map(file => {
        const meta = file.meta || {};
        const isPartialBadge = meta.isPartial ? '<span style="color:#f59e0b; font-weight:bold;"> [일부수집]</span>' : '';
        const createdDate = meta.createdAt ? new Date(meta.createdAt).toLocaleString() : '-';

        return `
          <tr>
            <td><strong>${file.fileName}</strong>${isPartialBadge}</td>
            <td>${meta.symbol || '-'} (${meta.interval || '-'})</td>
            <td>${meta.totalCount ? meta.totalCount.toLocaleString() : 0} 캔들</td>
            <td>${createdDate}</td>
            <td>
              <button class="btn btn-danger" onclick="deleteFile('${file.fileName}')">삭제</button>
            </td>
          </tr>
        `;
      }).join('');

    } catch (err) {
      fileListTable.innerHTML = `<tr><td colspan="5">오류 발생: ${err.message}</td></tr>`;
    }
  }

  // 경고 표시 토스트 함수
  function showWarning(message) {
    warningMsg.textContent = message;
    warningToast.classList.add('show');
  }

  // 경고 숨김 함수
  function hideWarning() {
    warningToast.classList.remove('show');
    warningMsg.textContent = '';
  }

  // 전역 삭제 함수 등록
  window.deleteFile = async function(fileName) {
    if (!confirm(`'${fileName}' 파일을 정말 삭제하시겠습니까?`)) return;

    try {
      const response = await fetch(`/api/data/file/${fileName}`, { method: 'DELETE' });
      const result = await response.json();

      if (result.success) {
        loadFileList();
      } else {
        alert(`삭제 실패: ${result.error}`);
      }
    } catch (err) {
      alert(`삭제 오류: ${err.message}`);
    }
  };
});
