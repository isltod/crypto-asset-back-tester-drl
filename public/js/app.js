// 브라우저가 DOM 트리를 다 읽으면 발생하는 이벤트 DOMContentLoaded에 함수를 등록
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
    // submit 버튼의 화면 새로고침 동작을 막고...
    e.preventDefault();
    hideWarning();
    jsonOutput.textContent = '데이터 수집 중... 잠시만 기다려주세요.';

    // 요청 데이터의 심볼, 타임프레임, 봉 갯수, 저장 여부...
    const symbol = document.getElementById('symbolInput').value.trim();
    const interval = document.getElementById('intervalSelect').value;
    const limit = document.getElementById('limitInput').value.trim();
    const save = document.getElementById('saveSelect').value;

    // 요청 매개변수 객체 생성
    const queryParams = new URLSearchParams({
      symbol,
      interval,
      save,
    });
    if (limit) queryParams.append('limit', limit);

    try {
      // 바이낸스 데이터 받고 JSON으로 읽기...
      const response = await fetch(`/api/data/fetch?${queryParams.toString()}`);
      const result = await response.json();

      // 바이낸스 데이터 받기 결과 표시...
      jsonOutput.textContent = JSON.stringify(result, null, 2);

      if (!result.success) {
        alert(`수집 에러: ${result.error}`);
        return;
      }

      // 받은 데이터가 요청 갯수에 모자라면(isPartial 또는 warning) 경고 알림 토스트 출력
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
      // list로 get 요청 보내고 기다려서 결과 읽기...
      const response = await fetch('/api/data/list');
      const result = await response.json();

      // 응답 success가 false이거나 파일 목록이 배열이 아니면 오류 표시하고 리턴
      if (!result.success || !Array.isArray(result.files)) {
        fileListTable.innerHTML = '<tr><td colspan="5">목록을 불러올 수 없습니다.</td></tr>';
        return;
      }

      // 파일 목록이 비어있으면 표시하고 리턴
      if (result.files.length === 0) {
        fileListTable.innerHTML = '<tr><td colspan="5">저장된 데이터 파일이 없습니다.</td></tr>';
        return;
      }

      // 여기 왔다는 건 데이터 파일들이 있어서 files 배열로 받았다는 거고...
      fileListTable.innerHTML = result.files.map(file => {
        // 메타 정보 받아서 일부만 받았는지, 생성시간, 파일 이름 등 정보를 테이블에 표시...
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

  // 파일 삭제 함수 등록
  window.deleteFile = async function (fileName) {
    if (!confirm(`'${fileName}' 파일을 정말 삭제하시겠습니까?`)) return;

    try {
      // 이것도 파일 삭제 http delete 요청하고 기다려서 결과 받기...
      const response = await fetch(`/api/data/file/${fileName}`, { method: 'DELETE' });
      const result = await response.json();

      // 성공이면 목록 갱신 아니면 에러 표시
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
