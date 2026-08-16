const fs = require('fs').promises;
const path = require('path');
const { DATA_DIR } = require('../config/constants');

/**
 * 수집된 데이터를 로컬 파일 시스템에 저장 및 관리하는 서비스
 */
class DataStorageService {
  constructor() {
    this.ensureDataDir();
  }

  /**
   * data 디렉터리가 없으면 자동 생성
   */
  async ensureDataDir() {
    try {
      // recursive로 있으면 넘어가고, 없으면 중간 디렉토리까지 다 만들기...
      await fs.mkdir(DATA_DIR, { recursive: true });
    } catch (err) {
      console.error('Data directory creation failed:', err);
    }
  }

  /**
   * 데이터를 JSON 파일로 저장
   */
  async saveData(symbol, interval, dataset) {
    // 생성자에서도 했는데, 저장할 때마다 확인하네...
    await this.ensureDataDir();

    // 파일 저장 경로 문자열로 만들고
    const formattedSymbol = symbol.toLowerCase();
    const fileName = `${formattedSymbol}_${interval}_${dataset.totalCount}bars.json`;
    const filePath = path.join(DATA_DIR, fileName);

    // 데이터 저장 형식을 meta와 data로 잡고,
    const payload = {
      meta: {
        symbol: dataset.symbol,
        interval: dataset.interval,
        requestedLimit: dataset.requestedLimit || null,
        totalCount: dataset.totalCount,
        isPartial: dataset.isPartial || false,
        warning: dataset.warning || null,
        startTime: dataset.startTime,
        endTime: dataset.endTime,
        startDate: dataset.startTime ? new Date(dataset.startTime).toISOString() : null,
        endDate: dataset.endTime ? new Date(dataset.endTime).toISOString() : null,
        createdAt: new Date().toISOString(),
      },
      data: dataset.data,
    };

    // 파일로 저장하고, 파일이름과 메타정보 반환
    await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf-8');
    return { fileName, meta: payload.meta };
  }

  /**
   * 저장된 데이터 파일 목록 조회
   */
  async listDataFiles() {
    // 데이터 디렉토리 확인해서 만드는 코드는 다 들어가네...
    await this.ensureDataDir();
    // 데이터 디렉토리에 있는 JSON 파일들만 받아서...
    const files = await fs.readdir(DATA_DIR);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    // 괄호로 묶인 부분을 Promise.all로 병렬 처리...await로 끝날 때 기다려 넘어가기...
    const fileDetails = await Promise.all(
      // 안에 화살표 함수는 그 안쪽에 await할 비동기 함수들 때문에 async로 선언되고...
      jsonFiles.map(async (fileName) => {
        // 파일 경로 문자열 만들고
        const filePath = path.join(DATA_DIR, fileName);
        // 크기와 수정시간 등 상태정보 읽고,
        const stats = await fs.stat(filePath);
        try {
          // 파일 내용을 읽는다? 메타 정보만 쓰는데 뭔가 비효율적으로 보이기도...
          const content = await fs.readFile(filePath, 'utf-8');
          const parsed = JSON.parse(content);
          return {
            fileName,
            sizeBytes: stats.size,
            updatedAt: stats.mtime,
            meta: parsed.meta || {},
          };
        } catch (e) {
          return {
            fileName,
            sizeBytes: stats.size,
            updatedAt: stats.mtime,
            meta: {},
          };
        }
      })
    );

    return fileDetails;
  }

  /**
   * 특정 데이터 파일 읽기
   */
  async readDataFile(fileName) {
    const safeFileName = path.basename(fileName);
    const filePath = path.join(DATA_DIR, safeFileName);
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  }

  /**
   * 특정 데이터 파일 삭제
   */
  async deleteDataFile(fileName) {
    const safeFileName = path.basename(fileName);
    const filePath = path.join(DATA_DIR, safeFileName);
    await fs.unlink(filePath);
    return { fileName: safeFileName, deleted: true };
  }
}

module.exports = new DataStorageService();
