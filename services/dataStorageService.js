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

    const formattedSymbol = symbol.toLowerCase();
    const fileName = `${formattedSymbol}_${interval}_${dataset.totalCount}bars.json`;
    const filePath = path.join(DATA_DIR, fileName);

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

    await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf-8');
    return { fileName, meta: payload.meta };
  }

  /**
   * 저장된 데이터 파일 목록 조회
   */
  async listDataFiles() {
    await this.ensureDataDir();
    const files = await fs.readdir(DATA_DIR);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    const fileDetails = await Promise.all(
      jsonFiles.map(async (fileName) => {
        const filePath = path.join(DATA_DIR, fileName);
        const stats = await fs.stat(filePath);
        try {
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
