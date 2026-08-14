const binanceService = require('../services/binanceService');
const dataStorageService = require('../services/dataStorageService');

/**
 * 데이터 수집 및 관리 API 컨트롤러
 */
class DataController {
  /**
   * GET /api/data/fetch
   * 바이낸스 데이터 수집 및 자동 로컬 저장
   */
  async fetchAndSaveData(req, res) {
    try {
      const { symbol = 'BTCUSDT', interval = '1h', startTime, endTime, limit, save = 'true' } = req.query;

      const dataset = await binanceService.fetchKlines({
        symbol,
        interval,
        startTime,
        endTime,
        limit: limit ? parseInt(limit, 10) : undefined,
      });

      if (save === 'true' && dataset.totalCount > 0) {
        const saveResult = await dataStorageService.saveData(symbol, interval, dataset);
        return res.json({
          success: true,
          message: `${dataset.totalCount}개의 캔들 데이터 수집 및 저장 완료`,
          savedFile: saveResult.fileName,
          meta: saveResult.meta,
          data: dataset.data,
        });
      }

      return res.json({
        success: true,
        message: `${dataset.totalCount}개의 캔들 데이터 수집 완료 (저장 안함)`,
        meta: {
          symbol: dataset.symbol,
          interval: dataset.interval,
          totalCount: dataset.totalCount,
          startTime: dataset.startTime,
          endTime: dataset.endTime,
        },
        data: dataset.data,
      });
    } catch (error) {
      console.error('fetchAndSaveData Error:', error.message);
      return res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  /**
   * GET /api/data/list
   * 저장된 데이터 파일 목록 조회
   */
  async listFiles(req, res) {
    try {
      const files = await dataStorageService.listDataFiles();
      return res.json({
        success: true,
        total: files.length,
        files,
      });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  /**
   * GET /api/data/file/:fileName
   * 특정 데이터 파일 상세 및 OHLCV 데이터 조회
   */
  async getFileData(req, res) {
    try {
      const { fileName } = req.params;
      const fileData = await dataStorageService.readDataFile(fileName);
      return res.json({
        success: true,
        meta: fileData.meta,
        data: fileData.data,
      });
    } catch (error) {
      return res.status(404).json({
        success: false,
        error: `파일을 찾을 수 없거나 읽기 실패: ${error.message}`,
      });
    }
  }

  /**
   * DELETE /api/data/file/:fileName
   * 특정 데이터 파일 삭제
   */
  async deleteFile(req, res) {
    try {
      const { fileName } = req.params;
      const result = await dataStorageService.deleteDataFile(fileName);
      return res.json({
        success: true,
        message: `파일 삭제 완료: ${fileName}`,
        result,
      });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
}

module.exports = new DataController();
