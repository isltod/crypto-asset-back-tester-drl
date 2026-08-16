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
  // 메서드 내부에서 비동기 함수를 사용하고 있으므로 async...
  async fetchAndSaveData(req, res) {
    try {
      // 1. html get 요청에서 ?로 매개변수 전달...
      const { symbol = 'BTCUSDT', interval = '1h', startTime, endTime, limit, save = 'true' } = req.query;

      // 2. 바이낸스에서 요청 데이터 받아오고...
      // fetchKlines가 비동기 함수인데, 그냥 넘어가면 안되니 await...
      const dataset = await binanceService.fetchKlines({
        // 매개변수 이름과 전달하는 변수 이름이 같으면 한 번만...
        symbol,
        interval,
        startTime,
        endTime,
        limit: limit ? parseInt(limit, 10) : undefined,
      });

      // 3. 저장 옵션이면 데이터 저장하고 반환...
      if (save === 'true' && dataset.totalCount > 0) {
        // saveData가 writeFile를 사용...이게 비동기 함수라서 await...
        const saveResult = await dataStorageService.saveData(symbol, interval, dataset);
        // 반환값들을 JSON 객체로 변환하고, http 응답 형식을 갖추고, 데이터 보내고, http 요청 종료까지...
        return res.json({
          success: true,
          message: `${dataset.totalCount}개의 캔들 데이터 수집 및 저장 완료`,
          savedFile: saveResult.fileName,
          meta: saveResult.meta,
          data: dataset.data,
        });
      }

      // 여기 왔다는 건 저장은 안하고 결과 반환한다는 얘기...
      return res.json({
        success: true,
        message: `${dataset.totalCount}개의 캔들 데이터 수집 완료 (저장 안함)`,
        meta: {
          symbol: dataset.symbol,
          interval: dataset.interval,
          requestedLimit: dataset.requestedLimit,
          totalCount: dataset.totalCount,
          isPartial: dataset.isPartial,
          warning: dataset.warning,
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

// 이게 싱글톤 패턴...
// 여기서 new로 생성해서 인스턴스를 내보내므로,
// 다른 모든 곳에서는 여기서 생성한 하나의 객체만을 사용하게 됨.
module.exports = new DataController();
