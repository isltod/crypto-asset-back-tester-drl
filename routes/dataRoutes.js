// server.js에서 use로 '/api/data'로 시작하는 모든 요청을 여기로 보냄
const express = require('express');
const router = express.Router();
// 여기서는 다시 컨트롤러 등록해서 처리...
const dataController = require('../controllers/dataController');

// 바이낸스 Klines 데이터 수집 및 저장 (GET /api/data/fetch?symbol=BTCUSDT&interval=1h&limit=1000)
router.get('/fetch', dataController.fetchAndSaveData);

// 저장된 파일 목록 조회 (GET /api/data/list)
router.get('/list', dataController.listFiles);

// 특정 파일 데이터 조회 (GET /api/data/file/:fileName)
router.get('/file/:fileName', dataController.getFileData);

// 특정 파일 삭제 (DELETE /api/data/file/:fileName)
router.delete('/file/:fileName', dataController.deleteFile);

module.exports = router;
