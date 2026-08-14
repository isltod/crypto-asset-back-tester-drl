// 외부 라이브러리, JS 모듈 import
const express = require('express');                 // 1. 외부 npm 패키지
const cors = require('cors');                       // 1. 외부 npm 패키지
const path = require('path');                       // 2. Node.js 내장 모듈
const { PORT } = require('./config/constants');     // 3. 이 프로젝트 파일 모듈
const dataRoutes = require('./routes/dataRoutes');   // 3. 이 프로젝트 파일 모듈


const app = express();

// 미들웨어 설정
// 서버가 브라우저에 응답할 때 HTTP 헤더에 교차 출처 리소스 공유(Cross-Origin Resource Sharing)를 허용하도록 설정
app.use(cors());
// JSON 형식의 요청 본문을 파싱하기 위한 미들웨어
app.use(express.json());
// URL-encoded 형식의 요청 본문을 파싱하기 위한 미들웨어
app.use(express.urlencoded({ extended: true }));

// 정적 파일 제공 (웹 브라우저 GUI용)
app.use(express.static(path.join(__dirname, 'public')));

// API 라우트 등록
app.use('/api/data', dataRoutes);

// 서버 상태 헬스체크
app.use('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Crypto Asset BackTester DRL Server is running',
    timestamp: new Date().toISOString(),
  });
});

// 기본 인덱스 라우트 (웹 GUI 연동 대비)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 에러 핸들러
app.use((err, req, res, next) => {
  console.error('Unhandled Server Error:', err.stack);
  // 클라이언트에 상태코드를 500으로 설정하고 오류 메시지를 JSON 형태로 반환
  res.status(500).json({
    success: false,
    error: err.message || 'Internal Server Error',
  });
});

// 서버 시작
app.listen(PORT, () => {
  console.log(`===================================================`);
  console.log(` 🚀 CABT DRL Server is running on port ${PORT}`);
  console.log(` 📊 API Base: http://localhost:${PORT}/api/data`);
  console.log(` 🏥 Health Check: http://localhost:${PORT}/api/health`);
  console.log(`===================================================`);
});
