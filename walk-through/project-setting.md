# 프로젝트 환경 설정 가이드 (Project Setting Guide)

본 문서는 `cabt_drl` (Crypto Asset BackTester - Deep Reinforcement Learning) 프로젝트의 기본 설정 파일들(.env, package.json, package-lock.json)의 역할과 구성 요소를 정리한 설명서입니다.

---

## 1. `.env` (환경 변수 설정 파일)

### 역할
코드 내부에 포트 번호, 외부 API 주소, API 키 등을 직접 하드코딩하는 것을 방지하고, **환경 변수(Environment Variables)**를 독립적으로 관리하기 위한 설정 파일입니다.

### 파일 내용 예시
```ini
PORT=5000
BINANCE_FUTURES_API_URL=https://fapi.binance.com
```

### 특징 및 주의사항
- **보안 및 환경 분리**: GitHub 등의 저장소에 비밀키가 노출되는 것을 방지합니다 (`.gitignore`에 등록되어 관리됨).
- **사용 방법**: Node.js 코드 내에서 `dotenv` 라이브러리를 통해 `process.env.PORT` 형태로 호출하여 사용합니다.

---

## 2. `package.json` (Node.js 프로젝트 메인 명세서)

### 역할
프로젝트의 이름, 버전, 실행 스크립트, 저작자 정보, 그리고 설치된 외부 라이브러리(의존성) 목록을 관리하는 **프로젝트 명세서**입니다.

### 주요 항목 명세

```json
{
  "name": "cabt_drl",
  "version": "1.0.0",
  "description": "\"# crypto-asset-back-tester-drl\"",
  "main": "index.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [
    "crypto",
    "binance",
    "bitcoin",
    "backtest",
    "reinforcement-learning",
    "drl",
    "tradingview"
  ],
  "author": {
    "name": "E.K. Yi",
    "email": "wolf@teoal.net",
    "url": "https://github.com/isltod"
  },
  "license": "ISC",
  "dependencies": {
    "axios": "^1.19.0",
    "cors": "^2.8.6",
    "dotenv": "^17.4.2",
    "express": "^5.2.1"
  },
  "devDependencies": {
    "nodemon": "^3.1.14"
  }
}
```

### 각 항목별 설명

1. **`scripts`**: 터미널에서 자주 사용하는 실행 명령어를 단축키로 등록합니다.
   - `npm start` ➡️ `node server.js` (서버 1회 실행)
   - `npm run dev` ➡️ `nodemon server.js` (코드 수정 시 자동 재시작되는 개발용 서버 실행)
2. **`keywords`**: 프로젝트의 주요 분야나 사용 기술을 대표하는 검색 태그 배열입니다.
3. **`author`**: 작성자(개발자)의 이름, 이메일, 웹사이트/저장소 주소를 포함하는 객체입니다.
4. **`license`**: 프로젝트 라이선스입니다. 기본값인 `ISC`는 자유로운 복제, 수정, 배포, 상업적 이용을 허용하는 허용적 오픈소스 라이선스입니다.
5. **`dependencies` (프로덕션 의존성)**:
   - `express`: Node.js 웹 서버 구축 프레임워크
   - `axios`: 바이낸스 API 호출용 HTTP 클라이언트
   - `cors`: 웹 브라우저 GUI 차트와의 통신을 위한 크로스 오리진 허용 미들웨어
   - `dotenv`: `.env` 파일 읽기 라이브러리
6. **`devDependencies` (개발용 의존성)**:
   - `nodemon`: 소스코드 수정 시 서버를 자동으로 재시작해 주는 개발 보조 도구

---

## 3. `package-lock.json` (의존성 버전 고정 잠금 파일)

### 역할
`package.json`에 정의된 외부 라이브러리 및 그 라이브러리가 사용하는 수많은 하위 서브 라이브러리들의 **정확한 버전과 해시(Integrity) 정보**를 보관하는 파일입니다.

### 특징 및 수칙
- **개발자 직접 수정 금지**: `npm install`, `npm update` 등에 의해 `npm` 도구가 자동으로 관리하므로 사람이 직접 수정하지 않습니다.
- **재현성 보장**: 이 파일이 존재함으로써 어떤 환경(다른 개발자 컴퓨터, 운영 서버 등)에서 `npm install`을 실행하더라도 100% 동일한 패키지 버전 환경을 구축할 수 있습니다.
- **Git 포함**: 직접 수정하진 않지만 Git 버전 관리에는 반드시 포함(commit)되어야 합니다.
