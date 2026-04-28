# 건강차림 — Chat Interface

LLM × RAG 기반 개인 맞춤형 음식 챗봇의 대화 인터페이스입니다.

## ✨ 기능

이 페이지는 단일 HTML 파일로 동작하는 **데모 챗봇**입니다.

- **3-panel 레이아웃**: 좌측 대화 히스토리 · 가운데 채팅 · 우측 처리 과정 시각화
- **음식 점수 카드**: 0~100 점수와 색상 코딩 (그린/앰버/레드)
- **출처 인용 칩**: RAG 검색 결과를 의료 문서 단위로 표시
- **Function Call 라이브 패널**: `get_user_profile`, `search_documents` 등 호출 흐름 실시간 시각화
- **DB 조회 내역**: SQLite 쿼리 시뮬레이션
- **대화 히스토리**: localStorage에 자동 저장 (브라우저별)
- **빠른 답변**: bot 응답 후 추천 후속 질문 버튼 제공

## 🎯 시도해볼 수 있는 질문

내장된 mock 응답 엔진이 다음 키워드를 인식합니다:

| 키워드 | 점수 | 응답 카테고리 |
|---|---|---|
| 김밥 | 72 (warn) | 당뇨 적합도 |
| 짜장면 | 48 (danger) | 나트륨·지방 과다 |
| 떡볶이 | 42 (danger) | 당분·나트륨 |
| 라면 | 38 (danger) | 고혈압 위험 |
| 바나나 | 78 (warn) | 간식 GI |
| 샐러드 / 닭가슴살 | 91 (good) | 단백질 식단 |

다른 음식을 입력하면 기본 안내 메시지로 응답합니다.

## 🚀 GitHub Pages 배포

### 가장 빠른 방법

1. GitHub에 새 저장소 생성
2. `index.html`을 저장소 루트에 업로드
3. Settings → Pages → Source: `main` branch, `/ (root)` → Save
4. 1~2분 후 `https://<username>.github.io/<repo-name>/` 으로 접속

### `.nojekyll` 파일

GitHub Pages가 Jekyll로 처리하는 것을 방지하기 위해 빈 `.nojekyll` 파일을 함께 업로드하세요.

## 🔌 실제 백엔드 연결하기

`index.html` 하단의 `sendMessage()` 함수에서 mock 응답 부분을 실제 API 호출로 교체하세요.

```javascript
// 현재: setTimeout 체인으로 mock
async function sendMessage(text) {
  // ... user message 저장 ...
  
  // 실제 백엔드 연결 (예시)
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      message: text, 
      conversation_id: state.activeId 
    })
  });
  
  // SSE 스트리밍의 경우
  const reader = response.body.getReader();
  // ... process 패널 업데이트 + 메시지 스트리밍 ...
}
```

### 권장 SSE 이벤트 스키마

서버 → 클라이언트로 멀티플렉싱할 이벤트 타입:

```
event: function_call
data: { "name": "get_user_profile", "duration_ms": 420 }

event: rag_result
data: { "title": "당뇨식 가이드라인", "meta": "p.42", "score": 92 }

event: db_query
data: { "table": "food_log", "duration_ms": 95 }

event: token
data: { "text": "참치김밥은..." }

event: food_score
data: { "name": "김밥", "score": 72, "level": "warn", "detail": "..." }

event: done
data: {}
```

## 🎨 디자인 시스템

### 컬러 (CSS 변수)

```css
--sage: #2D9D78;        /* primary */
--sage-dark: #1F6E55;   /* primary dark */
--coral: #FF8A65;       /* user bubble, accent */
--cream: #FAFAF7;       /* background */
--ink: #1A2E2A;         /* text primary */
--amber: #F4A300;       /* warning, disclaimer */
```

### 폰트

- **Pretendard Variable** — 본문, UI
- **Noto Serif KR** — 헤드라인, 강조 (점수 숫자, 빈 상태 제목)
- **JetBrains Mono** — function call 이름, 라벨

## 📐 레이아웃 사양

| 요소 | 사이즈 |
|---|---|
| 좌측 사이드바 | 260px (1100px↓에서 220px) |
| 우측 처리 패널 | 320px (1100px↓에서 280px) |
| 채팅 max-width | 720px (가독성) |
| 상단 바 | 56px |

900px 이하에서는 사이드바와 처리 패널이 숨겨지고 채팅만 표시됩니다.

## 🔧 로컬 실행

```bash
# Python
python3 -m http.server 8000

# Node
npx http-server . -p 8000
```

`http://localhost:8000` 접속.

## 📝 라이선스

이 코드는 프로젝트 데모용입니다. 자유롭게 수정해서 사용하세요.
