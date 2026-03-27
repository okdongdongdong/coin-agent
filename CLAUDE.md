# Coin-Agent: 멀티에이전트 경쟁 기반 코인 자동매매 시스템

## 프로젝트 개요
빗썸 거래소에서 Claude Code가 AI 두뇌 역할을 하는 자동매매 시스템.
4개의 기술적 서브에이전트가 수익률 경쟁을 통해 자본을 배분받는 구조.

## CLI 명령어
```bash
python -m coin_agent doctor    # API 연결 확인
python -m coin_agent tick      # 1틱 실행 (데이터 수집 + 에이전트 시그널)
python -m coin_agent report    # 분석 리포트 출력
python -m coin_agent decide    # Claude 시그널 주입 + 매매 실행
python -m coin_agent status    # 포트폴리오 상태
python -m coin_agent leaderboard  # 에이전트 랭킹
python -m coin_agent rebalance # 자본 재배분
python -m coin_agent run       # 자동 루프
python -m coin_agent stop      # 비상 정지
python -m coin_agent history   # 거래 이력
```

## 운영 플로우
1. `tick` → 데이터 수집 + 기술 에이전트 시그널 생성
2. `report` → Claude Code가 분석 리포트 읽기
3. Claude Code가 감성분석/패턴인식 수행
4. `decide` → Claude의 결정을 시스템에 반영
5. `status` → 결과 확인
6. 주기적으로 `leaderboard` 보고 `rebalance`

## 에이전트 목록
- **sma_agent**: 이동평균 교차 전략
- **momentum_agent**: RSI 기반 모멘텀
- **mean_reversion_agent**: 볼린저밴드 평균회귀
- **breakout_agent**: 거래량 돌파

## 지침 파일 (guidelines/)
- `trading_rules.md`: 거래 규칙, 포지션 한도
- `agent_behavior.md`: 에이전트 경쟁 규칙
- `market_conditions.md`: 시장 국면 정의
- `risk_management.md`: 리스크 관리 정책
- `emergency_protocol.md`: 비상 정지 프로토콜

## 기술 스택
- Python 3.11+ (외부 의존성 없음, stdlib only)
- 빗썸 REST API (JWT HS256 인증)
- JSONL + JSON 파일 기반 저장
- decimal.Decimal 금융 연산
