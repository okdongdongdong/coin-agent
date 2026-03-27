# 에이전트 경쟁 규칙

## 에이전트 목록
| ID | 전략 | 설명 |
|----|------|------|
| sma_agent | SMA 교차 | SMA(5)와 SMA(20) 교차 시 매매 신호 |
| momentum_agent | RSI 모멘텀 | RSI 과매수/과매도 기반 모멘텀 |
| mean_reversion_agent | 볼린저밴드 평균회귀 | 가격이 밴드 터치 시 반전 기대 |
| breakout_agent | 거래량 돌파 | 거래량 급증 + 가격 돌파 시 매매 |
| sentiment_agent | 감성분석 (Claude) | Claude Code가 캔들 데이터 분석 |
| pattern_agent | 패턴인식 (Claude) | Claude Code가 차트 패턴 인식 |

## 경쟁 메커니즘

### 생명주기
1. **워밍업**: 최초 등록 후 10거래 완료 전까지. 균등 자본 배정.
2. **활성**: 점수 기반 경쟁. 성과에 따라 자본 배분 변동.
3. **벤치**: 점수 30/100 이하 시 자본 배분 0. 시그널은 계속 추적.
4. **은퇴**: 수동으로 비활성화된 에이전트.

### 복합점수 (0~100)
| 지표 | 가중치 | 범위 매핑 |
|------|--------|----------|
| Sharpe Ratio | 30% | [-2, 3] → [0, 100] |
| Win Rate | 20% | [0%, 100%] → [0, 100] |
| Profit Factor | 20% | [0, 3] → [0, 100] |
| Max Drawdown | 20% | [50%, 0%] → [0, 100] (역수) |
| Consistency | 10% | 연속 수익 비율 |

### 자본배분 규칙
- 알고리즘: Temperature-Scaled Softmax
- 기본 Temperature: 2.0 (균등 분산형)
- 에이전트당 최소 배분: 5%
- 에이전트당 최대 배분: 40%
- 최소 활성 에이전트: 2개
- 리밸런싱 주기: 50틱

## Claude Code 에이전트 가이드

### 감성분석 (sentiment_agent)
report 데이터를 읽고 다음을 판단:
- 캔들의 전체적 흐름 (상승세/하락세/횡보)
- 거래량 변화 추세 (증가/감소/급증)
- 매수/매도 압력의 균형

결과 형식:
```json
{"sentiment": {"action": "buy|sell|hold", "confidence": 0.0-1.0, "reason": "..."}}
```

### 패턴인식 (pattern_agent)
report의 캔들 데이터를 읽고 다음을 판단:
- 캔들스틱 패턴 (망치형, 장악형, 도지, 별형 등)
- 지지/저항선 수준
- 추세 패턴 (상승 삼각형, 하락 쐐기 등)

결과 형식:
```json
{"pattern": {"action": "buy|sell|hold", "confidence": 0.0-1.0, "reason": "..."}}
```

### 통합 결정
```bash
python -m coin_agent decide \
  --claude-signals '{"sentiment": {"action": "buy", "confidence": 0.7}, "pattern": {"action": "buy", "confidence": 0.6}}' \
  --meta-action buy \
  --execute
```
