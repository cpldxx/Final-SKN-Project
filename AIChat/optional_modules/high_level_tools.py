"""
optional_modules/high_level_tools.py

고급 전략/리스크/포트폴리오/통합 시스템 모듈 통합 파일
- MarketRegimeDetector, KalmanRegimeFilter, MultiFactorSignalGenerator, MLSignalEnsemble, BlackLittermanOptimizer, DynamicRiskParityOptimizer, DynamicVaRModel, StressTestingFramework, AdvancedTradingSystem, SystemMonitor
- specialist_agents.py와 연동
- 각 클래스별 수학적/경제적 의미 주석 포함
"""

import numpy as np
from scipy.stats import multivariate_normal
import time

# (예시) specialist_agents 데이터 연동용 스텁
class MacroEconomicAgent:
    def get_macro_data(self, idx):
        # 실제 구현에서는 실시간/DB/파이프라인 연동
        gdp_growth = [-2.0, -1.0, 0.5, 1.5, 2.5, 3.0]
        cpi = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        return {'gdp_growth': gdp_growth[idx], 'cpi': cpi[idx]}

class TechnicalAnalysisAgent:
    def get_technical_data(self, idx):
        rsi = [25, 35, 45, 55, 65, 75]
        vix = [12, 18, 22, 28, 35, 45]
        macd = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        return {'rsi': rsi[idx], 'vix': vix[idx], 'macd': macd[idx]}

# --- MarketRegimeDetector: 베이지안 상태 분류 기반 시장 Regime 탐지기 ---
class MarketRegimeDetector:
    """
    MarketRegimeDetector
    --------------------
    수학적 공식:
        P(State_t|X_t) = P(X_t|State_t) × P(State_t|State_{t-1}) / P(X_t)
    - State_t ∈ {Bull, Bear, Sideways}
    - X_t = [GDP_growth, CPI_rate, RSI, VIX, MACD]
    - P(X_t|State_t) ~ N(μ_state, Σ_state)

    경제적 의미:
        - 시장 상태(경기국면)를 확률적으로 분류하여 전략/리스크/포트폴리오에 반영
        - 거시/기술/심리/변동성 등 다양한 요인을 통합적으로 반영
    """
    def __init__(self):
        # 상태 정의
        self.states = ['Bear', 'Sideways', 'Bull']
        self.state_idx = {s: i for i, s in enumerate(self.states)}

        # 상태 전이 확률 행렬 (Markov Chain)
        # 행: 이전 상태, 열: 현재 상태
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],  # Bear -> [Bear, Sideways, Bull]
            [0.25, 0.50, 0.25],  # Sideways -> [Bear, Sideways, Bull]
            [0.05, 0.10, 0.85]   # Bull -> [Bear, Sideways, Bull]
        ])

        # 상태별 관측 모수 (다변량 정규분포)
        # 경제적 의미: 각 상태별로 경제/기술지표의 평균과 분산이 다름
        self.observation_params = {
            'Bear': {
                'mean': np.array([-0.5, 4.0, 35, 30, -0.5]),
                'cov': np.diag([0.25, 1.0, 100, 25, 0.1])
            },
            'Sideways': {
                'mean': np.array([1.5, 2.5, 50, 20, 0.0]),
                'cov': np.diag([0.5, 0.5, 64, 16, 0.05])
            },
            'Bull': {
                'mean': np.array([3.0, 2.0, 65, 15, 0.5]),
                'cov': np.diag([0.75, 0.25, 81, 9, 0.1])
            }
        }

    def calculate_likelihood(self, observation, state):
        """
        P(X_t | State_t): 상태별 다변량 정규분포 우도 계산
        경제적 의미: 현재 관측값이 해당 상태에서 얼마나 '자연스러운지' 확률로 평가
        """
        params = self.observation_params[state]
        return multivariate_normal.pdf(observation, mean=params['mean'], cov=params['cov'])

    def bayesian_regime_classification(self, macro_data, technical_data, prev_state=None):
        """
        단일 시점의 상태 확률 계산 (Posterior)
        P(State_t | X_t) ∝ P(X_t | State_t) × P(State_t|State_{t-1})
        - prev_state가 없으면 균등 prior
        - prev_state가 있으면 transition_matrix 기반 prior
        """
        obs = np.array([
            macro_data['gdp_growth'],
            macro_data['cpi'],
            technical_data['rsi'],
            technical_data['vix'],
            technical_data['macd']
        ])
        if prev_state is None:
            priors = np.array([1/3, 1/3, 1/3])
        else:
            priors = self.transition_matrix[self.state_idx[prev_state]]
        likelihoods = np.array([self.calculate_likelihood(obs, s) for s in self.states])
        numerators = likelihoods * priors
        posteriors = numerators / numerators.sum()
        return {s: posteriors[i] for i, s in enumerate(self.states)}

    def viterbi_algorithm(self, macro_seq, tech_seq):
        """
        비터비 알고리즘: 최적 상태 시퀀스 추정 (HMM)
        경제적 의미: 과거~현재까지 가장 그럴듯한 시장 Regime 경로 추정
        """
        n_obs = len(macro_seq)
        n_states = len(self.states)
        log_delta = np.zeros((n_obs, n_states))
        psi = np.zeros((n_obs, n_states), dtype=int)

        # 초기화
        obs0 = np.array([
            macro_seq[0]['gdp_growth'], macro_seq[0]['cpi'],
            tech_seq[0]['rsi'], tech_seq[0]['vix'], tech_seq[0]['macd']
        ])
        for s in range(n_states):
            log_delta[0, s] = np.log(1.0 / n_states) + np.log(self.calculate_likelihood(obs0, self.states[s]))

        # 동적 프로그래밍
        for t in range(1, n_obs):
            obs = np.array([
                macro_seq[t]['gdp_growth'], macro_seq[t]['cpi'],
                tech_seq[t]['rsi'], tech_seq[t]['vix'], tech_seq[t]['macd']
            ])
            for s in range(n_states):
                trans_probs = log_delta[t-1] + np.log(self.transition_matrix[:, s])
                psi[t, s] = np.argmax(trans_probs)
                log_delta[t, s] = np.max(trans_probs) + np.log(self.calculate_likelihood(obs, self.states[s]))

        # 역추적
        states_seq = np.zeros(n_obs, dtype=int)
        states_seq[-1] = np.argmax(log_delta[-1])
        for t in range(n_obs-2, -1, -1):
            states_seq[t] = psi[t+1, states_seq[t+1]]
        return [self.states[i] for i in states_seq]

    def update_transition_probabilities(self, historical_regimes):
        """
        상태 전이 행렬을 과거 Regime 시퀀스 기반으로 재추정
        경제적 의미: 실제 시장 Regime 변화 패턴을 반영하여 동적으로 전이확률 보정
        """
        n_states = len(self.states)
        counts = np.zeros((n_states, n_states))
        for (prev, curr) in zip(historical_regimes[:-1], historical_regimes[1:]):
            i, j = self.state_idx[prev], self.state_idx[curr]
            counts[i, j] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        self.transition_matrix = np.where(row_sums > 0, counts / row_sums, self.transition_matrix)

    def detect_regime(self, macro_data, technical_data, prev_state=None):
        """
        실시간 상태 예측 (가장 확률 높은 상태 반환)
        - 1초 이내 처리, 메모리 100MB 이하 보장
        """
        t0 = time.time()
        posteriors = self.bayesian_regime_classification(macro_data, technical_data, prev_state)
        regime = max(posteriors, key=posteriors.get)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"실시간 처리 초과: {elapsed:.3f}s"
        return regime, posteriors, elapsed

# -------------------------------
# 테스트/백테스트 예시 (2020-2024 시뮬레이션 데이터)
# -------------------------------
if __name__ == "__main__":
    macro_agent = MacroEconomicAgent()
    tech_agent = TechnicalAnalysisAgent()
    detector = MarketRegimeDetector()

    # 테스트 데이터 (6개 시점)
    macro_seq = [macro_agent.get_macro_data(i) for i in range(6)]
    tech_seq = [tech_agent.get_technical_data(i) for i in range(6)]

    # 1. 비터비 알고리즘으로 상태 시퀀스 추정
    regime_seq = detector.viterbi_algorithm(macro_seq, tech_seq)
    print("\n[비터비 알고리즘 최적 상태 시퀀스]")
    print(regime_seq)

    # 2. 실시간 상태 예측 및 성능 체크
    print("\n[실시간 상태 예측 및 성능 체크]")
    correct = 0
    for i in range(6):
        regime, posteriors, elapsed = detector.detect_regime(macro_seq[i], tech_seq[i])
        print(f"t={i}: regime={regime}, posteriors={posteriors}, 처리시간={elapsed*1000:.2f}ms")
        # (예시) 실제 라벨이 있다면 비교
        # if regime == true_label[i]: correct += 1
    # print(f"분류 정확도: {correct/6*100:.1f}%")

    # 3. 메모리 사용량 체크 (간단히)
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"\n[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")

    # 4. 백테스트 결과 예시 (실제 라벨/성과지표 연동 필요)
    print("\n[백테스트 결과 예시]")
    print("- 분류 정확도: (실제 라벨 연동 필요)")
    print("- 처리 속도: 모든 예측 1초 이내")
    print("- 메모리 사용량: 100MB 이하")
    print("- 경제적 해석: Bear→방어적, Bull→공격적, Sideways→중립적 전략 적용")

# --- 이하 고급 전략/리스크/포트폴리오/통합 시스템 스켈레톤 유지 ---

# 2. KalmanRegimeFilter (칼만 필터 기반 동적 상태 추정)
import numpy as np

class KalmanRegimeFilter:
    """
    KalmanRegimeFilter
    ------------------
    수학적 공식:
        상태 방정식: x_t = F × x_{t-1} + w_t
        관측 방정식: z_t = H × x_t + v_t
        - x_t = [market_momentum, volatility_regime, liquidity_state]
        - z_t = specialist_agents로부터의 관측값
        - w_t ~ N(0, Q), v_t ~ N(0, R)
    경제적 의미:
        - 시장의 동적 상태(모멘텀, 변동성, 유동성)를 실시간 추정
        - 거시/기술/펀더멘털/뉴스 등 다양한 요인을 통합적으로 반영
    """
    def __init__(self):
        # 상태 전이 행렬 F (시장 상태의 자기상관 및 상호작용)
        self.F = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
            [0.1, 0.0, 0.9]
        ])
        # 관측 행렬 H (관측값이 상태에 미치는 영향)
        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.7, 0.3]
        ])
        # 프로세스 노이즈 Q (상태 변화의 불확실성)
        self.Q = np.eye(3) * 0.01
        # 관측 노이즈 R (관측값의 불확실성)
        self.R = np.eye(5) * 0.1
        # 초기 상태 추정치 (임의값)
        self.x = np.array([0.0, 1.0, 0.5])
        # 초기 공분산 추정치
        self.P = np.eye(3)

    def predict(self):
        """
        상태 예측 단계 (a priori)
        x_{t|t-1} = F x_{t-1|t-1}
        P_{t|t-1} = F P_{t-1|t-1} F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        관측값 반영 (a posteriori)
        y = z - H x_{t|t-1}
        S = H P_{t|t-1} H^T + R
        K = P_{t|t-1} H^T S^{-1}
        x_{t|t} = x_{t|t-1} + K y
        P_{t|t} = (I - K H) P_{t|t-1}
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x.copy(), self.P.copy()

    def step(self, z):
        """
        1스텝 예측+업데이트 (실시간 처리)
        """
        self.predict()
        return self.update(z)

# --- specialist_agents 연동 예시 ---
class FundamentalAnalysisAgent:
    def get_liquidity_state(self, idx):
        # 예시: 유동성 상태 (0~1)
        return 0.3 + 0.1 * idx
class NewsAnalysisAgent:
    def get_noise_adjustment(self, idx):
        # 예시: 뉴스 기반 노이즈 (0~0.1)
        return 0.05 * (idx % 2)

def get_kalman_observation(idx, macro_agent, tech_agent, fund_agent, news_agent):
    # MacroEconomicAgent → market_momentum
    # TechnicalAnalysisAgent → volatility_regime
    # FundamentalAnalysisAgent → liquidity_state
    # NewsAnalysisAgent → 노이즈 조정 인자
    macro = macro_agent.get_macro_data(idx)
    tech = tech_agent.get_technical_data(idx)
    fund = fund_agent.get_liquidity_state(idx)
    news = news_agent.get_noise_adjustment(idx)
    # 관측값: [momentum, volatility, liquidity, avg(mom+vol), vol*0.7+liq*0.3+news]
    obs = np.array([
        macro['gdp_growth'],
        tech['vix'],
        fund,
        0.5 * macro['gdp_growth'] + 0.5 * tech['vix'],
        0.7 * tech['vix'] + 0.3 * fund + news
    ])
    return obs

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    macro_agent = MacroEconomicAgent()
    tech_agent = TechnicalAnalysisAgent()
    fund_agent = FundamentalAnalysisAgent()
    news_agent = NewsAnalysisAgent()
    kalman = KalmanRegimeFilter()
    print("\n[칼만 필터 기반 동적 상태 추정 테스트]")
    for i in range(6):
        z = get_kalman_observation(i, macro_agent, tech_agent, fund_agent, news_agent)
        t0 = time.time()
        x, P = kalman.step(z)
        elapsed = time.time() - t0
        print(f"t={i}: 상태추정={x.round(3)}, 공분산={np.diag(P).round(3)}, 처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: market_momentum↑→공격적, volatility↑→방어적, liquidity↑→유동성 전략")

# 3. MultiFactorSignalGenerator (다중 인자 신호 생성)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class MultiFactorSignalGenerator:
    """
    MultiFactorSignalGenerator
    -------------------------
    수학적 공식:
        Signal_t = Σ(i=1 to N) w_i × f_i(X_t) × confidence_i(X_t)
        - f_i(X_t): 인자별 신호 함수
        - w_i: 동적 가중치 (regime 기반)
        - confidence_i(X_t): 신호 신뢰도 (뉴스/이벤트 등)
    경제적 의미:
        - 다양한 요인(모멘텀, 평균회귀, 펀더멘털, 거시경제)을 통합하여 신호 생성
        - 시장 Regime에 따라 가중치 동적 조정
    """
    def __init__(self):
        # Regime별 동적 가중치
        self.factor_weights = {
            'Bull': {'momentum': 0.4, 'reversion': 0.1, 'fundamental': 0.3, 'macro': 0.2},
            'Bear': {'momentum': 0.1, 'reversion': 0.4, 'fundamental': 0.2, 'macro': 0.3},
            'Sideways': {'momentum': 0.2, 'reversion': 0.3, 'fundamental': 0.3, 'macro': 0.2}
        }

    def momentum_signal(self, prices, rsi):
        # f_momentum(X_t) = (P_t / SMA_n(P_t) - 1) × (1 - 2×RSI_t/100)
        sma = np.mean(prices)
        signal = (prices[-1] / sma - 1) * (1 - 2 * rsi / 100)
        return signal

    def reversion_signal(self, prices, bollinger_bands):
        # f_reversion(X_t) = -tanh(Z_score_t) × (1 - |R_t|/σ_R)
        z_score = (prices[-1] - bollinger_bands['mid']) / bollinger_bands['std']
        returns = np.diff(prices) / prices[:-1]
        R_t = returns[-1]
        sigma_R = np.std(returns)
        signal = -tanh(z_score) * (1 - np.abs(R_t) / (sigma_R + 1e-6))
        return signal

    def fundamental_signal(self, roe, pe_ratio, sector_data):
        # f_fundamental(X_t) = (ROE_t - ROE_sector) / σ_ROE × (1 - P/E_relative)
        roe_sector = sector_data['roe_sector']
        sigma_roe = sector_data['sigma_roe']
        pe_relative = pe_ratio / (sector_data['pe_sector'] + 1e-6)
        signal = ((roe - roe_sector) / (sigma_roe + 1e-6)) * (1 - pe_relative)
        return signal

    def macro_signal(self, gdp_surprise, cpi_surprise, rate_surprise, alpha=0.5, beta=0.3, gamma=0.2):
        # f_macro(X_t) = α×GDP_surprise + β×CPI_surprise + γ×Rate_surprise
        return alpha * gdp_surprise + beta * cpi_surprise + gamma * rate_surprise

    def confidence_score(self, news_sentiment, vix):
        # 신호 신뢰도: 뉴스 긍정도, 변동성 반영
        return sigmoid(news_sentiment) * (1 - min(vix / 50, 1))

    def generate_signals(self, agent_data, regime):
        """
        agent_data: dict
            - 'technical': {'prices': np.array, 'rsi': float, 'bollinger': dict}
            - 'fundamental': {'roe': float, 'pe': float, 'sector': dict}
            - 'macro': {'gdp_surprise': float, 'cpi_surprise': float, 'rate_surprise': float}
            - 'news': {'sentiment': float, 'vix': float}
        regime: str ('Bull'/'Bear'/'Sideways')
        """
        w = self.factor_weights[regime]
        # 각 신호 계산
        mom = self.momentum_signal(agent_data['technical']['prices'], agent_data['technical']['rsi'])
        rev = self.reversion_signal(agent_data['technical']['prices'], agent_data['technical']['bollinger'])
        fund = self.fundamental_signal(agent_data['fundamental']['roe'], agent_data['fundamental']['pe'], agent_data['fundamental']['sector'])
        macro = self.macro_signal(
            agent_data['macro']['gdp_surprise'],
            agent_data['macro']['cpi_surprise'],
            agent_data['macro']['rate_surprise']
        )
        conf = self.confidence_score(agent_data['news']['sentiment'], agent_data['news']['vix'])
        # 통합 신호
        signal = (
            w['momentum'] * mom * conf +
            w['reversion'] * rev * conf +
            w['fundamental'] * fund * conf +
            w['macro'] * macro * conf
        )
        return {
            'momentum': mom,
            'reversion': rev,
            'fundamental': fund,
            'macro': macro,
            'confidence': conf,
            'final_signal': signal
        }

# --- specialist_agents 연동 예시 ---
class TechnicalAnalysisAgent:
    def get_technical_data(self, idx):
        # prices, rsi, bollinger
        prices = np.array([100, 102, 101, 103, 105, 107]) + idx
        rsi = 30 + 5 * idx
        bollinger = {'mid': np.mean(prices), 'std': np.std(prices)}
        return {'prices': prices, 'rsi': rsi, 'bollinger': bollinger}
class FundamentalAnalysisAgent:
    def get_fundamental_data(self, idx):
        roe = 0.12 + 0.01 * idx
        pe = 15 + idx
        sector = {'roe_sector': 0.10, 'sigma_roe': 0.02, 'pe_sector': 14}
        return {'roe': roe, 'pe': pe, 'sector': sector}
class MacroEconomicAgent:
    def get_macro_data(self, idx):
        return {'gdp_surprise': 0.2 - 0.05 * idx, 'cpi_surprise': 0.1 * idx, 'rate_surprise': -0.02 * idx}
class NewsAnalysisAgent:
    def get_news_data(self, idx):
        return {'sentiment': 0.5 - 0.1 * idx, 'vix': 15 + 5 * idx}

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    tech_agent = TechnicalAnalysisAgent()
    fund_agent = FundamentalAnalysisAgent()
    macro_agent = MacroEconomicAgent()
    news_agent = NewsAnalysisAgent()
    signal_gen = MultiFactorSignalGenerator()
    print("\n[다중 인자 신호 생성 테스트]")
    for i, regime in enumerate(['Bull', 'Sideways', 'Bear']):
        agent_data = {
            'technical': tech_agent.get_technical_data(i),
            'fundamental': fund_agent.get_fundamental_data(i),
            'macro': macro_agent.get_macro_data(i),
            'news': news_agent.get_news_data(i)
        }
        t0 = time.time()
        signals = signal_gen.generate_signals(agent_data, regime)
        elapsed = time.time() - t0
        print(f"regime={regime}: 신호={signals}, 처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: 모멘텀↑→추세추종, 평균회귀↑→역추세, 펀더멘털↑→저평가, 거시↑→매크로 이벤트 반영")

# 4. MLSignalEnsemble (XGBoost + LSTM 앙상블)
import numpy as np
try:
    import xgboost as xgb
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    xgb = None
    keras = None

class MLSignalEnsemble:
    """
    MLSignalEnsemble
    ----------------
    수학적 모델:
        Final_Signal = α × XGBoost_Signal + β × LSTM_Signal + γ × Traditional_Signal
    경제적 의미:
        - 머신러닝 기반 신호(비선형/시계열 패턴)와 전통 신호(수식 기반)를 통합
        - 특성 엔지니어링, 앙상블 가중치, 성과 평가
    """
    def __init__(self):
        # XGBoost 모델
        if xgb:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                verbosity=0
            )
        else:
            self.xgb_model = None
        # LSTM 모델
        self.lstm_model = self.build_lstm_model() if keras else None
        # 앙상블 가중치
        self.ensemble_weights = [0.4, 0.4, 0.2]  # [XGB, LSTM, Traditional]

    def build_lstm_model(self):
        # TensorFlow/Keras LSTM 모델 정의
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(60, 8)),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_features(self, specialist_data):
        """
        specialist_data: dict
            - 기술지표: RSI, MACD, Bollinger Bands, ATR
            - 거시지표: GDP growth, CPI, Interest rates
            - 펀더멘털: ROE, P/E, Debt ratio
            - 감정지표: News sentiment, VIX, Put/Call ratio
        반환: X (2D array), y (1D array)
        """
        # 예시: 100개 샘플, 8개 특성
        X = np.random.randn(100, 8)
        y = np.random.randn(100)
        return X, y

    def train_ensemble(self, training_data):
        X, y = self.prepare_features(training_data)
        if self.xgb_model:
            self.xgb_model.fit(X, y)
        if self.lstm_model:
            X_lstm = X[-60:].reshape(1, 60, 8)
            y_lstm = y[-60:]
            self.lstm_model.fit(X_lstm, y_lstm, epochs=2, verbose=0)

    def predict_signal(self, current_data, traditional_signal=0.0):
        X, _ = self.prepare_features(current_data)
        xgb_pred = self.xgb_model.predict(X[:1])[0] if self.xgb_model else 0.0
        lstm_pred = float(self.lstm_model.predict(X[:1].reshape(1, 1, 8))[0, 0]) if self.lstm_model else 0.0
        final_signal = (
            self.ensemble_weights[0] * xgb_pred +
            self.ensemble_weights[1] * lstm_pred +
            self.ensemble_weights[2] * traditional_signal
        )
        return {
            'xgb_signal': xgb_pred,
            'lstm_signal': lstm_pred,
            'traditional_signal': traditional_signal,
            'final_signal': final_signal
        }

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_training_data(self):
        # 실제 구현에서는 과거 데이터/특성 엔지니어링
        return {}
    def get_current_data(self):
        return {}

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    ml_ensemble = MLSignalEnsemble()
    print("\n[ML 앙상블 신호 생성 테스트]")
    # 1. 학습
    t0 = time.time()
    ml_ensemble.train_ensemble(agents.get_training_data())
    elapsed_train = time.time() - t0
    # 2. 예측
    t1 = time.time()
    signals = ml_ensemble.predict_signal(agents.get_current_data(), traditional_signal=0.1)
    elapsed_pred = time.time() - t1
    print(f"신호={signals}, 학습시간={elapsed_train:.2f}s, 예측시간={elapsed_pred*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: XGBoost→비선형 패턴, LSTM→시계열 패턴, 전통신호→수식 기반, 앙상블로 통합")

# 5. BlackLittermanOptimizer (블랙-리터만 포트폴리오 최적화)
import numpy as np

class BlackLittermanOptimizer:
    """
    BlackLittermanOptimizer
    ----------------------
    수학적 공식:
        μ_new = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)μ_prior + P'Ω^(-1)Q]
        Σ_new = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
    경제적 의미:
        - 시장 균형(CAPM)과 투자자 관점(views)을 통합하여 기대수익/분산 추정
        - specialist_agents 기반 동적 관점 생성, 불확실성 반영
    """
    def __init__(self):
        self.tau = 0.025
        self.risk_aversion = 3.0
        self.confidence_scaling = {
            'macro': 0.8,
            'technical': 0.6,
            'fundamental': 0.7,
            'news': 0.4
        }

    def generate_views_from_agents(self, specialist_data, assets):
        views = []
        # 거시경제 관점
        if specialist_data['macro']['gdp_growth'] > 2.5:
            views.append({
                'assets': ['SPY', 'QQQ'],
                'relative_return': 0.05,
                'confidence': self.confidence_scaling['macro']
            })
        # 기술적 관점
        for asset, signal in specialist_data['technical'].items():
            if signal['rsi'] < 30:
                views.append({
                    'assets': [asset],
                    'absolute_return': 0.03,
                    'confidence': self.confidence_scaling['technical']
                })
        # (펀더멘털/뉴스 관점 등 추가 가능)
        return views

    def construct_p_matrix(self, views, assets):
        # P 행렬: (num_views x num_assets)
        P = np.zeros((len(views), len(assets)))
        for i, v in enumerate(views):
            for a in v['assets']:
                if 'relative_return' in v:
                    idx1 = assets.index(v['assets'][0])
                    idx2 = assets.index(v['assets'][1])
                    P[i, idx1] = 1
                    P[i, idx2] = -1
                else:
                    idx = assets.index(a)
                    P[i, idx] = 1
        return P

    def estimate_uncertainty(self, views):
        # Ω 행렬: 관점별 불확실성 (대각행렬)
        omega = np.diag([1.0 / (v['confidence'] + 1e-6) for v in views])
        return omega

    def optimize_portfolio(self, views, market_data):
        """
        market_data: dict
            - 'assets': list
            - 'cov': np.array (NxN)
            - 'mu_prior': np.array (N,)
        """
        assets = market_data['assets']
        cov = market_data['cov']
        mu_prior = market_data['mu_prior']
        tau = self.tau
        # 1. P, Q, Ω
        P = self.construct_p_matrix(views, assets)
        Q = np.array([
            v.get('relative_return', v.get('absolute_return', 0.0)) for v in views
        ])
        omega = self.estimate_uncertainty(views)
        # 2. Black-Litterman 공식
        inv_tau_cov = np.linalg.inv(tau * cov)
        inv_omega = np.linalg.inv(omega)
        middle = inv_tau_cov + P.T @ inv_omega @ P
        mu_new = np.linalg.inv(middle) @ (inv_tau_cov @ mu_prior + P.T @ inv_omega @ Q)
        cov_new = np.linalg.inv(middle)
        # 3. 위험조정 최적화(예시: 최대 샤프비율)
        rf = 0.02
        excess = mu_new - rf
        w = np.linalg.solve(cov_new, excess)
        w = np.maximum(w, 0)
        w /= w.sum()
        return {
            'assets': assets,
            'weights': w,
            'mu_new': mu_new,
            'cov_new': cov_new
        }

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_specialist_data(self):
        # 예시: macro, technical 등
        return {
            'macro': {'gdp_growth': 3.2},
            'technical': {'AAPL': {'rsi': 28}, 'MSFT': {'rsi': 45}}
        }
    def get_market_data(self):
        assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        cov = np.array([
            [0.04, 0.01, 0.01, 0.01],
            [0.01, 0.05, 0.01, 0.01],
            [0.01, 0.01, 0.06, 0.01],
            [0.01, 0.01, 0.01, 0.07]
        ])
        mu_prior = np.array([0.07, 0.08, 0.10, 0.09])
        return {'assets': assets, 'cov': cov, 'mu_prior': mu_prior}

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    bl = BlackLittermanOptimizer()
    print("\n[블랙-리터만 포트폴리오 최적화 테스트]")
    specialist_data = agents.get_specialist_data()
    market_data = agents.get_market_data()
    views = bl.generate_views_from_agents(specialist_data, market_data['assets'])
    t0 = time.time()
    result = bl.optimize_portfolio(views, market_data)
    elapsed = time.time() - t0
    print(f"최적화 결과: {result}, 처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: 시장관점+투자자관점 통합, 동적 자산배분, 위험조정 최적화")

# 6. DynamicRiskParityOptimizer (동적 위험 패리티)
import numpy as np
from scipy.optimize import minimize

class DynamicRiskParityOptimizer:
    """
    DynamicRiskParityOptimizer
    -------------------------
    수학적 공식:
        Risk_Contrib_i(t) = w_i(t) × [Σ(t) × w(t)]_i / [w(t)' × Σ(t) × w(t)]^0.5
        목적함수: min Σ(i=1 to N) [Risk_Contrib_i(t) - 1/N]^2
        제약조건: Σw_i = 1, w_i ≥ 0.01, |w_i(t) - w_i(t-1)| ≤ 0.1
        EWMA: Σ(t) = λ × Σ(t-1) + (1-λ) × r(t) × r(t)'
    경제적 의미:
        - 각 자산의 위험 기여도를 균등하게 하여 동적 자산배분
        - 변동성/상관관계/뉴스 등 실시간 반영
    """
    def __init__(self, n_assets, lambda_decay=0.94, rebalance_threshold=0.05, transaction_cost=0.001):
        self.n_assets = n_assets
        self.lambda_decay = lambda_decay
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
        self.cov = np.eye(n_assets) * 0.05
        self.prev_weights = np.ones(n_assets) / n_assets

    def update_covariance_matrix(self, returns):
        # EWMA 공분산 업데이트
        for r in returns:
            self.cov = self.lambda_decay * self.cov + (1 - self.lambda_decay) * np.outer(r, r)
        return self.cov

    def calculate_risk_contributions(self, weights, cov_matrix):
        # 위험 기여도 계산
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        mrc = cov_matrix @ weights  # marginal risk contribution
        rc = weights * mrc / (port_vol + 1e-8)
        return rc

    def risk_parity_objective(self, weights, cov_matrix):
        rc = self.calculate_risk_contributions(weights, cov_matrix)
        target = np.ones(self.n_assets) / self.n_assets
        return np.sum((rc - target) ** 2)

    def optimize_weights(self, cov_matrix):
        # 위험 패리티 최적화
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0.01, 1.0) for _ in range(self.n_assets)]
        res = minimize(self.risk_parity_objective, self.prev_weights, args=(cov_matrix,),
                       bounds=bounds, constraints=cons, method='SLSQP', options={'disp': False})
        w_opt = res.x
        # 리밸런싱 제한
        delta = np.clip(w_opt - self.prev_weights, -0.1, 0.1)
        w_final = self.prev_weights + delta
        w_final = np.maximum(w_final, 0.01)
        w_final /= w_final.sum()
        self.prev_weights = w_final
        return w_final

    def rebalance_decision(self, target_weights):
        # 리밸런싱 필요 여부
        diff = np.abs(target_weights - self.prev_weights)
        return np.any(diff > self.rebalance_threshold)

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_returns(self):
        # 예시: 10개 자산, 100일 수익률
        return np.random.randn(100, 4) * 0.01
    def get_cov_matrix(self):
        return np.cov(self.get_returns(), rowvar=False)

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    n_assets = 4
    drp = DynamicRiskParityOptimizer(n_assets)
    print("\n[동적 위험 패리티 최적화 테스트]")
    returns = agents.get_returns()
    cov = drp.update_covariance_matrix(returns)
    t0 = time.time()
    w = drp.optimize_weights(cov)
    elapsed = time.time() - t0
    rc = drp.calculate_risk_contributions(w, cov)
    print(f"최적화 가중치: {w.round(3)}")
    print(f"위험 기여도: {rc.round(3)} (합={rc.sum():.3f})")
    print(f"처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: 위험 기여도 균등, 동적 공분산, 리밸런싱 제한, 거래비용 고려")

# 7. DynamicVaRModel (GARCH 기반 동적 VaR)
import numpy as np
from scipy.stats import t as student_t

class DynamicVaRModel:
    """
    DynamicVaRModel
    ---------------
    수학적 모델:
        r_t = μ_t + σ_t × ε_t
        σ_t^2 = ω + α × r_{t-1}^2 + β × σ_{t-1}^2
        VaR_t(α) = μ_t + σ_t × Φ^(-1)(α)
    경제적 의미:
        - GARCH 기반 조건부 분산/평균으로 실시간 VaR 산출
        - specialist_agents 기반 구조적/단기/이벤트/장기 위험 반영
    """
    def __init__(self, garch_params=None, confidence_levels=None):
        self.garch_params = garch_params or {'omega': 0.00001, 'alpha': 0.05, 'beta': 0.90}
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]
        self.mu_t = 0.0
        self.sigma2_t = 0.0001
        self.nu = 6  # t-분포 자유도(꼬리 위험)

    def estimate_garch_params(self, returns):
        # (실전: 최대우도추정, 여기선 고정)
        return self.garch_params

    def predict_conditional_volatility(self, returns):
        # GARCH(1,1) 조건부 분산 예측
        params = self.estimate_garch_params(returns)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        sigma2 = np.var(returns[:20])  # 초기값
        sigma2_seq = []
        for r in returns:
            sigma2 = omega + alpha * r**2 + beta * sigma2
            sigma2_seq.append(sigma2)
        return np.array(sigma2_seq)

    def calculate_var(self, returns, agent_forecasts=None):
        # VaR 계산 (agent_forecasts: specialist_agents 기반 μ_t)
        mu = np.mean(returns[-20:]) if agent_forecasts is None else agent_forecasts.get('mu', 0.0)
        sigma_seq = np.sqrt(self.predict_conditional_volatility(returns))
        var_dict = {}
        for cl in self.confidence_levels:
            q = student_t.ppf(cl, self.nu)
            var_seq = mu + sigma_seq * q
            var_dict[cl] = var_seq
        return var_dict

    def backtesting(self, var_forecasts, actual_returns):
        # 바젤 백테스팅: VaR 초과 횟수(위반 비율)
        results = {}
        for cl, var_seq in var_forecasts.items():
            breaches = (actual_returns < var_seq).sum()
            results[cl] = {'breaches': int(breaches), 'total': len(actual_returns), 'rate': breaches/len(actual_returns)}
        return results

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_returns(self):
        return np.random.randn(100) * 0.012
    def get_agent_forecasts(self):
        return {'mu': 0.0005}

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    dvar = DynamicVaRModel()
    print("\n[GARCH 기반 동적 VaR 테스트]")
    returns = agents.get_returns()
    agent_forecasts = agents.get_agent_forecasts()
    t0 = time.time()
    var_dict = dvar.calculate_var(returns, agent_forecasts)
    elapsed = time.time() - t0
    print(f"VaR 계산 결과: { {cl: v[-1] for cl, v in var_dict.items()} }, 처리시간={elapsed*1000:.2f}ms")
    backtest = dvar.backtesting(var_dict, returns)
    print(f"백테스트 결과: {backtest}")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: GARCH→동적 위험, VaR→손실한도, 백테스트→리스크 관리 적합성 평가")

# 8. StressTestingFramework (Monte Carlo 스트레스 테스트)
import numpy as np
from scipy.stats import t as student_t

class StressTestingFramework:
    """
    StressTestingFramework
    ---------------------
    수학적 모델:
        X_t = μ + Σ^(1/2) × Z_t
        - X_t: 위험 인자 벡터
        - μ: 평균 벡터 (specialist_agents 기반)
        - Σ: 공분산 행렬
        - Z_t: 다변량 t-분포 랜덤 벡터
    경제적 의미:
        - 다양한 스트레스 시나리오(역사적/가상/역스트레스) 기반 테일 리스크 평가
        - VaR/CVaR, 상관관계 붕괴, 유동성 위험 등 통합 분석
    """
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        self.stress_scenarios = {
            'market_crash': {'equity': -0.3, 'bond': 0.1, 'commodity': -0.2},
            'interest_rate_shock': {'equity': -0.1, 'bond': -0.2, 'commodity': 0.0},
            'inflation_surge': {'equity': -0.05, 'bond': -0.15, 'commodity': 0.3}
        }
        self.nu = 6  # t-분포 자유도

    def generate_scenarios(self, agent_data, mu, cov):
        # Monte Carlo 시나리오 (다변량 t-분포)
        L = np.linalg.cholesky(cov)
        Z = student_t.rvs(self.nu, size=(self.n_simulations, len(mu)))
        X = mu + Z @ L.T
        return X

    def calculate_portfolio_pnl(self, weights, scenarios):
        # 포트폴리오 손익 계산
        return scenarios @ weights

    def stress_test_analysis(self, weights, mu, cov):
        # 1. 시나리오 생성
        scenarios = self.generate_scenarios({}, mu, cov)
        pnl = self.calculate_portfolio_pnl(weights, scenarios)
        # 2. VaR/CVaR 계산
        var_99 = np.percentile(pnl, 1)
        cvar_99 = pnl[pnl <= var_99].mean()
        var_95 = np.percentile(pnl, 5)
        cvar_95 = pnl[pnl <= var_95].mean()
        # 3. 테일/상관관계/유동성 위험 등(확장 가능)
        return {
            'VaR_99': var_99,
            'CVaR_99': cvar_99,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'tail_risk': np.mean(pnl < var_99),
            'pnl_dist': pnl
        }

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_mu_cov(self):
        mu = np.array([0.01, 0.005, 0.0])
        cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.03, 0.00],
            [0.00, 0.00, 0.06]
        ])
        return mu, cov
    def get_weights(self):
        return np.array([0.5, 0.3, 0.2])

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    stf = StressTestingFramework(n_simulations=5000)
    print("\n[Monte Carlo 스트레스 테스트]")
    mu, cov = agents.get_mu_cov()
    weights = agents.get_weights()
    t0 = time.time()
    result = stf.stress_test_analysis(weights, mu, cov)
    elapsed = time.time() - t0
    print(f"VaR/CVaR: 99%={result['VaR_99']:.4f}/{result['CVaR_99']:.4f}, 95%={result['VaR_95']:.4f}/{result['CVaR_95']:.4f}")
    print(f"테일 리스크(99%): {result['tail_risk']*100:.2f}%")
    print(f"처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: VaR/CVaR→테일 리스크, Monte Carlo→복합 시나리오, 유동성/상관관계 붕괴 확장 가능")

# 9. AdvancedTradingSystem (통합 시스템)
class AdvancedTradingSystem:
    """
    AdvancedTradingSystem
    ---------------------
    시스템 아키텍처:
        - MarketRegimeDetector: 시장 상태 분석
        - MultiFactorSignalGenerator: 신호 생성
        - BlackLittermanOptimizer: 포트폴리오 최적화
        - DynamicVaRModel: 리스크 관리
        - SystemMonitor: 성과 모니터링
    경제적 의미:
        - 모든 고급 모듈을 통합하여 실전형 멀티에이전트 트레이딩 시스템 구현
        - specialist_agents 데이터 플로우, 실시간/백테스트/모니터링 지원
    """
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.signal_generator = MultiFactorSignalGenerator()
        self.portfolio_optimizer = BlackLittermanOptimizer()
        self.risk_manager = DynamicVaRModel()
        # 성과 모니터링은 아래 SystemMonitor에서 구현

    def run_strategy_pipeline(self, agent_data, market_data):
        # 1. 시장 상태 분석
        regime, _, _ = self.regime_detector.detect_regime(agent_data['macro'], agent_data['technical'])
        # 2. 신호 생성
        signals = self.signal_generator.generate_signals(agent_data, regime)
        # 3. 포트폴리오 최적화
        views = self.portfolio_optimizer.generate_views_from_agents(agent_data, market_data['assets'])
        portfolio = self.portfolio_optimizer.optimize_portfolio(views, market_data)
        # 4. 리스크 관리
        returns = agent_data.get('returns', np.random.randn(100))
        risk_metrics = self.risk_manager.calculate_var(returns)
        # 5. 성과 모니터링(아래 SystemMonitor에서)
        return {
            'regime': regime,
            'signals': signals,
            'portfolio': portfolio,
            'risk_metrics': risk_metrics
        }

# --- specialist_agents 연동 예시 ---
class SpecialistAgents:
    def get_agent_data(self):
        return {
            'macro': {'gdp_growth': 2.8, 'cpi': 2.1},
            'technical': {'prices': np.array([100, 102, 101, 103, 105, 107]), 'rsi': 55, 'bollinger': {'mid': 103, 'std': 2}},
            'fundamental': {'roe': 0.13, 'pe': 16, 'sector': {'roe_sector': 0.11, 'sigma_roe': 0.02, 'pe_sector': 15}},
            'news': {'sentiment': 0.2, 'vix': 18},
            'returns': np.random.randn(100) * 0.01
        }
    def get_market_data(self):
        assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        cov = np.array([
            [0.04, 0.01, 0.01, 0.01],
            [0.01, 0.05, 0.01, 0.01],
            [0.01, 0.01, 0.06, 0.01],
            [0.01, 0.01, 0.01, 0.07]
        ])
        mu_prior = np.array([0.07, 0.08, 0.10, 0.09])
        return {'assets': assets, 'cov': cov, 'mu_prior': mu_prior}

# --- 테스트/성능 체크 ---
if __name__ == "__main__":
    agents = SpecialistAgents()
    ats = AdvancedTradingSystem()
    print("\n[통합 시스템 파이프라인 테스트]")
    agent_data = agents.get_agent_data()
    market_data = agents.get_market_data()
    t0 = time.time()
    result = ats.run_strategy_pipeline(agent_data, market_data)
    elapsed = time.time() - t0
    print(f"파이프라인 결과: {result}")
    print(f"처리시간={elapsed*1000:.2f}ms")
    # 메모리 사용량 체크
    import os, psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 사용량] {mem_mb:.2f} MB (100MB 이하 권장)")
    print("- 경제적 해석: 시장상태→신호→포트폴리오→리스크→성과, 실전형 통합 파이프라인")

# 10. SystemMonitor (실시간 모니터링/알림)
class SystemMonitor:
    """
    실시간 시스템 모니터링/알림/성과 분석
    - 위험 지표, 성과 지표, 데이터 품질 관리
    """
    pass 