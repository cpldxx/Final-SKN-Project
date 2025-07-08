from .BaseFinanceTool import BaseFinanceTool
from .schemas.models import SpecialistOutput, StrategyOutput, SignalOutput, PortfolioOutput, RiskOutput

class MarketRegimeAgent(BaseFinanceTool):
    def get(self, fin_out: SpecialistOutput, macro_out: SpecialistOutput, **kwargs):
        # 예시: macro_out의 summary에 따라 bull/bear regime 분류
        if "CPI" in macro_out.summary and float(macro_out.summary.split(":")[-1]) > 250:
            regime = "Bull"
        else:
            regime = "Bear"
        return {"agent": "MarketRegime", "strategy": f"Mock regime: {regime}"}
    def process(self, fin_out: SpecialistOutput, macro_out: SpecialistOutput) -> StrategyOutput:
        result = self.get(fin_out, macro_out)
        print(f"[MarketRegimeAgent] Processing: {fin_out}, {macro_out}")
        return StrategyOutput(agent=result["agent"], strategy=result["strategy"])

class StrategySelectionAgent(BaseFinanceTool):
    def get(self, regime_out: StrategyOutput, tech_out: SpecialistOutput, news_out: SpecialistOutput, **kwargs):
        # 예시: regime, tech, news summary를 조합해 전략 선택
        if "Bull" in regime_out.strategy and "RSI" in tech_out.summary:
            strategy = "Momentum"
        else:
            strategy = "Value"
        return {"agent": "StrategySelection", "strategy": f"Selected: {strategy}"}
    def process(self, regime_out: StrategyOutput, tech_out: SpecialistOutput, news_out: SpecialistOutput) -> StrategyOutput:
        result = self.get(regime_out, tech_out, news_out)
        print(f"[StrategySelectionAgent] Processing: {regime_out}, {tech_out}, {news_out}")
        return StrategyOutput(agent=result["agent"], strategy=result["strategy"])

class SignalGenerationAgent(BaseFinanceTool):
    def get(self, strategy_out: StrategyOutput, industry_out: SpecialistOutput, **kwargs):
        # 예시: 전략과 산업 summary에 따라 신호 생성
        if "Momentum" in strategy_out.strategy:
            signal = "BUY"
        else:
            signal = "HOLD"
        return {"agent": "SignalGeneration", "signal": signal}
    def process(self, strategy_out: StrategyOutput, industry_out: SpecialistOutput) -> SignalOutput:
        result = self.get(strategy_out, industry_out)
        print(f"[SignalGenerationAgent] Processing: {strategy_out}, {industry_out}")
        return SignalOutput(agent=result["agent"], signal=result["signal"])

class PortfolioConstructionAgent(BaseFinanceTool):
    def get(self, signal_out: SignalOutput, **kwargs):
        # 예시: 신호에 따라 포트폴리오 구성
        if signal_out.signal == "BUY":
            portfolio = "100% in target stock"
        else:
            portfolio = "50% cash, 50% stock"
        return {"agent": "PortfolioConstruction", "portfolio": portfolio}
    def process(self, signal_out: SignalOutput) -> PortfolioOutput:
        result = self.get(signal_out)
        print(f"[PortfolioConstructionAgent] Processing: {signal_out}")
        return PortfolioOutput(agent=result["agent"], portfolio=result["portfolio"])

class RiskManagementAgent(BaseFinanceTool):
    def get(self, portfolio_out: PortfolioOutput, **kwargs):
        # 예시: 포트폴리오에 따라 리스크 평가
        if "100%" in portfolio_out.portfolio:
            risk = "High"
        else:
            risk = "Low"
        return {"agent": "RiskManagement", "risk": risk}
    def process(self, portfolio_out: PortfolioOutput) -> RiskOutput:
        result = self.get(portfolio_out)
        print(f"[RiskManagementAgent] Processing: {portfolio_out}")
        return RiskOutput(agent=result["agent"], risk=result["risk"])

class StrategyAgent:
    def get(self, parsed, fin_data, tech, macro, sector):
        # 예시: 단순 모멘텀/평균회귀 신호, 포트폴리오 비중, 리스크 수치 등
        # 실제로는 위 입력값을 분석해 전략을 선택/수치화
        signal = "모멘텀 매수" if "RSI" in str(tech) and "RSI: " in str(tech) and float(tech.split("RSI: ")[1].split(",")[0]) < 70 else "관망"
        portfolio = {"TSLA": 0.5, "XLK": 0.3, "USD/KRW": 0.2}
        risk = "VaR: 2.1%"
        return f"신호: {signal}, 포트폴리오: {portfolio}, 리스크: {risk}" 