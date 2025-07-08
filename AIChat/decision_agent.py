from .BaseFinanceTool import BaseFinanceTool
from .schemas.models import (
    SpecialistOutput, StrategyOutput, SignalOutput, PortfolioOutput, RiskOutput, TradingDecisionOutput
)

class DecisionAgent(BaseFinanceTool):
    def get(self, fin_out, news_out, tech_out, macro_out, industry_out, regime_out, strategy_out, signal_out, portfolio_out, risk_out, **kwargs):
        # 예시: 모든 output을 종합해 최종 의사결정
        if signal_out.signal == "BUY" and risk_out.risk == "Low":
            decision = "BUY"
            rationale = "Strong buy signal with low risk."
        else:
            decision = "HOLD"
            rationale = "No strong buy signal or risk too high."
        return {"decision": decision, "rationale": rationale}
    """
    Final decision maker integrating all agent outputs.
    """
    def process(
        self,
        fin_out: SpecialistOutput,
        news_out: SpecialistOutput,
        tech_out: SpecialistOutput,
        macro_out: SpecialistOutput,
        industry_out: SpecialistOutput,
        regime_out: StrategyOutput,
        strategy_out: StrategyOutput,
        signal_out: SignalOutput,
        portfolio_out: PortfolioOutput,
        risk_out: RiskOutput
    ) -> TradingDecisionOutput:
        result = DecisionAgent.get(
            None, fin_out, news_out, tech_out, macro_out, industry_out,
            regime_out, strategy_out, signal_out, portfolio_out, risk_out
        )
        print("[DecisionAgent] Integrating all agent outputs...")
        return TradingDecisionOutput(
            decision=result["decision"],
            rationale=result["rationale"]
        )

    def get(self, fin_data, news, tech, macro, sector, other, strategy):
        # 실제 각 Agent의 수치/헤드라인/지표를 종합해 자연어로 요약
        return (
            f"[재무] {fin_data}\n"
            f"[뉴스] {news}\n"
            f"[기술] {tech}\n"
            f"[거시] {macro}\n"
            f"[섹터] {sector}\n"
            f"[기타] {other}\n"
            f"[전략] {strategy}\n"
            "위 데이터를 종합적으로 고려해 투자 결정을 내리세요."
        ) 