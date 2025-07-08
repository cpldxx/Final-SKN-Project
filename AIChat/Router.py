# Router.py: LangGraph Orchestrator for multi-agent trading system.
# - Defines workflow, node connections, and routing logic
from __future__ import annotations

import json, os
from typing import List, Dict, Any
import re

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from .specialist_agents import (
    FinancialSpecialistAgent, NewsSpecialistAgent, TechnicalAnalysisAgent, MacroEconomicAgent, IndustrySectorAgent
)
from .strategy_agents import (
    MarketRegimeAgent, StrategySelectionAgent, SignalGenerationAgent, PortfolioConstructionAgent, RiskManagementAgent
)
from .decision_agent import DecisionAgent
from .schemas.models import MarketDataInput, TradingDecisionOutput

from AIChat.specialist_agents import FinancialAgent, NewsAgent, TechnicalAgent, MacroAgent, SectorAgent, OtherAgent
from AIChat.strategy_agents import StrategyAgent
from AIChat.decision_agent import DecisionAgent
import openai

# ──────────────────────────── 0. 환경 변수
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not (OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-")):
    raise ValueError("❌ 유효한 OPENAI_API_KEY가 없습니다.")

# ──────────────────────────── 1. ToolCall 모델
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# ──────────────────────────── 2. Router (필요 시 사용)
class Router:
    """OpenAI 함수-콜로 어떤 툴을 쓸지 결정하는 라우터"""
    def __init__(
        self,
        tool_specs: List[Dict[str, Any]],
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self.tool_specs = tool_specs
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a router that decides which tools the assistant should call."
        )

    def route(self, query: str) -> List[ToolCall]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tool_specs,
            temperature=0,
        )
        calls = resp.choices[0].message.tool_calls or []
        out: List[ToolCall] = []
        for tc in calls:
            try:
                out.append(ToolCall(name=tc.function.name,
                                    arguments=json.loads(tc.function.arguments)))
            except Exception:
                pass
        return out

# ──────────────────────────── 3. 툴 정의
@tool
def web_search(query: str, k: int = 5) -> str:
    """인터넷 뉴스/정보 검색"""
    from tool.newsAPI import GNewsClient  # local import
    return GNewsClient().get(keyword=query, max_results=k)

@tool
def financial_statements(ticker: str, limit: int = 3) -> str:
    """종목 재무제표 조회"""
    from tool.financial_statements import IncomeStatementClient
    data = IncomeStatementClient().get(ticker, limit)
    return f"{ticker}: {data}"

TOOLS = [web_search, financial_statements]

# ──────────────────────────── 4. LLM (툴 바인딩)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)

# ──────────────────────────── 5. LangGraph 워크플로
def should_continue(state: MessagesState):
    """마지막 AI 메시지에 tool_calls 가 있으면 툴 실행 단계로"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def call_model(state: MessagesState):
    msgs = state["messages"]
    resp = llm_with_tools.invoke(msgs)
    return {"messages": [resp]}

def build_workflow():
    builder = StateGraph(MessagesState)
    tool_node = ToolNode(TOOLS)                      # ★ agent 인자 없음 :contentReference[oaicite:0]{index=0}
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    from langgraph.graph import START
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    return builder.compile()

# ──────────────────────────── 6. CLI
if __name__ == "__main__":
    print("🔧 LangGraph Tool Router CLI (Ctrl-C to exit)")
    graph = build_workflow()

    try:
        while True:
            q = input("\nQuestion > ").strip()
            if not q:
                continue
            result = graph.invoke({"messages": [{"role": "user", "content": q}]})
            for m in result["messages"]:
                print("→", getattr(m, "content", m))
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")

class TradingOrchestrator:
    def __init__(self):
        self.financial_agent = FinancialAgent()
        self.news_agent = NewsAgent()
        self.technical_agent = TechnicalAgent()
        self.macro_agent = MacroAgent()
        self.sector_agent = SectorAgent()
        self.other_agent = OtherAgent()
        self.strategy_agent = StrategyAgent()
        self.decision_agent = DecisionAgent()
        self.llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def normalize_company_name(self, name: str) -> str:
        # 소문자 변환, 특수문자/공백 제거
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", name.lower())

    def parse_input(self, user_input: str) -> dict:
        norm_name = self.normalize_company_name(user_input)
        ticker = None
        # 1. yahooquery로 한글/영문/혼합 입력 바로 검색
        try:
            from yahooquery import search
            result = search(user_input)
            quotes = result.get('quotes', [])
            if quotes:
                ticker = quotes[0].get('symbol')
        except Exception:
            pass
        # 2. yahooquery 실패시 LLM으로 영문명 변환 후 재검색
        if not ticker:
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                prompt = f"'{user_input}'라는 한글 기업명을 영문 공식 기업명으로 변환해줘. (예: '삼성전자'→'Samsung Electronics', '엔비디아'→'Nvidia')"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                en_name = response.choices[0].message.content.strip()
                # 영문명으로 yahooquery 재검색
                from yahooquery import search
                result = search(en_name)
                quotes = result.get('quotes', [])
                if quotes:
                    ticker = quotes[0].get('symbol')
            except Exception:
                pass
        return {"ticker": ticker, "raw": user_input}

    def run(self, user_input: str) -> str:
        parsed = self.parse_input(user_input)
        # 1. Specialist/Strategy Agents 호출
        fin_data = self.financial_agent.get(parsed)
        news = self.news_agent.get(parsed)
        tech = self.technical_agent.get(parsed)
        macro = self.macro_agent.get(parsed)
        sector = self.sector_agent.get(parsed)
        other = self.other_agent.get(parsed)
        # 2. 전략/시그널/포트폴리오/리스크 관리
        strategy = self.strategy_agent.get(parsed, fin_data, tech, macro, sector)
        # 3. 통합 의사결정 Agent
        decision = self.decision_agent.get(
            fin_data, news, tech, macro, sector, other, strategy
        )
        # 4. LLM 프롬프트로 자연어 설명 생성
        llm_prompt = self.build_llm_prompt(user_input, fin_data, news, tech, macro, sector, other, strategy, decision)
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "아래는 멀티에이전트 트레이딩 시스템의 분석 결과입니다. 사용자가 이해하기 쉽게 요약해 주세요."},
                      {"role": "user", "content": llm_prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return answer

    def build_llm_prompt(self, user_input, fin_data, news, tech, macro, sector, other, strategy, decision):
        """
        LLM에 넘길 프롬프트를 통합적으로 생성
        """
        prompt = f"""
[사용자 질문]
{user_input}

[재무 데이터]
{fin_data}

[뉴스]
{news}

[기술적 분석]
{tech}

[거시경제]
{macro}

[산업/섹터]
{sector}

[기타]
{other}

[전략/시그널/포트폴리오/리스크 관리]
{strategy}

[통합 의사결정]
{decision}

위의 모든 정보를 종합적으로 고려하여, 사용자의 질문에 대해 쉽고 명확하게 답변해 주세요.
"""
        return prompt