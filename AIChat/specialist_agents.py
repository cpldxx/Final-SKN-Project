import os
import requests
import redis
import logging
import time
import random
from typing import List, Optional, Any, Union, Dict
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import ta
from tool.BaseFinanceTool import BaseFinanceTool
from tool.financial_statements import FinancialStatementsClient
from tool.newsAPI import NewsAPIClient
from tool.financial_statements import SectorAnalysisClient
# config.py에서 redis_client, REDIS_TTL import (없으면 fallback)
try:
    from config import redis_client, REDIS_TTL
except ImportError:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_TTL = int(os.getenv("REDIS_TTL_SECONDS", 60 * 60 * 24))
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client = redis.Redis(connection_pool=pool)

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 공통 SpecialistOutput
class SpecialistOutput(BaseModel):
    agent: str
    summary: str
    data: Optional[Any] = None

# 1. FinancialStatementAgent
class FinancialStatementInput(BaseModel):
    ticker: str

class FinancialStatementOutput(SpecialistOutput):
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None

class FinancialStatementAgent(BaseFinanceTool):
    def process(self, input_data: FinancialStatementInput) -> FinancialStatementOutput:
        try:
            logger.info(f"[FinancialStatementAgent] Processing: {input_data}")
            client = FinancialStatementsClient()
            result = client.get_income_statement(input_data.ticker)
            if not result:
                return FinancialStatementOutput(agent="error", summary=f"{input_data.ticker}의 재무제표를 찾을 수 없습니다.")
            revenue = result.get("revenue")
            net_income = result.get("net_income")
            eps = result.get("eps")
            summary = (
                f"{input_data.ticker}의 최근 연간 매출은 {revenue:,}달러, "
                f"순이익은 {net_income:,}달러, 주당순이익(EPS)은 {eps}입니다."
            )
            return FinancialStatementOutput(
                agent="FinancialStatementAgent",
                summary=summary,
                revenue=revenue,
                net_income=net_income,
                eps=eps,
                data=result
            )
        except Exception as e:
            logger.error(f"[FinancialStatementAgent] 오류: {e}")
            return FinancialStatementOutput(agent="error", summary=f"재무제표 조회 오류: {e}")

# 2. NewsAgent
class NewsInput(BaseModel):
    query: str
    k: int = 5

class NewsOutput(SpecialistOutput):
    news: Optional[List[dict]] = None  # [{title, url}]

class NewsAgent(BaseFinanceTool):
    def process(self, input_data: NewsInput) -> NewsOutput:
        try:
            logger.info(f"[NewsAgent] Processing: {input_data}")
            client = NewsAPIClient()
            articles = client.get_news(input_data.query, input_data.k)
            if not articles:
                return NewsOutput(agent="error", summary=f"{input_data.query}에 대한 뉴스를 찾을 수 없습니다.")
            news_list = [{"title": a["title"], "url": a["url"]} for a in articles]
            summary = f"{input_data.query} 관련 최신 뉴스 {len(news_list)}건:\n" + "\n".join(
                [f"- {n['title']} ({n['url']})" for n in news_list]
            )
            return NewsOutput(
                agent="NewsAgent",
                summary=summary,
                news=news_list,
                data=articles
            )
        except Exception as e:
            logger.error(f"[NewsAgent] 오류: {e}")
            return NewsOutput(agent="error", summary=f"뉴스 조회 오류: {e}")

# 3. TechnicalAnalysisAgent (Bulk 지원, dict 반환 지원)
class TechnicalAnalysisInput(BaseModel):
    tickers: List[str]

class TechnicalAnalysisSingleOutput(SpecialistOutput):
    ticker: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    ema: Optional[float] = None

class TechnicalAnalysisAgent(BaseFinanceTool):
    def process(self, input_data: TechnicalAnalysisInput, as_dict: bool = False) -> Union[List[TechnicalAnalysisSingleOutput], Dict[str, TechnicalAnalysisSingleOutput]]:
        results = []
        logger.info(f"[TechnicalAnalysisAgent] Processing: {input_data.tickers}")
        for ticker in input_data.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo")
                if hist.empty:
                    results.append(TechnicalAnalysisSingleOutput(
                        agent="error", ticker=ticker, summary=f"{ticker}의 가격 데이터를 찾을 수 없습니다."
                    ))
                    continue
                close = hist["Close"]
                rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
                macd = ta.trend.MACD(close).macd().iloc[-1]
                ema = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
                summary = (
                    f"{ticker}의 기술적 분석: RSI={rsi:.2f}, MACD={macd:.2f}, 20일 EMA={ema:.2f}입니다."
                )
                results.append(TechnicalAnalysisSingleOutput(
                    agent="TechnicalAnalysisAgent",
                    ticker=ticker,
                    summary=summary,
                    rsi=rsi,
                    macd=macd,
                    ema=ema,
                    data={"rsi": rsi, "macd": macd, "ema": ema}
                ))
            except Exception as e:
                logger.error(f"[TechnicalAnalysisAgent] {ticker} 오류: {e}")
                results.append(TechnicalAnalysisSingleOutput(
                    agent="error", ticker=ticker, summary=f"기술적 분석 오류: {e}"
                ))
        if as_dict:
            return {r.ticker: r for r in results}
        return results

# 4. MacroEconomicAgent (Retry logic + jitter)
def get_with_retry(url, max_retries=3, backoff=2):
    for i in range(max_retries):
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                return resp
            logger.warning(f"[MacroEconomicAgent] {url} status {resp.status_code}, retry {i+1}")
        except Exception as e:
            logger.warning(f"[MacroEconomicAgent] {url} 예외: {e}, retry {i+1}")
        time.sleep(backoff ** i + random.uniform(0, 1))
    return resp  # 마지막 응답 반환

class MacroEconomicInput(BaseModel):
    series_ids: List[str]

class MacroEconomicSeries(BaseModel):
    series_id: str
    latest_value: Optional[float]
    observation_date: Optional[str]
    units: Optional[str]

class MacroEconomicOutput(SpecialistOutput):
    series: List[MacroEconomicSeries] = []

class MacroEconomicAgent(BaseFinanceTool):
    def process(self, input_data: MacroEconomicInput) -> MacroEconomicOutput:
        try:
            logger.info(f"[MacroEconomicAgent] Processing: {input_data.series_ids}")
            # Redis 캐싱 (환경변수 TTL)
            cache_key = f"macro:{'-'.join(input_data.series_ids)}"
            cached = redis_client.get(cache_key)
            if cached:
                import json
                data = json.loads(cached)
                summary = "캐시에서 불러온 거시경제 데이터입니다."
                return MacroEconomicOutput(agent="MacroEconomicAgent", summary=summary, series=data)
            # FRED API 직접 호출
            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                return MacroEconomicOutput(agent="error", summary="FRED API 키가 없습니다.")
            series_list = []
            for sid in input_data.series_ids:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={sid}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
                )
                resp = get_with_retry(url)
                if resp.status_code != 200:
                    series_list.append({
                        "series_id": sid,
                        "latest_value": None,
                        "observation_date": None,
                        "units": None
                    })
                    continue
                obs = resp.json().get("observations", [])
                if obs:
                    latest = obs[0]
                    value = float(latest["value"]) if latest["value"] not in ("", ".") else None
                    date = latest["date"]
                    units = resp.json().get("units", "")
                else:
                    value, date, units = None, None, None
                series_list.append({
                    "series_id": sid,
                    "latest_value": value,
                    "observation_date": date,
                    "units": units
                })
            # 캐시 저장
            import json
            redis_client.setex(cache_key, REDIS_TTL, json.dumps(series_list))
            summary = "거시경제 주요 지표: " + ", ".join(
                [f"{s['series_id']}={s['latest_value']}({s['observation_date']})" for s in series_list]
            )
            return MacroEconomicOutput(
                agent="MacroEconomicAgent",
                summary=summary,
                series=[MacroEconomicSeries(**s) for s in series_list],
                data=series_list
            )
        except Exception as e:
            logger.error(f"[MacroEconomicAgent] 오류: {e}")
            return MacroEconomicOutput(agent="error", summary=f"거시경제 데이터 오류: {e}")

# 5. SectorAnalysisAgent
class SectorAnalysisInput(BaseModel):
    sector_name: str

class SectorAnalysisOutput(SpecialistOutput):
    tickers: Optional[List[str]] = None
    avg_per: Optional[float] = None
    avg_pbr: Optional[float] = None

class SectorAnalysisAgent(BaseFinanceTool):
    def process(self, input_data: SectorAnalysisInput) -> SectorAnalysisOutput:
        try:
            logger.info(f"[SectorAnalysisAgent] Processing: {input_data.sector_name}")
            client = SectorAnalysisClient()
            result = client.get_sector_analysis(input_data.sector_name)
            if not result or not result.get("tickers"):
                return SectorAnalysisOutput(agent="error", summary=f"{input_data.sector_name} 섹터 정보를 찾을 수 없습니다.")
            tickers = result["tickers"]
            avg_per = result.get("avg_per")
            avg_pbr = result.get("avg_pbr")
            summary = (
                f"{input_data.sector_name} 섹터 주요 종목: {', '.join(tickers)}\n"
                f"평균 PER: {avg_per}, 평균 PBR: {avg_pbr}"
            )
            return SectorAnalysisOutput(
                agent="SectorAnalysisAgent",
                summary=summary,
                tickers=tickers,
                avg_per=avg_per,
                avg_pbr=avg_pbr,
                data=result
            )
        except Exception as e:
            logger.error(f"[SectorAnalysisAgent] 오류: {e}")
            return SectorAnalysisOutput(agent="error", summary=f"섹터 분석 오류: {e}")

# -------------------------------
# 사용 예시
# -------------------------------

if __name__ == "__main__":
    # 1. FinancialStatementAgent
    fin_agent = FinancialStatementAgent()
    fin_input = FinancialStatementInput(ticker="AAPL")
    print(fin_agent.process(fin_input).json(indent=2, ensure_ascii=False))

    # 2. NewsAgent
    news_agent = NewsAgent()
    news_input = NewsInput(query="Apple", k=3)
    print(news_agent.process(news_input).json(indent=2, ensure_ascii=False))

    # 3. TechnicalAnalysisAgent (Bulk, dict 반환)
    tech_agent = TechnicalAnalysisAgent()
    tech_input = TechnicalAnalysisInput(tickers=["AAPL", "MSFT", "GOOGL"])
    # 리스트 반환
    for result in tech_agent.process(tech_input):
        print(result.json(indent=2, ensure_ascii=False))
    # dict 반환
    tech_dict = tech_agent.process(tech_input, as_dict=True)
    print({k: v.json(ensure_ascii=False) for k, v in tech_dict.items()})

    # 4. MacroEconomicAgent
    macro_agent = MacroEconomicAgent()
    macro_input = MacroEconomicInput(series_ids=["FEDFUNDS", "CPIAUCSL"])
    print(macro_agent.process(macro_input).json(indent=2, ensure_ascii=False))

    # 5. SectorAnalysisAgent
    sector_agent = SectorAnalysisAgent()
    sector_input = SectorAnalysisInput(sector_name="IT")
    print(sector_agent.process(sector_input).json(indent=2, ensure_ascii=False)) 


