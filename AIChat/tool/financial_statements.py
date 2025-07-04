# income_statement_client.py
import requests
from typing import Optional
from tool.BaseFinanceTool import BaseFinanceTool

class IncomeStatementClient(BaseFinanceTool):
    BASE_URL = "https://financialmodelingprep.com/api/v3/income-statement"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("FMP_API_KEY" if api_key is None else None)
        self.api_key = api_key or self.api_key
        if not self.api_key:
            raise ValueError("❌ FMP_API_KEY가 설정돼 있지 않습니다.")

    def get(self, ticker: str, limit: int = 1) -> str:
        url = f"{self.BASE_URL}/{ticker.upper()}"
        params = {
            "apikey": self.api_key,
            "limit": limit
        }

        try:
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
        except requests.RequestException as e:
            return f"🌐 HTTP 오류: {e}"

        data = res.json()
        if not data:
            return f"📭 {ticker}의 재무제표 데이터가 없습니다."

        # 최근 연도 기준으로 출력
        lines = [f"📊 [ {ticker.upper()} ] 최근 {len(data)}개 재무제표 요약"]
        for report in data:
            date = report.get("date", "N/A")
            revenue = report.get("revenue", "N/A")
            net_income = report.get("netIncome", "N/A")
            eps = report.get("eps", "N/A")

            lines.append(
                f"- 📅 {date}\n"
                f"  매출액: ${revenue:,}\n"
                f"  순이익: ${net_income:,}\n"
                f"  EPS: {eps}"
            )

        return "\n\n".join(lines)
