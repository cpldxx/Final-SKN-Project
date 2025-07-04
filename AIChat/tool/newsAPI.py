# gnews_client.py
import requests
from typing import Optional
from tool.BaseFinanceTool import BaseFinanceTool

class GNewsClient(BaseFinanceTool):
    BASE_URL = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("GNEWS_API_KEY" if api_key is None else None)
        self.api_key = api_key or self.api_key
        if not self.api_key:
            raise ValueError("❌ GNEWS_API_KEY가 설정돼 있지 않습니다.")

    def get(self, keyword: str, max_results: int = 5) -> str:
        params = {
            "q": keyword,
            "lang": "en",
            "token": self.api_key,
            "max": max_results
        }

        try:
            res = requests.get(self.BASE_URL, params=params, timeout=5)
            res.raise_for_status()
        except requests.RequestException as e:
            return f"🌐 HTTP 오류: {e}"

        articles = res.json().get("articles", [])
        if not articles:
            return "📭 관련 뉴스가 없습니다."

        lines = [f"📰 [ {keyword} ] 관련 최신 뉴스 {len(articles)}건"]
        for i, article in enumerate(articles, 1):
            title = article.get("title", "제목 없음")
            source = article.get("source", {}).get("name", "알 수 없음")
            date = article.get("publishedAt", "")[:10]
            url = article.get("url", "")
            lines.append(f"{i}. {title} ({source}, {date})\n   🔗 {url}")

        return "\n\n".join(lines)
