# gnews_client.py
import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)                    # .env 1회 로드

class GNewsClient:
    BASE_URL = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GNEWS_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GNEWS_API_KEY가 설정돼 있지 않습니다.")

    def fetch_news(self, keyword: str, max_results: int = 5) -> str:
        params = {
            "q": keyword,
            "lang": "en",
            "apikey": self.api_key,   # ✅ 여기! token → apikey
            "max": max_results
        }

        try:
            res = requests.get(self.BASE_URL, params=params, timeout=5)
            res.raise_for_status()    # 200 아니면 예외
        except requests.exceptions.RequestException as e:
            return f"🌐 HTTP/네트워크 오류: {e}"

        arts = res.json().get("articles", [])
        if not arts:
            return "📭 관련 뉴스가 없습니다."

        lines = [f"📰 [ {keyword} ] 관련 최신 뉴스 {len(arts)}건"]
        for i, a in enumerate(arts, 1):
            lines.append(f"{i}. {a['title']} ({a['source']['name']}, {a['publishedAt'][:10]})\n   🔗 {a['url']}")
        return "\n\n".join(lines)
