from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from tool.newsAPI import GNewsClient
from tool.financial_statements import IncomeStatementClient
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# ------------------------------------------------------
# 툴 함수 정의
# ------------------------------------------------------
def web_search(query: str, k: int = 5):
    news = GNewsClient()
    answer = news.get(keyword=query, max_results=k)
    return answer

def financial_statements(ticker: str, limit: int = 1):
    client = IncomeStatementClient()
    answer = client.get(ticker, limit)
    return f"ticker: {ticker}, financial data: {answer}"

# ------------------------------------------------------
# Tool 객체 정의
# ------------------------------------------------------
tools = [
    Tool(
        name="web_search",
        func=web_search,
        description="인터넷에서 최신 뉴스나 사실 정보를 검색합니다. query(키워드)와 k(개수)를 입력받습니다."
    ),
    Tool(
        name="financial_statements",
        func=financial_statements,
        description="주식/암호화폐 실시간 가격, 재무 지표를 조회합니다. ticker(종목코드)와 limit(조회개수)를 입력받습니다."
    )
]

# ------------------------------------------------------
# LLM + Agent 초기화
# ------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# ------------------------------------------------------
# CLI 실행
# ------------------------------------------------------
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nQuestion > ")
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        result = agent.run(user_input)
        print(result)