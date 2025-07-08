"""
app.py: Entry point for the multi-agent trading AI system.
- FastAPI endpoint for pipeline execution
- CLI example for local testing
- LLM 챗봇: 자연어 입력 → 전체 멀티에이전트 파이프라인 → LLM 요약 답변
"""
from AIChat.Router import TradingOrchestrator
from AIChat.schemas.models import MarketDataInput, TradingDecisionOutput
from fastapi import FastAPI
import uvicorn
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.post("/trade_decision", response_model=TradingDecisionOutput)
def trade_decision(input_data: MarketDataInput):
    orchestrator = TradingOrchestrator()
    return orchestrator.run(input_data)

def cli():
    parser = argparse.ArgumentParser(description="Run trading pipeline via CLI.")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--end_date", type=str)
    parser.add_argument("--llm_chat", action="store_true", help="LLM 챗봇 모드 실행 (자연어 입력 → 전체 파이프라인)")
    args = parser.parse_args()
    if args.llm_chat:
        run_llm_chat()
    else:
        orchestrator = TradingOrchestrator()
        result = orchestrator.run(MarketDataInput(ticker=args.ticker, start_date=args.start_date, end_date=args.end_date))
        print("[CLI] Pipeline result:", result)

def run_llm_chat():
    """
    터미널에서 사용자가 입력한 자연어 질문을 전체 멀티에이전트 파이프라인에 전달하고, LLM 요약 답변을 출력
    """
    orchestrator = TradingOrchestrator()
    print("[LLM 챗봇] 종료하려면 Ctrl+C 또는 'exit' 입력")
    while True:
        try:
            user_input = input("[User] 질문: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("[LLM 챗봇] 종료합니다.")
                break
            response = orchestrator.run(user_input)
            print(f"[AI] 답변: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n[LLM 챗봇] 종료합니다.")
            break
        except Exception as e:
            print(f"[에러] {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000) 