from pydantic import BaseModel, Field
from typing import Optional

class MarketDataInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol (e.g., AAPL)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")

class SpecialistOutput(BaseModel):
    agent: str = Field(..., description="Agent name")
    summary: str = Field(..., description="Summary of analysis")

class StrategyOutput(BaseModel):
    agent: str = Field(..., description="Agent name")
    strategy: str = Field(..., description="Selected strategy or regime")

class SignalOutput(BaseModel):
    agent: str = Field(..., description="Agent name")
    signal: str = Field(..., description="Buy/Sell/Hold signal")

class PortfolioOutput(BaseModel):
    agent: str = Field(..., description="Agent name")
    portfolio: str = Field(..., description="Portfolio allocation details")

class RiskOutput(BaseModel):
    agent: str = Field(..., description="Agent name")
    risk: str = Field(..., description="Risk assessment details")

class TradingDecisionOutput(BaseModel):
    decision: str = Field(..., description="Final trading decision")
    rationale: str = Field(..., description="Rationale for the decision") 
    