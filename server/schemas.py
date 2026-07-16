from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class PlanRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    feedback: str


class SessionStateResponse(BaseModel):
    session_id: str
    status: str  # "idle", "streaming_phase1", "reviewing", "streaming_phase2", "completed"
    daily_itinerary: Optional[List[Dict[str, Any]]] = None
    currency_symbol: Optional[str] = None
    exchange_rate: Optional[float] = None
    is_chinese: Optional[bool] = None


class ReportResponse(BaseModel):
    session_id: str
    report: str
    status: str
