from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Route(str, Enum):
    DIRECT = "direct"
    INTERNAL_TOOL = "internal_tool"
    HANDOFF = "handoff"


class Evidence(BaseModel):
    source_id: str
    title: str
    text: str
    source_type: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: Optional[str] = None
    shop_id: Optional[str] = None
    product_id: Optional[str] = None
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    route: Route
    intent: str
    time_sensitive: bool = False
    need_clarification: bool = False
    confidence: float = 0.0
    rewrite_query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    rationale: str = ""


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    status: str
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class ToolStep(BaseModel):
    call: ToolCall
    result: ToolResult


class AnswerPayload(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    grounded: bool = False
    escalation_required: bool = False


class ChatResponse(BaseModel):
    route: Route
    intent: str
    answer: str
    confidence: float
    rewrite_query: str
    evidence: List[Evidence] = Field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    tool_steps: List[ToolStep] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    grounded: bool = False
    escalation_required: bool = False
    waiting_for_user: bool = False
    episode_done: bool = True
    debug: Dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    task_query: Optional[str] = None
    intent: Optional[str] = None
    route: Optional[Route] = None
    planned_tool_name: Optional[str] = None
    planned_arguments: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    resolved_slots: Dict[str, Any] = Field(default_factory=dict)
    status: str = "idle"


class EpisodeState(BaseModel):
    shop_id: Optional[str] = None
    product_id: Optional[str] = None
    order_id: Optional[str] = None
    current_task: TaskState = Field(default_factory=TaskState)
    recent_queries: List[str] = Field(default_factory=list)
    turn_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_pending_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        current_task = payload.get("current_task")
        if current_task is None:
            legacy_intent = payload.pop("pending_intent", None)
            legacy_tool = payload.pop("pending_tool_name", None)
            legacy_missing = payload.pop("pending_missing_slots", None) or []
            legacy_arguments = payload.pop("pending_arguments", None) or {}
            legacy_query = payload.pop("pending_original_query", None)

            payload["current_task"] = {
                "task_query": legacy_query,
                "intent": legacy_intent,
                "planned_tool_name": legacy_tool,
                "planned_arguments": legacy_arguments,
                "missing_slots": legacy_missing,
                "status": "pending_clarification" if legacy_tool and legacy_missing else "idle",
            }
        return payload

    @property
    def pending_intent(self) -> Optional[str]:
        return self.current_task.intent

    @pending_intent.setter
    def pending_intent(self, value: Optional[str]) -> None:
        self.current_task.intent = value

    @property
    def pending_tool_name(self) -> Optional[str]:
        return self.current_task.planned_tool_name

    @pending_tool_name.setter
    def pending_tool_name(self, value: Optional[str]) -> None:
        self.current_task.planned_tool_name = value

    @property
    def pending_missing_slots(self) -> List[str]:
        return self.current_task.missing_slots

    @pending_missing_slots.setter
    def pending_missing_slots(self, value: List[str]) -> None:
        self.current_task.missing_slots = list(value)

    @property
    def pending_arguments(self) -> Dict[str, Any]:
        return self.current_task.planned_arguments

    @pending_arguments.setter
    def pending_arguments(self, value: Dict[str, Any]) -> None:
        self.current_task.planned_arguments = dict(value)

    @property
    def pending_original_query(self) -> Optional[str]:
        return self.current_task.task_query

    @pending_original_query.setter
    def pending_original_query(self, value: Optional[str]) -> None:
        self.current_task.task_query = value


class TraceRecord(BaseModel):
    request: ChatRequest
    route_decision: RouteDecision
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    tool_steps: List[ToolStep] = Field(default_factory=list)
    response: ChatResponse
    state_before: Optional[EpisodeState] = None
    state_after: Optional[EpisodeState] = None


class EpisodeRecord(BaseModel):
    episode_id: str
    turns: List[TraceRecord] = Field(default_factory=list)
    final_response: ChatResponse
    final_state: EpisodeState = Field(default_factory=EpisodeState)
    waiting_for_user: bool = False
    completed: bool = True
