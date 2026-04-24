"""
event_engine.py — Gov Workflow OpenEnv v2.0
Deterministic daily event system. Same seed + day + scenario = same events always.
"""
import random
from typing import List
from app.models import EventType, ScenarioMode, TaskConfig

SCENARIO_MULTIPLIER = {
    ScenarioMode.NORMAL:           1.0,
    ScenarioMode.CRISIS:           2.0,
    ScenarioMode.EXTREME_OVERLOAD: 3.5,
}

BASE_PROBS = {
    EventType.SURGE_APPLICATIONS:       0.08,
    EventType.OFFICER_UNAVAILABLE:      0.07,
    EventType.DOCUMENT_REJECTION_SPIKE: 0.10,
    EventType.REVENUE_DB_DELAY:         0.06,
    EventType.SLA_ESCALATION_ORDER:     0.05,
}

EVENT_EFFECTS = {
    EventType.SURGE_APPLICATIONS:
        {ScenarioMode.NORMAL: 1.3, ScenarioMode.CRISIS: 1.5, ScenarioMode.EXTREME_OVERLOAD: 2.0},
    EventType.OFFICER_UNAVAILABLE:
        {ScenarioMode.NORMAL: 1,   ScenarioMode.CRISIS: 1,   ScenarioMode.EXTREME_OVERLOAD: 2},
    EventType.DOCUMENT_REJECTION_SPIKE:
        {ScenarioMode.NORMAL: 0.15, ScenarioMode.CRISIS: 0.20, ScenarioMode.EXTREME_OVERLOAD: 0.35},
    EventType.REVENUE_DB_DELAY:
        {ScenarioMode.NORMAL: 0.30, ScenarioMode.CRISIS: 0.40, ScenarioMode.EXTREME_OVERLOAD: 0.60},
    EventType.SLA_ESCALATION_ORDER:
        {ScenarioMode.NORMAL: 0.50, ScenarioMode.CRISIS: 0.50, ScenarioMode.EXTREME_OVERLOAD: 0.40},
}


class DayEventParams:
    def __init__(self):
        self.arrival_multiplier: float = 1.0
        self.officer_reduction: int = 0
        self.doc_defect_rate_boost: float = 0.0
        self.system_dependency_boost: float = 0.0
        self.sla_window_multiplier: float = 1.0
        self.active_events: List[EventType] = []

    def has_events(self) -> bool:
        return bool(self.active_events)


class EventEngine:
    def __init__(self, seed: int, scenario_mode: ScenarioMode):
        self.seed = seed
        self.scenario_mode = scenario_mode
        self._multiplier = SCENARIO_MULTIPLIER[scenario_mode]

    def get_events_for_day(self, day: int, task_config: "TaskConfig") -> List[EventType]:
        day_rng = random.Random(self.seed + day * 31337)
        active = []
        for event_type in task_config.allowed_events:
            if event_type == EventType.NO_EVENT:
                continue
            base_prob = BASE_PROBS.get(event_type, 0.0)
            effective_prob = min(0.80, base_prob * self._multiplier)
            if day_rng.random() < effective_prob:
                active.append(event_type)
        return active if active else [EventType.NO_EVENT]

    def apply_events(self, events: List[EventType], task_config: "TaskConfig") -> DayEventParams:
        params = DayEventParams()
        for event in events:
            if event == EventType.NO_EVENT:
                continue
            params.active_events.append(event)
            magnitude = EVENT_EFFECTS.get(event, {}).get(self.scenario_mode, 0)
            if event == EventType.SURGE_APPLICATIONS:
                params.arrival_multiplier *= magnitude
            elif event == EventType.OFFICER_UNAVAILABLE:
                params.officer_reduction += int(magnitude)
            elif event == EventType.DOCUMENT_REJECTION_SPIKE:
                params.doc_defect_rate_boost += magnitude
            elif event == EventType.REVENUE_DB_DELAY:
                params.system_dependency_boost += magnitude
            elif event == EventType.SLA_ESCALATION_ORDER:
                params.sla_window_multiplier = min(params.sla_window_multiplier, magnitude)
        if not params.active_events:
            params.active_events = [EventType.NO_EVENT]
        return params

    def describe_events(self, events: List[EventType]) -> str:
        descriptions = {
            EventType.SURGE_APPLICATIONS:       "Digital surge: arrivals increased",
            EventType.OFFICER_UNAVAILABLE:      "Officer absent: reduced capacity",
            EventType.DOCUMENT_REJECTION_SPIKE: "Doc rejection spike: higher defect rate",
            EventType.REVENUE_DB_DELAY:         "Revenue DB delay: land records slower",
            EventType.SLA_ESCALATION_ORDER:     "SLA escalation order: deadlines tightened",
            EventType.NO_EVENT:                 "No active events today",
        }
        active = [e for e in events if e != EventType.NO_EVENT]
        if not active:
            return "No active events today"
        return "; ".join(descriptions.get(e, str(e)) for e in active)
