import pytest
from app.models import ActionModel, ActionType, PriorityMode, ServiceCase, ServiceType

def test_action_model_validation():
    a = ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.URGENT_FIRST)
    assert a.priority_mode == PriorityMode.URGENT_FIRST

def test_service_case_urgency_bounds():
    with pytest.raises(Exception):
        ServiceCase(case_id="x", service=ServiceType.PASSPORT,
                    arrival_day=0, due_day=10, urgency=4)