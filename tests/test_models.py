import pytest
from app.models import ActionModel, ActionType, PriorityMode, ApplicationCase, ServiceType

def test_action_model_validation():
    a = ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.URGENT_FIRST)
    assert a.priority_mode == PriorityMode.URGENT_FIRST

def test_service_case_bounds():
    with pytest.raises(Exception):
        ApplicationCase(case_id="x", service_type=ServiceType.PASSPORT,
                        arrival_day=-1, sla_deadline_day=10)