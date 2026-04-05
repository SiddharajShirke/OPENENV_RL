from app.tasks import get_task, list_tasks

def test_three_tasks_present():
    assert list_tasks() == ["cross_department_hard", "district_backlog_easy", "mixed_urgency_medium"]

def test_task_determinism():
    assert get_task("mixed_urgency_medium").model_dump() == get_task("mixed_urgency_medium").model_dump()