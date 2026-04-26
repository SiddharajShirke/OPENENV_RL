from app.tasks import get_task, list_tasks

def test_core_tasks_present():
    """Verify the three core benchmark tasks are always present."""
    tasks = list_tasks()
    for expected in ["cross_department_hard", "district_backlog_easy", "mixed_urgency_medium"]:
        assert expected in tasks, f"Missing core task: {expected}"
    assert len(tasks) >= 3

def test_task_determinism():
    assert get_task("mixed_urgency_medium").model_dump() == get_task("mixed_urgency_medium").model_dump()