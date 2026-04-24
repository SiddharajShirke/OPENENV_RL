"""
tests/test_phase1_sector_and_tasks.py
Phase 1 validation: sector_profiles.py + tasks.py
Run: pytest tests/test_phase1_sector_and_tasks.py -v
"""
import pytest
from app.models import ServiceType, ScenarioMode, EventType
from app.sector_profiles import (
    get_sector_profile,
    SECTOR_REGISTRY,
    INCOME_CERTIFICATE_PROFILE,
    LAND_REGISTRATION_PROFILE,
    BIRTH_CERTIFICATE_PROFILE,
    PASSPORT_PROFILE,
    GST_REGISTRATION_PROFILE,
    CASTE_CERTIFICATE_PROFILE,
    DRIVING_LICENSE_PROFILE,
)
from app.tasks import (
    get_task,
    list_tasks,
    list_benchmark_tasks,
    TASK_EASY,
    TASK_MEDIUM,
    TASK_HARD,
    TASK_REGISTRY,
    make_extreme_variant,
)


# ─── Sector Profiles Registry ────────────────────────────────────────────────
class TestSectorRegistry:
    def test_all_services_have_profiles(self):
        for svc in ServiceType:
            assert svc in SECTOR_REGISTRY, f"Missing profile for {svc}"

    def test_get_sector_profile_all_services(self):
        for svc in ServiceType:
            profile = get_sector_profile(svc)
            assert profile.service_type == svc

    def test_unknown_service_raises_key_error(self):
        with pytest.raises(KeyError):
            get_sector_profile("nonexistent_service")  # type: ignore

    def test_registry_has_seven_entries(self):
        assert len(SECTOR_REGISTRY) == 7


# ─── Individual Sector Profile Values ────────────────────────────────────────
class TestIncomeCertificateProfile:
    def test_sla_days(self):
        assert INCOME_CERTIFICATE_PROFILE.sla_days == 21

    def test_missing_docs_probability_range(self):
        p = INCOME_CERTIFICATE_PROFILE.missing_docs_probability
        assert 0.0 <= p <= 1.0

    def test_field_verification_probability_range(self):
        p = INCOME_CERTIFICATE_PROFILE.field_verification_probability
        assert 0.0 <= p <= 1.0

    def test_base_processing_rate_positive(self):
        assert INCOME_CERTIFICATE_PROFILE.base_processing_rate > 0

    def test_field_verification_days_positive(self):
        assert INCOME_CERTIFICATE_PROFILE.field_verification_days >= 1

    def test_doc_defect_rate_paper_higher_than_digital(self):
        assert (INCOME_CERTIFICATE_PROFILE.doc_defect_rate_paper >
                INCOME_CERTIFICATE_PROFILE.doc_defect_rate_digital)


class TestLandRegistrationProfile:
    def test_sla_days_thirty(self):
        assert LAND_REGISTRATION_PROFILE.sla_days == 30

    def test_field_verification_heavy(self):
        # Land registration has the highest field verification probability
        assert LAND_REGISTRATION_PROFILE.field_verification_probability > 0.5

    def test_field_verification_days_longer(self):
        # Land should require more field verification days than income cert
        assert (LAND_REGISTRATION_PROFILE.field_verification_days >=
                INCOME_CERTIFICATE_PROFILE.field_verification_days)


class TestBirthCertificateProfile:
    def test_sla_days_seven(self):
        assert BIRTH_CERTIFICATE_PROFILE.sla_days == 7

    def test_fast_processing_rate(self):
        # Birth certificate should process faster than land registration
        assert (BIRTH_CERTIFICATE_PROFILE.base_processing_rate >
                LAND_REGISTRATION_PROFILE.base_processing_rate)

    def test_low_missing_docs_probability(self):
        assert BIRTH_CERTIFICATE_PROFILE.missing_docs_probability < 0.30


class TestGSTProfile:
    def test_sla_days_seven(self):
        assert GST_REGISTRATION_PROFILE.sla_days == 7

    def test_all_probabilities_in_range(self):
        p = GST_REGISTRATION_PROFILE
        for attr in ["missing_docs_probability", "doc_defect_rate_digital",
                     "doc_defect_rate_paper", "field_verification_probability"]:
            val = getattr(p, attr)
            assert 0.0 <= val <= 1.0, f"{attr} out of range: {val}"


class TestAllProfileConstraints:
    @pytest.mark.parametrize("service", list(ServiceType))
    def test_probabilities_in_range(self, service):
        p = get_sector_profile(service)
        for attr in ["missing_docs_probability", "doc_defect_rate_digital",
                     "doc_defect_rate_paper", "field_verification_probability",
                     "manual_scrutiny_intensity", "decision_backlog_sensitivity",
                     "system_dependency_risk"]:
            val = getattr(p, attr)
            assert 0.0 <= val <= 1.0, (
                f"{service.value}.{attr} = {val} is outside [0, 1]"
            )

    @pytest.mark.parametrize("service", list(ServiceType))
    def test_sla_days_positive(self, service):
        p = get_sector_profile(service)
        assert p.sla_days >= 1

    @pytest.mark.parametrize("service", list(ServiceType))
    def test_processing_rate_positive(self, service):
        p = get_sector_profile(service)
        assert p.base_processing_rate >= 0.1

    @pytest.mark.parametrize("service", list(ServiceType))
    def test_field_verification_days_positive(self, service):
        p = get_sector_profile(service)
        assert p.field_verification_days >= 1

    @pytest.mark.parametrize("service", list(ServiceType))
    def test_paper_defect_rate_higher_than_digital(self, service):
        p = get_sector_profile(service)
        assert p.doc_defect_rate_paper >= p.doc_defect_rate_digital, (
            f"{service.value}: paper defect rate should be >= digital"
        )


# ─── Tasks ────────────────────────────────────────────────────────────────────
class TestTaskRegistry:
    def test_three_benchmark_tasks_exist(self):
        tasks = list_benchmark_tasks()
        assert len(tasks) == 3

    def test_benchmark_task_ids(self):
        tasks = set(list_benchmark_tasks())
        assert "district_backlog_easy" in tasks
        assert "mixed_urgency_medium" in tasks
        assert "cross_department_hard" in tasks

    def test_all_tasks_retrievable(self):
        for tid in list_tasks():
            task = get_task(tid)
            assert task.task_id == tid

    def test_unknown_task_raises_value_error(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task_id_xyz")

    def test_registry_has_at_least_three_entries(self):
        assert len(TASK_REGISTRY) >= 3


class TestTaskEasy:
    def test_task_id(self):
        assert TASK_EASY.task_id == "district_backlog_easy"

    def test_difficulty(self):
        assert TASK_EASY.difficulty == "easy"

    def test_scenario_mode_normal(self):
        assert TASK_EASY.scenario_mode == ScenarioMode.NORMAL

    def test_seed_deterministic(self):
        assert TASK_EASY.seed == 42

    def test_max_days_thirty(self):
        assert TASK_EASY.max_days == 30

    def test_single_service(self):
        assert len(TASK_EASY.enabled_services) == 1
        assert ServiceType.INCOME_CERTIFICATE in TASK_EASY.enabled_services

    def test_arrival_rate_positive(self):
        for svc, rate in TASK_EASY.arrival_rate_per_day.items():
            assert rate > 0, f"Arrival rate for {svc} should be positive"

    def test_officer_pool_valid(self):
        pool = TASK_EASY.initial_officer_pool
        assert pool.total_officers >= 1
        assert pool.available_officers >= 1

    def test_escalation_budget_nonnegative(self):
        assert TASK_EASY.escalation_budget >= 0

    def test_no_fairness_threshold(self):
        assert TASK_EASY.fairness_threshold is None

    def test_low_event_probability(self):
        assert TASK_EASY.event_probability <= 0.10


class TestTaskMedium:
    def test_task_id(self):
        assert TASK_MEDIUM.task_id == "mixed_urgency_medium"

    def test_difficulty(self):
        assert TASK_MEDIUM.difficulty == "medium"

    def test_two_services(self):
        assert len(TASK_MEDIUM.enabled_services) == 2

    def test_max_days_forty_five(self):
        assert TASK_MEDIUM.max_days == 45

    def test_higher_event_probability_than_easy(self):
        assert TASK_MEDIUM.event_probability > TASK_EASY.event_probability

    def test_arrival_rates_for_all_services(self):
        for svc in TASK_MEDIUM.enabled_services:
            key = svc if svc in TASK_MEDIUM.arrival_rate_per_day                 else svc.value
            rate = TASK_MEDIUM.arrival_rate_per_day.get(svc,
                   TASK_MEDIUM.arrival_rate_per_day.get(svc.value, None))
            assert rate is not None and rate > 0

    def test_officer_pool_covers_both_services(self):
        pool = TASK_MEDIUM.initial_officer_pool
        allocated_services = set(pool.allocated.keys())
        # At least one service should have officers
        assert len(allocated_services) >= 1


class TestTaskHard:
    def test_task_id(self):
        assert TASK_HARD.task_id == "cross_department_hard"

    def test_difficulty(self):
        assert TASK_HARD.difficulty == "hard"

    def test_scenario_mode_crisis(self):
        assert TASK_HARD.scenario_mode == ScenarioMode.CRISIS

    def test_max_days_sixty(self):
        assert TASK_HARD.max_days == 60

    def test_fairness_threshold_set(self):
        assert TASK_HARD.fairness_threshold is not None
        assert 0.0 <= TASK_HARD.fairness_threshold <= 1.0

    def test_has_escalation_events(self):
        assert EventType.SLA_ESCALATION_ORDER in TASK_HARD.allowed_events

    def test_event_probability_highest(self):
        assert TASK_HARD.event_probability > TASK_MEDIUM.event_probability

    def test_escalation_budget_higher_than_easy(self):
        assert TASK_HARD.escalation_budget >= TASK_EASY.escalation_budget


class TestExtremeVariant:
    def test_extreme_variant_creation(self):
        extreme = make_extreme_variant(TASK_EASY)
        assert "_extreme" in extreme.task_id

    def test_extreme_scenario_mode(self):
        extreme = make_extreme_variant(TASK_MEDIUM)
        assert extreme.scenario_mode == ScenarioMode.EXTREME_OVERLOAD

    def test_extreme_event_probability_higher(self):
        extreme = make_extreme_variant(TASK_EASY)
        assert extreme.event_probability > TASK_EASY.event_probability

    def test_extreme_does_not_mutate_original(self):
        original_mode = TASK_EASY.scenario_mode
        make_extreme_variant(TASK_EASY)
        assert TASK_EASY.scenario_mode == original_mode


class TestTaskDeterminism:
    def test_same_seed_same_task(self):
        t1 = get_task("district_backlog_easy")
        t2 = get_task("district_backlog_easy")
        assert t1.seed == t2.seed
        assert t1.max_days == t2.max_days

    def test_tasks_have_different_seeds(self):
        seeds = {get_task(tid).seed for tid in list_benchmark_tasks()}
        assert len(seeds) == 3, "Each benchmark task must have a unique seed"
