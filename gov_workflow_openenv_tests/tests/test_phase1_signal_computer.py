"""
tests/test_phase1_signal_computer.py
Phase 1 validation: signal_computer.py
Run: pytest tests/test_phase1_signal_computer.py -v
"""
import pytest
from app.models import ServiceType, OfficerPool, QueueSnapshot
from app.signal_computer import SignalComputer, ComputedSignals


def make_snapshot(
    service: ServiceType,
    total_pending: int = 0,
    completed_today: int = 0,
    sla_breached: int = 0,
    urgent: int = 0,
    blocked_missing: int = 0,
    field_pending: int = 0,
    sla_risk: float = 0.0,
) -> QueueSnapshot:
    return QueueSnapshot(
        service_type=service,
        total_pending=total_pending,
        total_completed_today=completed_today,
        total_sla_breached=sla_breached,
        urgent_pending=urgent,
        blocked_missing_docs=blocked_missing,
        field_verification_pending=field_pending,
        current_sla_risk=sla_risk,
    )


def make_pool(total=10, available=10, allocated=None) -> OfficerPool:
    return OfficerPool(
        total_officers=total,
        available_officers=available,
        allocated=allocated or {},
    )


class TestComputedSignalsDefaults:
    def test_defaults_all_zero_or_reasonable(self):
        s = ComputedSignals()
        assert s.backlog_pressure == 0.0
        assert s.sla_risk_score == 0.0
        assert s.fairness_index == 1.0
        assert s.resource_utilization == 0.0
        assert s.digital_intake_ratio == 0.5
        assert s.blocked_cases_missing_docs == 0
        assert s.field_verification_load == 0.0


class TestSignalComputerEmpty:
    def test_empty_snapshots_returns_defaults(self):
        sc = SignalComputer()
        pool = make_pool()
        signals = sc.compute({}, pool)
        assert signals.backlog_pressure == 0.0
        assert signals.fairness_index == 1.0
        assert signals.blocked_cases_missing_docs == 0


class TestBacklogPressure:
    def test_no_backlog_gives_zero_pressure(self):
        sc = SignalComputer()
        pool = make_pool(total=10, available=10)
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=0)}
        signals = sc.compute(snap, pool, capacity_per_day=10.0)
        assert signals.backlog_pressure == 0.0

    def test_high_backlog_gives_high_pressure(self):
        sc = SignalComputer()
        pool = make_pool(total=5, available=5)
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=1000)}
        signals = sc.compute(snap, pool, capacity_per_day=5.0)
        assert signals.backlog_pressure > 0.8

    def test_backlog_pressure_bounded_at_one(self):
        sc = SignalComputer()
        pool = make_pool(total=1, available=1)
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=99999)}
        signals = sc.compute(snap, pool, capacity_per_day=1.0)
        assert signals.backlog_pressure <= 1.0


class TestSLARiskScore:
    def test_zero_risk_when_all_cases_fresh(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=10, sla_risk=0.0)}
        signals = sc.compute(snap, pool)
        assert signals.sla_risk_score == 0.0

    def test_full_risk_when_all_cases_at_deadline(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=10, sla_risk=1.0)}
        signals = sc.compute(snap, pool)
        assert abs(signals.sla_risk_score - 1.0) < 0.01

    def test_sla_risk_bounded(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=5, sla_risk=0.99)}
        signals = sc.compute(snap, pool)
        assert 0.0 <= signals.sla_risk_score <= 1.0


class TestFairnessIndex:
    def test_single_service_fairness_is_one(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=5, completed_today=3)}
        signals = sc.compute(snap, pool)
        assert signals.fairness_index == 1.0

    def test_equal_completion_rates_fairness_is_one(self):
        sc = SignalComputer()
        pool = make_pool()
        snaps = {
            ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=5, completed_today=5),
            ServiceType.LAND_REGISTRATION.value:
                make_snapshot(ServiceType.LAND_REGISTRATION,
                              total_pending=5, completed_today=5),
        }
        signals = sc.compute(snaps, pool)
        assert abs(signals.fairness_index - 1.0) < 0.05

    def test_unequal_completion_rates_reduce_fairness(self):
        sc = SignalComputer()
        pool = make_pool()
        snaps = {
            ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=10, completed_today=10),
            ServiceType.LAND_REGISTRATION.value:
                make_snapshot(ServiceType.LAND_REGISTRATION,
                              total_pending=10, completed_today=0),
        }
        signals = sc.compute(snaps, pool)
        assert signals.fairness_index < 1.0

    def test_fairness_bounded(self):
        sc = SignalComputer()
        pool = make_pool()
        snaps = {
            "a": make_snapshot(ServiceType.INCOME_CERTIFICATE,
                               total_pending=100, completed_today=100),
            "b": make_snapshot(ServiceType.LAND_REGISTRATION,
                               total_pending=100, completed_today=0),
        }
        signals = sc.compute(snaps, pool)
        assert 0.0 <= signals.fairness_index <= 1.0


class TestResourceUtilization:
    def test_fully_allocated_gives_one(self):
        sc = SignalComputer()
        pool = make_pool(total=10, available=10,
                         allocated={"income_certificate": 10})
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=5)}
        signals = sc.compute(snap, pool)
        assert abs(signals.resource_utilization - 1.0) < 0.01

    def test_zero_allocation_gives_zero_utilization(self):
        sc = SignalComputer()
        pool = make_pool(total=10, available=10, allocated={})
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=5)}
        signals = sc.compute(snap, pool)
        assert signals.resource_utilization == 0.0

    def test_utilization_bounded(self):
        sc = SignalComputer()
        pool = make_pool(total=10, available=10,
                         allocated={"income_certificate": 99})
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=5)}
        signals = sc.compute(snap, pool)
        assert 0.0 <= signals.resource_utilization <= 1.0


class TestDigitalIntakeRatio:
    def test_all_digital_gives_one(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE, total_pending=5)}
        signals = sc.compute(snap, pool,
                             todays_arrivals=10, digital_arrivals=10)
        assert signals.digital_intake_ratio == 1.0

    def test_no_arrivals_gives_half(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE)}
        signals = sc.compute(snap, pool, todays_arrivals=0, digital_arrivals=0)
        assert signals.digital_intake_ratio == 0.5

    def test_ratio_bounded(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE)}
        signals = sc.compute(snap, pool, todays_arrivals=5, digital_arrivals=5)
        assert 0.0 <= signals.digital_intake_ratio <= 1.0


class TestBlockedAndFieldLoad:
    def test_blocked_cases_aggregated_across_services(self):
        sc = SignalComputer()
        pool = make_pool()
        snaps = {
            ServiceType.INCOME_CERTIFICATE.value:
                make_snapshot(ServiceType.INCOME_CERTIFICATE,
                              total_pending=10, blocked_missing=3),
            ServiceType.LAND_REGISTRATION.value:
                make_snapshot(ServiceType.LAND_REGISTRATION,
                              total_pending=8, blocked_missing=2),
        }
        signals = sc.compute(snaps, pool)
        assert signals.blocked_cases_missing_docs == 5

    def test_field_verification_load_fraction(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.PASSPORT.value:
                make_snapshot(ServiceType.PASSPORT,
                              total_pending=10, field_pending=4)}
        signals = sc.compute(snap, pool)
        assert abs(signals.field_verification_load - 0.4) < 0.05

    def test_field_load_bounded(self):
        sc = SignalComputer()
        pool = make_pool()
        snap = {ServiceType.PASSPORT.value:
                make_snapshot(ServiceType.PASSPORT,
                              total_pending=5, field_pending=5)}
        signals = sc.compute(snap, pool)
        assert 0.0 <= signals.field_verification_load <= 1.0
