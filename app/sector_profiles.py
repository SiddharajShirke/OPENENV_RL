"""
sector_profiles.py — Phase 2 update: enrichment type, probability, delay range per service.
"""

from app.models import (
    DocEnrichmentType, SectorProfile, ServiceType, UrgencyProfile
)

INCOME_CERTIFICATE_PROFILE = SectorProfile(
    service_type=ServiceType.INCOME_CERTIFICATE,
    sector_name="Revenue Sector — Income Certificate",
    missing_docs_probability=0.45,
    doc_defect_rate_digital=0.30,
    doc_defect_rate_paper=0.65,
    field_verification_probability=0.30,
    manual_scrutiny_intensity=0.60,
    decision_backlog_sensitivity=0.70,
    system_dependency_risk=0.20,
    sla_days=21,
    urgency_profile=UrgencyProfile.MODERATE,
    base_processing_rate=8.0,
    field_verification_days=3,
    doc_enrichment_type=DocEnrichmentType.NONE,
    doc_enrichment_probability=0.0,
    doc_enrichment_delay_days_min=1,
    doc_enrichment_delay_days_max=2,
)

LAND_REGISTRATION_PROFILE = SectorProfile(
    service_type=ServiceType.LAND_REGISTRATION,
    sector_name="Land Sector — 7/12 Mutation",
    missing_docs_probability=0.35,
    doc_defect_rate_digital=0.25,
    doc_defect_rate_paper=0.55,
    field_verification_probability=0.65,
    manual_scrutiny_intensity=0.75,
    decision_backlog_sensitivity=0.85,
    system_dependency_risk=0.55,
    sla_days=30,
    urgency_profile=UrgencyProfile.LOW_BUT_STICKY,
    base_processing_rate=4.0,
    field_verification_days=5,
    doc_enrichment_type=DocEnrichmentType.PAST_LAND_RECORDS,
    doc_enrichment_probability=0.70,
    doc_enrichment_delay_days_min=2,
    doc_enrichment_delay_days_max=5,   # REVENUE_DB_DELAY event adds 1-2 more
)

CASTE_CERTIFICATE_PROFILE = SectorProfile(
    service_type=ServiceType.CASTE_CERTIFICATE,
    sector_name="Revenue Sector — Caste Certificate",
    missing_docs_probability=0.40,
    doc_defect_rate_digital=0.25,
    doc_defect_rate_paper=0.60,
    field_verification_probability=0.35,
    manual_scrutiny_intensity=0.65,
    decision_backlog_sensitivity=0.65,
    system_dependency_risk=0.25,
    sla_days=21,
    urgency_profile=UrgencyProfile.MODERATE,
    base_processing_rate=7.0,
    field_verification_days=3,
    doc_enrichment_type=DocEnrichmentType.FAMILY_CASTE_HISTORY,
    doc_enrichment_probability=0.55,
    doc_enrichment_delay_days_min=2,
    doc_enrichment_delay_days_max=4,
)

BIRTH_CERTIFICATE_PROFILE = SectorProfile(
    service_type=ServiceType.BIRTH_CERTIFICATE,
    sector_name="Municipal Sector — Birth Certificate",
    missing_docs_probability=0.20,
    doc_defect_rate_digital=0.15,
    doc_defect_rate_paper=0.35,
    field_verification_probability=0.05,
    manual_scrutiny_intensity=0.30,
    decision_backlog_sensitivity=0.40,
    system_dependency_risk=0.30,
    sla_days=7,
    urgency_profile=UrgencyProfile.HIGH,
    base_processing_rate=15.0,
    field_verification_days=1,
    doc_enrichment_type=DocEnrichmentType.NONE,
    doc_enrichment_probability=0.0,
    doc_enrichment_delay_days_min=1,
    doc_enrichment_delay_days_max=1,
)

PASSPORT_PROFILE = SectorProfile(
    service_type=ServiceType.PASSPORT,
    sector_name="National Sector — Passport",
    missing_docs_probability=0.25,
    doc_defect_rate_digital=0.20,
    doc_defect_rate_paper=0.50,
    field_verification_probability=0.90,
    manual_scrutiny_intensity=0.80,
    decision_backlog_sensitivity=0.75,
    system_dependency_risk=0.35,
    sla_days=30,
    urgency_profile=UrgencyProfile.HIGH,
    base_processing_rate=5.0,
    field_verification_days=14,
    doc_enrichment_type=DocEnrichmentType.POLICE_VERIFICATION,
    doc_enrichment_probability=0.85,
    doc_enrichment_delay_days_min=7,
    doc_enrichment_delay_days_max=14,
)

GST_REGISTRATION_PROFILE = SectorProfile(
    service_type=ServiceType.GST_REGISTRATION,
    sector_name="Tax Sector — GST Registration",
    missing_docs_probability=0.30,
    doc_defect_rate_digital=0.20,
    doc_defect_rate_paper=0.50,
    field_verification_probability=0.20,
    manual_scrutiny_intensity=0.55,
    decision_backlog_sensitivity=0.60,
    system_dependency_risk=0.45,
    sla_days=7,
    urgency_profile=UrgencyProfile.HIGH,
    base_processing_rate=10.0,
    field_verification_days=2,
    doc_enrichment_type=DocEnrichmentType.TAX_RECORD_CROSS_CHECK,
    doc_enrichment_probability=0.50,
    doc_enrichment_delay_days_min=1,
    doc_enrichment_delay_days_max=3,
)

DRIVING_LICENSE_PROFILE = SectorProfile(
    service_type=ServiceType.DRIVING_LICENSE,
    sector_name="Transport Sector — Driving License",
    missing_docs_probability=0.28,
    doc_defect_rate_digital=0.18,
    doc_defect_rate_paper=0.45,
    field_verification_probability=0.40,
    manual_scrutiny_intensity=0.50,
    decision_backlog_sensitivity=0.55,
    system_dependency_risk=0.30,
    sla_days=14,
    urgency_profile=UrgencyProfile.MODERATE,
    base_processing_rate=12.0,
    field_verification_days=2,
    doc_enrichment_type=DocEnrichmentType.NONE,
    doc_enrichment_probability=0.0,
    doc_enrichment_delay_days_min=1,
    doc_enrichment_delay_days_max=1,
)

SECTOR_REGISTRY: dict = {
    ServiceType.INCOME_CERTIFICATE: INCOME_CERTIFICATE_PROFILE,
    ServiceType.LAND_REGISTRATION:  LAND_REGISTRATION_PROFILE,
    ServiceType.CASTE_CERTIFICATE:  CASTE_CERTIFICATE_PROFILE,
    ServiceType.BIRTH_CERTIFICATE:  BIRTH_CERTIFICATE_PROFILE,
    ServiceType.PASSPORT:           PASSPORT_PROFILE,
    ServiceType.GST_REGISTRATION:   GST_REGISTRATION_PROFILE,
    ServiceType.DRIVING_LICENSE:    DRIVING_LICENSE_PROFILE,
}

def get_sector_profile(service_type: ServiceType) -> SectorProfile:
    if service_type not in SECTOR_REGISTRY:
        raise KeyError(f"No SectorProfile for {service_type}")
    return SECTOR_REGISTRY[service_type]
