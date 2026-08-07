"""Validate the kchat.jurisdiction.us.guardrail.v1 country pack.

Mirrors the structural pattern from
``test_archetype_strict_adult.py``; country-specific overrides
match the ones declared in the overlay's docstring (federal CSAM
statute, FTO list, FTC / wire-fraud framework).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "us"


# ---------------------------------------------------------------------------
# Shared structural assertions.
# ---------------------------------------------------------------------------
def test_run_all_structural_assertions(us_overlay, us_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        us_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=us_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


# ---------------------------------------------------------------------------
# Country-specific overrides.
# ---------------------------------------------------------------------------
def test_category_1_child_safety_floor_5(us_overlay):
    override = A.get_override(us_overlay, 1)
    assert override["severity_floor"] == 5, (
        "US: CHILD_SAFETY override must pin severity_floor at 5 "
        "(federal CSAM statute, 18 USC §2251 / §2252)"
    )


def test_category_4_extremism_floor_4(us_overlay):
    override = A.get_override(us_overlay, 4)
    assert override["severity_floor"] == 4, (
        "US: EXTREMISM override must be severity_floor 4 (US-designated "
        "FTO / 18 USC §2339B)"
    )


def test_category_7_scam_fraud_floor_3(us_overlay):
    override = A.get_override(us_overlay, 7)
    assert override["severity_floor"] == 3, (
        "US: SCAM_FRAUD override must be severity_floor 3 (FTC / wire-fraud)"
    )


# ---------------------------------------------------------------------------
# Local definitions — federal alcohol / tobacco minima.
# ---------------------------------------------------------------------------
def test_legal_age_marketplace_alcohol_is_21(us_overlay):
    assert us_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_21(us_overlay):
    assert us_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 21


def test_protected_classes_match_federal_civil_rights(us_overlay):
    classes = set(us_overlay["local_definitions"]["protected_classes"])
    expected = {
        "race",
        "color",
        "religion",
        "sex",
        "national_origin",
        "disability",
        "age_40_plus",
        "genetic_information",
    }
    assert expected <= classes, (
        f"US protected_classes must include the federal civil-rights "
        f"enumeration; missing: {expected - classes}"
    )


def test_user_notice_opt_out_allowed_true(us_overlay):
    assert us_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_have_authority(us_overlay):
    rules = us_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "us_fec_authority_v1"
    assert rules["civic_window_open"]
    assert rules["civic_window_close"]


# ---------------------------------------------------------------------------
# Anti-misuse — pack must pass the validator the compiler runs.
# ---------------------------------------------------------------------------
def test_pack_passes_anti_misuse_validation(us_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(us_overlay)
    assert report.passed, f"US pack failed anti-misuse: {report.errors}"
