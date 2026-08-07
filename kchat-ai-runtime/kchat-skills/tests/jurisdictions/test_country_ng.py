"""Validate the kchat.jurisdiction.ng.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Child Rights Act 2003; Terrorism Prevention Act; Cybercrimes Act 2015.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ng"


def test_run_all_structural_assertions(ng_overlay, ng_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ng_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ng_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ng_overlay):
    override = A.get_override(ng_overlay, 1)
    assert override["severity_floor"] == 5, (
        "NG: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ng_overlay):
    override = A.get_override(ng_overlay, 4)
    assert override["severity_floor"] == 4, (
        "NG: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ng_overlay):
    assert ng_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ng_overlay):
    assert ng_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ng_overlay):
    classes = set(ng_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'ethnic_origin'}
    assert expected <= classes, (
        "NG protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ng_overlay):
    langs = list(ng_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en'], (
        f"NG primary_languages must equal ['en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ng_overlay):
    assert ng_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ng_overlay):
    rules = ng_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ng_inec_authority_v1"


def test_pack_passes_anti_misuse_validation(ng_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ng_overlay)
    assert report.passed, f"NG pack failed anti-misuse: {report.errors}"
