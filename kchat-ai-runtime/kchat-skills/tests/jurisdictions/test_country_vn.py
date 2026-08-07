"""Validate the kchat.jurisdiction.vn.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Law on Children 2016; Anti-Terrorism Law 2013.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "vn"


def test_run_all_structural_assertions(vn_overlay, vn_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        vn_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=vn_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(vn_overlay):
    override = A.get_override(vn_overlay, 1)
    assert override["severity_floor"] == 5, (
        "VN: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(vn_overlay):
    override = A.get_override(vn_overlay, 4)
    assert override["severity_floor"] == 4, (
        "VN: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(vn_overlay):
    assert vn_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(vn_overlay):
    assert vn_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(vn_overlay):
    classes = set(vn_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'ethnic_origin'}
    assert expected <= classes, (
        "VN protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(vn_overlay):
    langs = list(vn_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['vi'], (
        f"VN primary_languages must equal ['vi']; got {langs}"
    )


def test_user_notice_opt_out_allowed_false(vn_overlay):
    assert vn_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(vn_overlay):
    rules = vn_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "vn_nec_authority_v1"


def test_pack_passes_anti_misuse_validation(vn_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(vn_overlay)
    assert report.passed, f"VN pack failed anti-misuse: {report.errors}"
