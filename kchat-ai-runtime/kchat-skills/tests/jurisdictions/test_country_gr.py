"""Validate the kchat.jurisdiction.gr.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Greek Penal Code child protection; Law 3251/2004 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "gr"


def test_run_all_structural_assertions(gr_overlay, gr_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        gr_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=gr_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(gr_overlay):
    override = A.get_override(gr_overlay, 1)
    assert override["severity_floor"] == 5, (
        "GR: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(gr_overlay):
    override = A.get_override(gr_overlay, 4)
    assert override["severity_floor"] == 4, (
        "GR: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(gr_overlay):
    assert gr_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(gr_overlay):
    assert gr_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(gr_overlay):
    classes = set(gr_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "GR protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(gr_overlay):
    langs = list(gr_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['el'], (
        f"GR primary_languages must equal ['el']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(gr_overlay):
    assert gr_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(gr_overlay):
    rules = gr_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "gr_ypes_authority_v1"


def test_pack_passes_anti_misuse_validation(gr_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(gr_overlay)
    assert report.passed, f"GR pack failed anti-misuse: {report.errors}"
