"""Validate the kchat.jurisdiction.tr.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Türk Ceza Kanunu (TCK) child protection; TMK anti-terrorism; Law 5651 internet content.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "tr"


def test_run_all_structural_assertions(tr_overlay, tr_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        tr_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=tr_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(tr_overlay):
    override = A.get_override(tr_overlay, 1)
    assert override["severity_floor"] == 5, (
        "TR: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(tr_overlay):
    override = A.get_override(tr_overlay, 4)
    assert override["severity_floor"] == 4, (
        "TR: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(tr_overlay):
    assert tr_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(tr_overlay):
    assert tr_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(tr_overlay):
    classes = set(tr_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'language', 'religion', 'disability'}
    assert expected <= classes, (
        "TR protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(tr_overlay):
    langs = list(tr_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['tr'], (
        f"TR primary_languages must equal ['tr']; got {langs}"
    )


def test_user_notice_opt_out_allowed_false(tr_overlay):
    assert tr_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(tr_overlay):
    rules = tr_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "tr_ysk_authority_v1"


def test_pack_passes_anti_misuse_validation(tr_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(tr_overlay)
    assert report.passed, f"TR pack failed anti-misuse: {report.errors}"
