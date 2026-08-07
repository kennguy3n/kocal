"""Validate the kchat.jurisdiction.it.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Codice Penale protection of minors; Legge Mancino (hate / fascist apologia); anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "it"


def test_run_all_structural_assertions(it_overlay, it_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        it_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=it_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(it_overlay):
    override = A.get_override(it_overlay, 1)
    assert override["severity_floor"] == 5, (
        "IT: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(it_overlay):
    override = A.get_override(it_overlay, 4)
    assert override["severity_floor"] == 4, (
        "IT: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(it_overlay):
    assert it_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(it_overlay):
    assert it_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(it_overlay):
    classes = set(it_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "IT protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(it_overlay):
    langs = list(it_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['it'], (
        f"IT primary_languages must equal ['it']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(it_overlay):
    assert it_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(it_overlay):
    rules = it_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "it_ministero_interno_authority_v1"


def test_pack_passes_anti_misuse_validation(it_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(it_overlay)
    assert report.passed, f"IT pack failed anti-misuse: {report.errors}"
