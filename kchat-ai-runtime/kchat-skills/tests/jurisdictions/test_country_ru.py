"""Validate the kchat.jurisdiction.ru.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Federal Law 124-FZ child protection; Federal Law 114-FZ anti-extremism (Roskomnadzor enforcement).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ru"


def test_run_all_structural_assertions(ru_overlay, ru_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ru_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ru_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ru_overlay):
    override = A.get_override(ru_overlay, 1)
    assert override["severity_floor"] == 5, (
        "RU: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ru_overlay):
    override = A.get_override(ru_overlay, 4)
    assert override["severity_floor"] == 4, (
        "RU: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ru_overlay):
    assert ru_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ru_overlay):
    assert ru_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ru_overlay):
    classes = set(ru_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "RU protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ru_overlay):
    langs = list(ru_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ru'], (
        f"RU primary_languages must equal ['ru']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ru_overlay):
    assert ru_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ru_overlay):
    rules = ru_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ru_cik_authority_v1"


def test_pack_passes_anti_misuse_validation(ru_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ru_overlay)
    assert report.passed, f"RU pack failed anti-misuse: {report.errors}"
