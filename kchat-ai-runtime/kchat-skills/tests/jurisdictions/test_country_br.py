"""Validate the kchat.jurisdiction.br.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring (ECA, Lei 7.716/89, TSE election rules).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "br"


def test_run_all_structural_assertions(br_overlay, br_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        br_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=br_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(br_overlay):
    override = A.get_override(br_overlay, 1)
    assert override["severity_floor"] == 5, (
        "BR: CHILD_SAFETY override must pin severity_floor at 5 (ECA, Lei 8.069/90)"
    )


def test_category_6_hate_floor_4(br_overlay):
    override = A.get_override(br_overlay, 6)
    assert override["severity_floor"] == 4, (
        "BR: HATE override must be severity_floor 4 (Lei 7.716/89 + Lei 14.532/23)"
    )


def test_category_14_misinformation_civic_floor_3(br_overlay):
    override = A.get_override(br_overlay, 14)
    assert override["severity_floor"] == 3, (
        "BR: MISINFORMATION_CIVIC override must be severity_floor 3 "
        "(TSE Resoluções 23.610/2019 / 23.732/2024)"
    )


def test_primary_languages_pt_br(br_overlay):
    langs = list(br_overlay["local_language_assets"]["primary_languages"])
    assert "pt-BR" in langs, "BR primary_languages must include pt-BR"


def test_protected_classes_constituicao_federal(br_overlay):
    classes = set(br_overlay["local_definitions"]["protected_classes"])
    expected = {"race", "color", "sex", "religion", "national_origin", "age", "disability"}
    assert expected <= classes, (
        f"BR protected_classes must include the Constituição Federal "
        f"enumeration; missing: {expected - classes}"
    )


def test_election_rules_reference_tse(br_overlay):
    rules = br_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "br_tse_authority_v1"


def test_pack_passes_anti_misuse_validation(br_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(br_overlay)
    assert report.passed, f"BR pack failed anti-misuse: {report.errors}"
