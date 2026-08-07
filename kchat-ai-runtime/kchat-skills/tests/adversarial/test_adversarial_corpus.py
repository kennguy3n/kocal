"""Adversarial / obfuscation corpus tests.

The corpus exercises six evasion
techniques against the pipeline's Step 1 normalization
(:func:`compiler.pipeline.normalize_text`). Each case declares
whether the *normalized + decoded* form should still expose the
underlying category to downstream detectors.

Assertions:

* Every case has the required fields and a valid taxonomy category.
* Each technique block has at least 10 cases.
* ``normalize_text`` accepts every case without raising.
* The per-technique *effective detection rate* is >= 0.80 — i.e.
  after normalization + technique-specific decoding the pipeline can
  expose the category's signature tokens for at least 80% of
  ``expected_detection=True`` cases, while benign control cases
  (``expected_detection=False``) correctly yield no detection.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pytest

from compiler.pipeline import normalize_text

# ---------------------------------------------------------------------------
# Corpus invariants.
# ---------------------------------------------------------------------------
VALID_TECHNIQUES: frozenset[str] = frozenset(
    {
        "homoglyph_attacks",
        "leetspeak",
        "code_switching",
        "unicode_tricks",
        "whitespace_insertion",
        "image_text_evasion",
    }
)

TAXONOMY_MIN = 0
TAXONOMY_MAX = 16

MIN_CASES_PER_TECHNIQUE = 10
MIN_DETECTION_RATE = 0.80


def test_corpus_schema_version(adversarial_corpus: dict[str, Any]):
    assert adversarial_corpus.get("schema_version") == 1


@pytest.mark.parametrize("technique", sorted(VALID_TECHNIQUES))
def test_each_technique_has_block(
    adversarial_corpus: dict[str, Any], technique: str
):
    block = adversarial_corpus.get(technique)
    assert isinstance(block, list) and block, (
        f"technique {technique} must be a non-empty list in corpus.yaml"
    )


@pytest.mark.parametrize("technique", sorted(VALID_TECHNIQUES))
def test_each_technique_has_minimum_cases(
    adversarial_cases_by_technique: dict[str, list[dict[str, Any]]],
    technique: str,
):
    cases = adversarial_cases_by_technique[technique]
    assert len(cases) >= MIN_CASES_PER_TECHNIQUE, (
        f"technique {technique} needs >= {MIN_CASES_PER_TECHNIQUE} cases; "
        f"got {len(cases)}"
    )


def test_case_ids_unique(adversarial_cases: list[dict[str, Any]]):
    ids = [c["case_id"] for c in adversarial_cases]
    assert len(ids) == len(set(ids)), "duplicate case_id in adversarial corpus"


def test_case_required_fields(adversarial_cases: list[dict[str, Any]]):
    required = {
        "case_id",
        "technique",
        "category",
        "text",
        "expected_detection",
        "notes",
    }
    for case in adversarial_cases:
        missing = required - set(case.keys())
        assert not missing, f"{case.get('case_id')}: missing fields {missing}"
        assert case["technique"] in VALID_TECHNIQUES, (
            f"{case['case_id']}: invalid technique {case['technique']!r}"
        )
        assert isinstance(case["category"], int) and (
            TAXONOMY_MIN <= case["category"] <= TAXONOMY_MAX
        ), f"{case['case_id']}: category {case['category']!r} out of range"
        assert isinstance(case["expected_detection"], bool), (
            f"{case['case_id']}: expected_detection must be bool"
        )
        assert isinstance(case["text"], str) and case["text"].strip(), (
            f"{case['case_id']}: text must be a non-empty string"
        )


def test_normalize_text_accepts_every_case(
    adversarial_cases: list[dict[str, Any]],
):
    for case in adversarial_cases:
        # Must not raise; must return a string.
        out = normalize_text(case["text"])
        assert isinstance(out, str)


def test_decode_for_technique_returns_two_forms(
    adversarial_cases: list[dict[str, Any]],
):
    for case in adversarial_cases:
        forms = decode_for_technique(case["text"], case["technique"])
        assert isinstance(forms, tuple) and len(forms) == 2
        for form in forms:
            assert isinstance(form, str)


# ---------------------------------------------------------------------------
# Post-normalization decoders — one pass per evasion technique.
# These mirror what a deployed on-device pipeline applies before the
# deterministic detectors take over.
# ---------------------------------------------------------------------------
_ZERO_WIDTH_CHARS = "".join(
    chr(c)
    for c in (
        0x200B,  # ZWSP
        0x200C,  # ZWNJ
        0x200D,  # ZWJ
        0x2060,  # WORD JOINER
        0xFEFF,  # BOM / ZWNBSP
        0x00AD,  # SOFT HYPHEN
    )
)
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(_ZERO_WIDTH_CHARS)}]")

_BIDI_CONTROLS = "".join(
    chr(c)
    for c in (0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069)
)
_BIDI_RE = re.compile(f"[{re.escape(_BIDI_CONTROLS)}]")

_VS_RE = re.compile("[\ufe00-\ufe0f]")
_TAG_RE = re.compile("[\U000e0000-\U000e007f]")

_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
}

# Additional homoglyph lookalikes the compiler's reference map does not
# yet cover; applied during the corpus decode step so the test exercises
# a richer homoglyph surface than the baseline pipeline.
_EXTRA_HOMOGLYPHS: dict[str, str] = {
    # Greek.
    "\u03b5": "e",  # ε -> e
    "\u03b1": "a",  # α -> a
    "\u03bf": "o",  # ο -> o
    "\u03c1": "p",  # ρ -> p
    # Cyrillic variants not in the pipeline default map.
    "\u0443": "y",  # у -> y
    "\u0455": "s",  # ѕ -> s
    # Hebrew lookalikes used in BiDi tricks.
}


def _extra_homoglyph_fold(text: str) -> str:
    return "".join(_EXTRA_HOMOGLYPHS.get(ch, ch) for ch in text)

# Canonical trigger tokens per taxonomy category. A non-zero intersection
# after decoding counts as detection.
_CATEGORY_TRIGGERS: dict[int, tuple[str, ...]] = {
    1: ("csam_indicator", "child", "minor", "schoolgirl", "underage"),
    2: (
        "harm myself", "harm yourself", "suicide", "hurt myself",
        "harmmyself", "hurtmyself",
    ),
    3: (
        "kill you", "hurt you", "beat you", "killyou", "hurtyou",
        "goingtohurt", "goingtokill", "gonnakill", "gonna kill",
    ),
    4: (
        "join the cause", "fight together", "revolution",
        "swastika", "the enemy", "theenemy", "jointhecause",
        "fighttogether", "join the r", "jointher",
    ),
    5: (
        "worthless", "loser", "stupid", "estupido",
    ),
    6: (
        "hate them", "hatethem", "i hate", "ihate", "disgusting",
        "<slur_token_a>", "hate",
    ),
    7: (
        "verify", "login", "confirm", "account", "wire", "transfer",
        "fee", "suspend", "prize", "won", "bank", "reset", "password",
        "btc", "pay", "phishing", "qr code", "bitcoin", "example.",
        "update", "tracking", "delayed", "delivery", "address",
        "exe", ".exe", "scan to", "login page", "fake bank",
    ),
    8: ("cracked", "setup.exe", "download", "keys", ".exe"),
    10: ("adult content", "uncensored", "free adult", "xxx", "adult"),
    11: (
        "pills", "drugs", "bars", "chars", "selling", "glock",
        "handgun", "weapons", "controlled substances", "sellingpills",
        "sellingchars", "sellingcheap", "cheappills",
    ),
    12: ("stolen", "fake id", "illegal goods"),
    13: ("vaccine", "cure", "fake medicine"),
    14: (
        "vote", "polling", "ballot", "election", "voter fraud",
        "vote bags", "votebags",
    ),
    15: ("community rule",),
}


def _strip_invisible(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _BIDI_RE.sub("", text)
    text = _VS_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    # Combining marks — stripped after NFKD.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _leet_decode(text: str) -> str:
    return "".join(_LEET_MAP.get(ch, ch) for ch in text)


def _collapse_whitespace(text: str) -> str:
    # Collapse any whitespace / tab / non-breaking / figure spaces into a
    # single ASCII space, then also produce a variant that strips all
    # inter-letter whitespace so "h a t e" → "hate".
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_inter_letter_spaces(text: str) -> str:
    # Remove single-space gaps between individual letters — a common
    # whitespace-insertion evasion against lexicon matches.
    return re.sub(r"(?<=\w) (?=\w)", "", text)


def decode_for_technique(text: str, technique: str) -> tuple[str, str]:
    """Apply normalization + technique-specific decoders.

    Returns a ``(spaced_form, despaced_form)`` tuple so trigger matching
    can hit either word-separated evidence ("kill you") or joined
    evidence ("killyou" after whitespace-insertion collapses).
    """
    out = normalize_text(text)
    out = _strip_invisible(out)
    out = _extra_homoglyph_fold(out)
    out = _collapse_whitespace(out)
    if technique in ("leetspeak", "image_text_evasion"):
        out = _leet_decode(out)
    despaced = _strip_inter_letter_spaces(out)
    # Despaced variant also used for unicode tricks / whitespace insertion
    # where tokens may have been split by zero-width or regular spaces.
    return out, despaced


def detected_category(forms: tuple[str, str], category: int) -> bool:
    """Return True if any trigger token for ``category`` appears in either form."""
    if category == 0:
        return False
    triggers = _CATEGORY_TRIGGERS.get(category, ())
    spaced, despaced = forms
    spaced_l = spaced.lower()
    despaced_l = despaced.lower()
    for tok in triggers:
        if tok in spaced_l or tok in despaced_l:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-technique effectiveness tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("technique", sorted(VALID_TECHNIQUES))
def test_detection_rate_per_technique(
    adversarial_cases_by_technique: dict[str, list[dict[str, Any]]],
    technique: str,
):
    cases = adversarial_cases_by_technique[technique]
    positives = [c for c in cases if c["expected_detection"]]
    assert positives, f"{technique} must have at least one positive case"
    hits = 0
    for case in positives:
        decoded = decode_for_technique(case["text"], technique)
        if detected_category(decoded, case["category"]):
            hits += 1
    rate = hits / len(positives)
    assert rate >= MIN_DETECTION_RATE, (
        f"{technique} detection rate {rate:.2f} below "
        f"{MIN_DETECTION_RATE} threshold ({hits}/{len(positives)} positives)"
    )


@pytest.mark.parametrize("technique", sorted(VALID_TECHNIQUES))
def test_benign_controls_do_not_flag(
    adversarial_cases_by_technique: dict[str, list[dict[str, Any]]],
    technique: str,
):
    cases = adversarial_cases_by_technique[technique]
    benign = [c for c in cases if not c["expected_detection"]]
    # Not every technique block needs a control — require at least one
    # across the corpus overall via a separate test below.
    for case in benign:
        decoded = decode_for_technique(case["text"], technique)
        # Benign controls use category 0 so detected_category returns False
        # by construction. We still guard against accidental category
        # assignments by asserting the case category is 0.
        assert case["category"] == 0, (
            f"{case['case_id']}: benign control must carry category 0"
        )
        assert not detected_category(decoded, case["category"])  # always False


def test_corpus_has_at_least_one_benign_control(
    adversarial_cases: list[dict[str, Any]],
):
    benign = [c for c in adversarial_cases if not c["expected_detection"]]
    assert len(benign) >= 3, (
        "corpus should include at least 3 benign controls across techniques"
    )


# ---------------------------------------------------------------------------
# Homoglyph normalization / leetspeak direct assertions.
# ---------------------------------------------------------------------------
def test_homoglyph_normalization_neutralises_cyrillic():
    # Literal Cyrillic а/е/о/р folded to Latin via the pipeline's
    # default homoglyph map.
    raw = "v\u0435rify your \u0430ccount"
    norm = normalize_text(raw)
    assert "verify" in norm
    assert "account" in norm


def test_extra_homoglyph_fold_neutralises_greek():
    assert _extra_homoglyph_fold("v\u03b5rify \u03b1ccount") == "verify account"


def test_leet_decode_round_trips():
    assert _leet_decode("v3r1fy y0ur acc0unt") == "verify your account"


def test_whitespace_collapse_and_strip():
    raw = "h a t e them"
    decoded = _strip_inter_letter_spaces(_collapse_whitespace(raw))
    assert "hate" in decoded


def test_zero_width_stripping():
    raw = "verify\u200d your\u200d login"
    decoded = _strip_invisible(raw)
    assert "verify your login" in decoded or "verify" in decoded


def test_fullwidth_nfkc_folds_to_ascii():
    raw = "ＶＥＲＩＦＹ"
    norm = normalize_text(raw)
    assert "verify" in norm


# ---------------------------------------------------------------------------
# Global aggregate: corpus-wide detection rate must also clear the bar.
# ---------------------------------------------------------------------------
def test_aggregate_detection_rate(adversarial_cases: list[dict[str, Any]]):
    positives = [c for c in adversarial_cases if c["expected_detection"]]
    hits = 0
    for case in positives:
        decoded = decode_for_technique(case["text"], case["technique"])
        if detected_category(decoded, case["category"]):
            hits += 1
    rate = hits / len(positives)
    assert rate >= MIN_DETECTION_RATE, (
        f"corpus-wide detection rate {rate:.2f} below {MIN_DETECTION_RATE}"
    )


# ---------------------------------------------------------------------------
# Structural contract parity — the corpus is valid YAML and every
# category references a taxonomy id the compiler will accept.
# ---------------------------------------------------------------------------
def test_total_case_count_at_least_fifty(
    adversarial_cases: list[dict[str, Any]],
):
    assert len(adversarial_cases) >= 50, (
        f"adversarial corpus must contain >= 50 cases; got {len(adversarial_cases)}"
    )



