import pytest

from evals.run_eval import evaluate
from verdict.checker import GroundingChecker, check_against, split_claims
from verdict.citations import LocalVerifier, extract_citations
from verdict.judge import parse_verdict
from verdict.report import to_html
from verdict.text import modality, quantities, split_sentences, tokens

CONTRACT = """
4.1 Fees. Customer shall pay Provider a monthly fee of $12,500. Payment is due within thirty (30) days of the invoice date.
9.2 Termination. Either party may terminate this Agreement upon ninety (90) days' prior written notice.
7.2 Survival. Confidentiality obligations survive termination for three (3) years.
PTO. Unused paid time off does not roll over.
"""
KNOWN = {"550 U.S. 544": "Bell Atlantic Corp. v. Twombly"}


@pytest.fixture
def checker():
    return GroundingChecker({"contract": CONTRACT}, LocalVerifier(KNOWN))


# ---- text handling ---------------------------------------------------------

def test_sentence_splitter_respects_legal_abbreviations():
    text = "See Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007). Courts agree. Hosts respond by 10 p.m. Monday."
    assert split_sentences(text) == [
        "See Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007).",
        "Courts agree.",
        "Hosts respond by 10 p.m. Monday.",
    ]


def test_quantities_normalize_words_parens_money_and_units():
    q = quantities("Payment is due within thirty (30) days. Fee of $12,500 and 1.5% interest for twenty-four months.")
    assert (30.0, "days") in q
    assert (12500.0, "$") in q
    assert (1.5, "%") in q
    assert (24.0, "months") in q


def test_decimals_are_not_mistaken_for_section_numbers():
    assert (1.25, "days") in quantities("Employees accrue 1.25 days of paid time off.")
    assert not any(v == 4.2 for v, _ in quantities("4.2 Late Payment. Interest applies."))


def test_modality_treats_prohibitions_as_obligations():
    assert modality(tokens("Hosts must respond")) == "obligation"
    assert modality(tokens("Employees may work remotely")) == "permission"
    assert modality(tokens("No person may operate a rental")) == "obligation"
    assert modality(tokens("The rental may not be occupied")) == "obligation"


# ---- individual checks -----------------------------------------------------

def test_wrong_deadline_is_caught():
    reasons = check_against("Payment is due within 45 days.", "Payment is due within thirty (30) days.")
    assert any("45 days" in r for r in reasons)


def test_number_must_match_its_unit():
    src = "A permit is valid for one (1) year. Occupancy is two adults per bedroom."
    assert check_against("A permit is valid for two years.", src)
    assert not check_against("A permit is valid for one year.", src)


def test_flipped_negation_both_directions():
    assert check_against("Unused paid time off rolls over.", "Unused paid time off does not roll over.")
    assert check_against("Obligations do not survive termination.", "Obligations survive termination.")


def test_double_negative_requirement_is_not_misread():
    src = "No person may operate a short-term rental without a permit."
    assert not check_against("Operating a short-term rental requires a permit.", src)


def test_obligation_vs_permission():
    assert check_against("Employees must work remotely three days per week.",
                         "Employees may work remotely up to three days per week.")


def test_added_exclusivity():
    assert check_against("Only the Customer may terminate.", "Either party may terminate.")


# ---- end to end ------------------------------------------------------------

def test_supported_claims_pass(checker):
    for claim in ["Customer pays a monthly fee of $12,500.",
                  "Either party may terminate with 90 days' prior written notice.",
                  "Confidentiality obligations survive termination for three years."]:
        r = checker.check_claim(claim)
        assert r.supported, (claim, r.reasons)
        assert r.evidence is not None


def test_fabricated_obligation_fails(checker):
    r = checker.check_claim("Provider must maintain cyber insurance of at least $5 million.")
    assert not r.supported
    assert any("$5,000,000" in x for x in r.reasons)


def test_citation_extraction_and_fake_case():
    cites = extract_citations("Harrington v. Delmont Logistics, 612 U.S. 118 (2021), held otherwise.")
    assert len(cites) == 1 and cites[0].volume == 612 and cites[0].year == 2021
    assert cites[0].case_name == "Harrington v. Delmont Logistics"
    checks = LocalVerifier(KNOWN).verify(cites)
    assert checks[0].status == "not_found"


def test_real_citation_with_wrong_case_name_is_caught():
    cites = extract_citations("Smith v. Jones, 550 U.S. 544 (2007).")
    assert LocalVerifier(KNOWN).verify(cites)[0].status == "not_found"


def test_split_claims_attaches_bare_citations_and_skips_framing():
    answer = ("Here is a summary of the law. Summary judgment requires no genuine dispute of material fact. "
              "Celotex Corp. v. Catrett, 477 U.S. 317 (1986). Courts apply this broadly.")
    claims = split_claims(answer)
    assert len(claims) == 2
    assert claims[0].endswith("477 U.S. 317 (1986).")


# ---- judge -----------------------------------------------------------------

class FakeJudge:
    def __init__(self, label):
        self.label, self.calls = label, 0

    def judge(self, claim, passages):
        self.calls += 1
        return {"label": self.label, "reason": "fake", "quote": passages[0] if passages else ""}


def test_judge_can_flag_a_claim_that_passed_the_checks():
    judge = FakeJudge("unsupported")
    c = GroundingChecker({"contract": CONTRACT}, judge=judge)
    r = c.check_claim("Customer pays a monthly fee of $12,500.")
    assert not r.supported and judge.calls == 1


def test_judge_never_overrides_a_hard_failure():
    judge = FakeJudge("supported")
    c = GroundingChecker({"contract": CONTRACT}, judge=judge)
    r = c.check_claim("Payment is due within 45 days of the invoice date.")
    assert not r.supported and judge.calls == 0


def test_parse_verdict_tolerates_code_fences():
    assert parse_verdict('```json\n{"label": "Supported", "reason": "ok"}\n```')["label"] == "supported"
    assert parse_verdict("no json here")["label"] == "unknown"


# ---- reporting and benchmark ----------------------------------------------

def test_html_report_escapes_and_renders(checker):
    report = checker.check_answer("Customer pays a monthly fee of $12,500. <script>x</script> is due in 45 days.")
    page = to_html(report)
    assert "<script>x</script>" not in page and "Unsupported" in page


def test_benchmark_floors():
    """Guards against regressions. The holdout floor is deliberately set to its
    first measured value: the heuristics were never tuned on it."""
    dev = evaluate(which="dev")
    holdout = evaluate(which="holdout")
    assert dev["recall"] >= 0.85 and dev["precision"] >= 0.9
    assert holdout["recall"] >= 0.85 and holdout["precision"] >= 0.6
