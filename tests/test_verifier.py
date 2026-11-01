from careagents.agents.state import CaseState, Evidence, Fact
from careagents.agents.verifier import grounded_numbers, ungrounded_amounts


def state():
    s = CaseState(case_id="c", member_id="m", actor_id="a", question="Is a $1,000 MRI covered?")
    s.add_fact(Fact(key="estimate.x.member_pays", value="440.00", source="t"))
    s.add_evidence(Evidence(source="p", doc_type="policy", chunk_index=0, text="plan pays 80%, deductible $500"))
    return s


def test_accepts_grounded_amounts_in_any_format():
    allowed = grounded_numbers(state())
    assert ungrounded_amounts("You pay $440, or $440.00; the plan pays 80% after $500. Charge $1,000.", allowed) == []


def test_flags_invented_amounts():
    allowed = grounded_numbers(state())
    assert ungrounded_amounts("You pay $450 and 30%.", allowed) == ["$450", "30%"]


def test_conflicting_facts_are_recorded():
    s = state()
    s.add_fact(Fact(key="estimate.x.member_pays", value="500.00", source="other"))
    assert s.conflicts
