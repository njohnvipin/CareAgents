from decimal import Decimal as D

import pytest

from careagents.tools.billing_calculator import estimate_member_cost


def test_deductible_then_coinsurance():
    e = estimate_member_cost(D("1000"), D("300"), D("20"), D("4800"))
    assert (e.deductible_applied, e.coinsurance_amount, e.member_pays, e.plan_pays) == (
        D("300.00"), D("140.00"), D("440.00"), D("560.00"))
    assert not e.oop_cap_applied


def test_out_of_pocket_cap_limits_member_cost():
    e = estimate_member_cost(D("10000"), D("500"), D("20"), D("600"))
    assert e.member_pays == D("600.00") and e.plan_pays == D("9400.00") and e.oop_cap_applied


def test_charge_below_remaining_deductible():
    e = estimate_member_cost(D("120"), D("300"), D("20"), D("5000"))
    assert e.member_pays == D("120.00") and e.plan_pays == D("0.00")


def test_rounding_to_cents():
    e = estimate_member_cost(D("333.33"), D("0"), D("20"), D("5000"))
    assert e.member_pays == D("66.67")


@pytest.mark.parametrize("args", [(D("-1"), D("0"), D("20"), D("1")), (D("1"), D("0"), D("120"), D("1"))])
def test_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        estimate_member_cost(*args)
