"""Deterministic cost-sharing math. The LLM explains these numbers; it never computes them."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class CostEstimate:
    charge: Decimal
    deductible_applied: Decimal
    coinsurance_amount: Decimal
    member_pays: Decimal
    plan_pays: Decimal
    oop_cap_applied: bool


def estimate_member_cost(charge: Decimal, deductible_remaining: Decimal, coinsurance_member_pct: Decimal,
                         oop_remaining: Decimal) -> CostEstimate:
    if charge < 0 or deductible_remaining < 0 or oop_remaining < 0:
        raise ValueError("amounts must be non-negative")
    if not (Decimal(0) <= coinsurance_member_pct <= Decimal(100)):
        raise ValueError("coinsurance must be between 0 and 100")
    charge = money(charge)
    deductible_applied = money(min(deductible_remaining, charge))
    coinsurance = money((charge - deductible_applied) * coinsurance_member_pct / Decimal(100))
    uncapped = deductible_applied + coinsurance
    member = money(min(uncapped, oop_remaining))
    return CostEstimate(
        charge=charge,
        deductible_applied=deductible_applied,
        coinsurance_amount=coinsurance,
        member_pays=member,
        plan_pays=money(charge - member),
        oop_cap_applied=member < uncapped,
    )
