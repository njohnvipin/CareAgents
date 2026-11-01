"""Mint short-lived dev tokens: python -m careagents.devtools --role member --member-id SARAH-1001"""

import argparse
import time

import jwt

from careagents.config import get_settings


def mint(role: str, subject: str, member_id: str | None, ttl_s: int = 3600) -> str:
    s = get_settings()
    if s.env in ("staging", "prod") or not s.jwt_secret:
        raise SystemExit("dev tokens need CA_JWT_SECRET and a dev/test environment")
    now = int(time.time())
    claims = {"sub": subject, "role": role, "aud": s.jwt_audience, "iat": now, "exp": now + ttl_s}
    if member_id:
        claims["member_id"] = member_id
    if s.jwt_issuer:
        claims["iss"] = s.jwt_issuer
    return jwt.encode(claims, s.jwt_secret.get_secret_value(), algorithm="HS256")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["member", "staff", "admin"], default="member")
    p.add_argument("--subject", default="dev-user")
    p.add_argument("--member-id")
    a = p.parse_args()
    print(mint(a.role, a.subject, a.member_id))
