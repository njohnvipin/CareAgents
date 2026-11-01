"""JWT authentication and role-based access.

Roles:
  member - may only access their own record (member_id claim)
  staff  - may act on behalf of any member; every access is audited
  admin  - operational endpoints (ingestion, status)
In production, tokens come from your IdP (Okta, Entra ID, Cognito...) and are verified with its
public key (RS256/ES256). HS256 with a shared secret is for development and service-to-service.
"""

from __future__ import annotations

from typing import Literal

import jwt
from pydantic import BaseModel

from careagents.config import Settings

Role = Literal["member", "staff", "admin"]


class Principal(BaseModel):
    subject: str
    role: Role
    member_id: str | None = None


class AuthError(Exception):
    pass


class TokenVerifier:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.jwt_public_key:
            self.key: str = settings.jwt_public_key
        elif settings.jwt_secret:
            self.key = settings.jwt_secret.get_secret_value()
        else:
            self.key = ""

    def verify(self, token: str) -> Principal:
        if not self.key:
            raise AuthError("authentication is not configured")
        try:
            claims = jwt.decode(token, self.key, algorithms=self.s.jwt_algorithms, audience=self.s.jwt_audience,
                                issuer=self.s.jwt_issuer, options={"require": ["exp", "sub", "aud"]}, leeway=30)
        except jwt.PyJWTError as exc:
            raise AuthError("invalid token") from exc
        role = claims.get("role")
        if role not in ("member", "staff", "admin"):
            raise AuthError("missing or unknown role")
        member_id = claims.get("member_id")
        if role == "member" and not member_id:
            raise AuthError("member token without member_id")
        return Principal(subject=str(claims["sub"]), role=role, member_id=member_id)
