from collections.abc import Awaitable, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from careagents.auth.security import AuthError, Principal
from careagents.container import Container
from careagents.metrics import RATE_LIMITED

bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


async def get_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    container: Container = Depends(get_container),
) -> Principal:
    if container.settings.auth_disabled:
        return Principal(subject="dev-user", role="member", member_id=container.settings.dev_member_id)
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return container.verifier.verify(creds.credentials)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


def require_role(*roles: str) -> Callable[..., Awaitable[Principal]]:
    async def dep(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
        return principal
    return dep


async def rate_limit(principal: Principal = Depends(get_principal),
                     container: Container = Depends(get_container)) -> None:
    allowed, retry_after = await container.limiter.hit(f"chat:{principal.subject}",
                                                       container.settings.rate_limit_per_minute, 60)
    if not allowed:
        RATE_LIMITED.inc()
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                            headers={"Retry-After": str(retry_after)})
