"""Bearer token for calls into this pod.

One token per pod, set by whoever brought it up. Not shared across pods: this machine is
rented, and a token that unlocks all four of them turns one compromised pod into control
of the whole fleet.

Declared through `HTTPBearer` rather than by reading the header by hand, and that is not
cosmetic. A plain dependency function authenticates the route but tells OpenAPI nothing,
so Swagger UI renders no **Authorize** button -- the docs come up, every "Try it out"
returns 401, and there is nowhere on the page to put the token. Declaring the scheme is
what makes the interactive docs usable at all.

`/live` is the one route without this, so a platform health check can use it. It returns
an empty body, which is why that is safe.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Config

#: `auto_error=False` so the messages below are ours: "no token configured on this pod"
#: and "wrong token" send an operator to different places, and FastAPI's default says
#: neither.
bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="WMRM_POD_TOKEN",
    description=(
        "The value this pod was started with. Paste it here to use Try it out, and "
        "into the Pods page of the web app so the scheduler can reach the pod."
    ),
)


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """Raise 401 unless the request carries this pod's token.

    A pod with no token configured refuses everything rather than serving openly. The
    alternative -- open when unconfigured -- fails in the direction where a
    misconfigured deploy is indistinguishable from a working one, and this API can start
    GPU jobs and read presigned input URLs.
    """
    cfg: Config = request.app.state.cfg
    if not cfg.token:
        raise HTTPException(
            status_code=503,
            detail="this pod has no WMRM_POD_TOKEN set, so it refuses all requests",
        )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
        or not _constant_time_eq(credentials.credentials, cfg.token)
    ):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")
