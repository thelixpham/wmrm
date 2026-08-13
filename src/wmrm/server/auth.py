"""Bearer token for calls into this pod.

One token per pod, issued by the control plane. Not shared across pods: this machine is
rented, and a token that unlocks all four of them turns one compromised pod into control
of the whole fleet -- the same reasoning that keeps R2 credentials off the pod at all.

`/live` is the one route without this, so a platform health check can use it. It returns
no information, which is why that is safe.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from .config import Config


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_token(request: Request) -> None:
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

    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value or not _constant_time_eq(value, cfg.token):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")
