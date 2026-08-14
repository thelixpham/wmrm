"""Reporting back to the control plane.

**The signing key is derived from this pod's own token**, not from a separate shared
secret. There used to be a `WEBHOOK_SECRET` that had to be configured identically on the
web app and on every pod, and it was wrong twice over:

- One value on every pod meant any pod could forge another pod's reports.
- "Must match on both sides" is the quietest possible misconfiguration. A single character
  out and every report is refused, the job runs correctly, and it looks stalled.

The pod token already satisfies both ends: the pod has it, and the control plane stores it
(encrypted) because it needs it to call in. So there is nothing to add and nothing to keep
in sync -- the value that already had to match is the only one there is.

Three properties the receiver depends on, so they are all implemented here rather than
assumed:

- **Signed over the raw body.** The signature covers `v1:{timestamp}:{body}` exactly as
  sent. Re-serialising JSON on either side changes key order and whitespace and breaks
  the signature for no visible reason, so the bytes are built once and both hashed and
  posted.
- **Timestamped inside the signature.** A replay window is worthless if the timestamp
  can be edited independently of what it protects.
- **Idempotent.** Every event carries a UUID, so the receiver can dedupe. This matters
  because the failure that actually happens is not a lost request, it is a lost
  *response* -- the write landed and we retry anyway.

A failed webhook must never fail the job. The job's truth is on this pod's disk and the
control plane can always come and ask; losing eight hours of GPU time because a POST
timed out would be absurd.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from typing import Any

import httpx

TIMEOUT = 10.0
MAX_ATTEMPTS = 4

#: Domain separator. The signing key is `SHA-256(token || LABEL)` rather than the token
#: itself, so the value that authenticates a request *to* this pod and the value that signs
#: a report *from* it are different bytes. Same root, two purposes, neither reusable as the
#: other -- which costs nothing here and is the difference between key derivation and key
#: reuse.
WEBHOOK_KEY_LABEL = b"wmrm-webhook-v1"


def webhook_key(pod_token: str) -> bytes:
    """The HMAC key for reports, derived from the pod's bearer token."""
    return hashlib.sha256(pod_token.encode() + WEBHOOK_KEY_LABEL).digest()


def sign(key: bytes | str, timestamp: int, body: bytes) -> str:
    """HMAC-SHA256 over `v1:{timestamp}:{body}`.

    Takes the already-derived key. Passing a str is accepted so a test can sign with a
    literal, but the server always passes bytes from `webhook_key`.
    """
    raw = key.encode() if isinstance(key, str) else key
    msg = b"v1:" + str(timestamp).encode() + b":" + body
    return hmac.new(raw, msg, hashlib.sha256).hexdigest()


class Notifier:
    """Posts events for one pod. Fire-and-forget by design."""

    def __init__(self, *, base_url: str | None, pod_token: str | None):
        self.base_url = (base_url or "").rstrip("/") or None
        # Derived once. The token itself is not kept, so nothing here can accidentally send
        # it: this object's job is to sign, not to authenticate outward.
        self.key = webhook_key(pod_token) if pod_token else None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.key)

    async def send(self, payload: dict[str, Any]) -> bool:
        """POST one event, retrying with backoff. Returns whether it landed."""
        if not self.enabled:
            return False

        payload = dict(payload)
        payload.setdefault("schema", 1)
        payload.setdefault("clientEventId", str(uuid.uuid4()))
        payload.setdefault("at", int(time.time()))

        # Compact and key-order-stable, because this exact byte string is what is signed.
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ts = int(payload["at"])
        assert self.key is not None                  # guarded by `enabled` above
        headers = {
            "content-type": "application/json",
            "x-wmrm-signature": f"v1={sign(self.key, ts, body)}",
            "x-wmrm-timestamp": str(ts),
            "x-wmrm-dispatch-token": str(payload.get("dispatchToken") or ""),
        }
        # No Cloudflare Access credential here. `/api/pod/*` is reached through an Access
        # **Bypass** policy, and the HMAC above is what authenticates the report -- one
        # mechanism for one hop rather than two that can disagree. Bypass does not log
        # the request, so the receiving handler logs it instead.

        url = f"{self.base_url}/api/pod/hooks"
        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    res = await client.post(url, content=body, headers=headers)
                if res.status_code < 300:
                    return True
                # 4xx that is not 429 will not become true by repeating it: a rejected
                # signature or a stale dispatch token is a decision, not a hiccup.
                if 400 <= res.status_code < 500 and res.status_code != 429:
                    return False
            except httpx.HTTPError:
                pass
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(delay)
                delay *= 2
        return False

    async def heartbeat(self, *, job_id: str, dispatch_token: str, state: str,
                        phase: str, progress: dict[str, Any] | None) -> bool:
        """Liveness, sent for **every** engine.

        Not folded into progress reporting, because only ProPainter has countable
        progress (its parts directory) and the others would then look dead. `unblend`
        is not the quick case that assumption would need: measured at 5.3 fps on a
        480x640 fixture, not the 34 fps the README quotes, so at 4K it is a long job
        that reports no progress at all.
        """
        return await self.send({
            "jobId": job_id,
            "dispatchToken": dispatch_token,
            "kind": "heartbeat",
            "state": state,
            "phase": phase,
            "progress": progress,
        })

    async def terminal(self, *, job_id: str, dispatch_token: str, state: str,
                       outcome: str, report: dict[str, Any] | None,
                       error: dict[str, Any] | None,
                       output_key: str | None = None) -> bool:
        return await self.send({
            "jobId": job_id,
            "dispatchToken": dispatch_token,
            "kind": "terminal",
            "state": state,
            "outcome": outcome,
            # Where the result landed. Sent by the pod because the pod is what put it
            # there -- with the upload on this side, the control plane never handles the
            # object, only the key.
            "outputKey": output_key,
            "report": report,
            "error": error,
        })
