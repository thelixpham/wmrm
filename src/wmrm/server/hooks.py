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

The same terminal event is also announced to a **Mezon channel webhook**, when one is
configured -- see `MezonNotifier`. That is a second destination for one event, not a
second source of truth: it carries no signature and nothing reads it back.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
import time
import uuid
from typing import Any

import httpx

from .. import __version__

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


#: Shorter than the control plane's. A chat notification that arrives four minutes late has
#: already been overtaken by whoever went and looked, so there is nothing to buy by waiting.
MEZON_TIMEOUT = 8.0
MEZON_ATTEMPTS = 2

#: Outcome -> the glyph that leads the line. Only the ones worth telling apart at a glance:
#: everything else is a failure and gets the same mark, because a reader who needs the
#: difference between `oom` and `verify_failed` is going to read the word anyway.
MEZON_MARK = {
    "ok": "✅",
    "canceled": "🚫",
    "interrupted": "⚠️",
    "upload_failed": "⚠️",
}


class MezonNotifier:
    """Announces terminal events to a Mezon channel webhook.

    A **channel** webhook, `https://webhook.mezon.ai/webhooks/{channelId}/{token}` -- the
    URL is the whole credential, so it is read from the environment and never logged.

    This is deliberately not the reporting path. It carries no signature, no dispatch
    token and no event id, so nothing can be deduped or replayed by the receiver, and the
    control plane must not learn anything here it did not learn from `Notifier`. It exists
    so a person finds out a nine-hour job finished without polling for it.

    Two attempts rather than four, and every failure is swallowed: the audience is a chat
    channel. If it does not arrive, the job is still finished and correct, and the line
    printed here is the only consequence.
    """

    def __init__(self, url: str | None, *, pod_id: str | None = None):
        self.url = (url or "").strip() or None
        self.pod_id = pod_id

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def describe(self) -> str:
        """What to print at startup. The channel id, never the token.

        The two are adjacent path segments of one URL, so printing "the webhook" in full
        -- the obvious thing to do -- publishes the credential to a console that RunPod
        shows in its dashboard.
        """
        if not self.url:
            return "(not configured)"
        parts = [p for p in self.url.split("?")[0].rstrip("/").split("/") if p]
        return f"channel {parts[-2]}" if len(parts) >= 2 else "configured"

    async def post(self, text: str) -> bool:
        """POST one message. Returns whether it landed; never raises."""
        if not self.url:
            return False

        # Mezon's envelope: `type` is the literal "hook", and the human-readable text is
        # `message.t`. `mk` (markdown spans) is left off on purpose -- its offsets are
        # character indexes into `t`, so any edit to the wording below would silently
        # move the formatting onto the wrong substring.
        body = {"type": "hook", "message": {"t": text}}

        for attempt in range(1, MEZON_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=MEZON_TIMEOUT) as client:
                    res = await client.post(
                        self.url, json=body,
                        headers={"user-agent": f"wmrm-pod/{__version__}"})
                if res.status_code < 300:
                    return True
                # A revoked or mistyped webhook URL answers 404 for good. Retrying it
                # only delays the job's cleanup by the backoff.
                if 400 <= res.status_code < 500 and res.status_code != 429:
                    return False
            except httpx.HTTPError:
                pass
            if attempt < MEZON_ATTEMPTS:
                await asyncio.sleep(2.0)
        return False

    def compose(self, *, job_id: str, state: str, outcome: str,
                output_key: str | None, error: dict[str, Any] | None) -> str:
        """The message body. One glyph, one summary line, then only what is known."""
        mark = MEZON_MARK.get(outcome, "❌")
        lines = [f"{mark} wmrm {state} — {outcome}"]
        if self.pod_id:
            lines.append(f"pod: {self.pod_id}")
        lines.append(f"job: {job_id}")
        if output_key:
            lines.append(f"output: {output_key}")
        message = (error or {}).get("message")
        if message:
            # Truncated because this is a chat message and a stack trace pasted into one
            # is how a channel gets muted. The whole thing is in the job record.
            lines.append(f"error: {str(message)[:300]}")
        return "\n".join(lines)

    async def terminal(self, *, job_id: str, state: str, outcome: str,
                       output_key: str | None = None,
                       error: dict[str, Any] | None = None) -> bool:
        return await self.post(self.compose(
            job_id=job_id, state=state, outcome=outcome,
            output_key=output_key, error=error))


class Notifier:
    """Posts events for one pod. Fire-and-forget by design."""

    def __init__(self, *, base_url: str | None, pod_token: str | None,
                 mezon: MezonNotifier | None = None):
        self.base_url = (base_url or "").rstrip("/") or None
        # Derived once. The token itself is not kept, so nothing here can accidentally send
        # it: this object's job is to sign, not to authenticate outward.
        self.key = webhook_key(pod_token) if pod_token else None
        # Independent of `enabled` below: a pod with no `callbackBaseUrl` still has a person
        # who would like to know the job ended, and the two destinations fail separately.
        self.mezon = mezon or MezonNotifier(None)

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
            # Named rather than left as httpx's default, because this request crosses a
            # network the pod does not control. Measured: with the app behind Cloudflare, a
            # report arriving as `python-httpx/x.y` is indistinguishable in the WAF log from
            # any other script, so the rule that blocks it cannot be narrowed to spare this
            # one client. A stable name is something an allow rule can be written against.
            #
            # It is not a way past anything. A bot rule that refuses non-browser clients
            # refuses this too, and it should -- what authenticates a report is the signature
            # below, and the fix for a blocked callback is a skip rule on `/api/pod/*`, not a
            # cleverer string here.
            "user-agent": f"wmrm-pod/{__version__}",
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
        sent = await self.send({
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

        # Announced after, and only for terminal events: heartbeats are every 30 seconds
        # for the life of the job, and a channel that receives those is a channel nobody
        # reads. The return value stays the control plane's -- that is the one the runner
        # warns about, and a chat message that failed to send is not a lost result.
        if self.mezon.enabled:
            try:
                ok = await self.mezon.terminal(
                    job_id=job_id, state=state, outcome=outcome,
                    output_key=output_key, error=error)
            except Exception as exc:                 # noqa: BLE001 -- see the class docstring
                ok, exc_note = False, f" ({type(exc).__name__}: {exc})"
            else:
                exc_note = ""
            if not ok:
                # Said out loud because nothing else will notice: the URL is the whole
                # credential, so the usual symptom is a silently revoked webhook.
                print(f"[job {job_id}] mezon: notification not delivered{exc_note}",
                      file=sys.stderr, flush=True)

        return sent
