"""HTTP wrapper around `wmrm run`, for a machine that holds a GPU.

The shape, and why it is this shape:

- **The pod is a server, the control plane is the client.** A RunPod Pod has a public
  proxy URL, so the scheduler can call in and does not need the pod to poll. That also
  means the pod never needs credentials for anything except talking back.
- **The pod holds no R2 credentials.** Input arrives as a presigned URL, output leaves
  through presigned part URLs minted by the control plane. An R2 API token is scoped to
  a bucket, not to a prefix, so putting one on four rented machines would be four copies
  of delete-the-bucket.
- **Work is never done inside a request.** `POST /jobs` validates, records, and returns
  202 in well under a second; the run itself is a background task that can last days.
- **Disk is the source of truth, not memory.** uvicorn is restarted by things outside
  our control -- a pod stop wipes the container filesystem, an operator redeploys. Job
  state lives in files so that a restart can tell the control plane what happened
  instead of losing the job silently.

Run it with one worker. State is per-process and on disk; a second worker would be a
second opinion about what this machine is doing.
"""
