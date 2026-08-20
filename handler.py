"""Stub RunPod Serverless handler.

This repo is not a queue-worker service -- it's a research pipeline meant to
run as a long-lived RunPod Pod over SSH (Epilepsy/run_pipelines.py's
leave-one-seizure-out training runs are hours long, request/response jobs
don't fit that shape). This file exists only because RunPod's GitHub-
integration build -- the mechanism used to get this repo's root Dockerfile
built into a hosted image for reuse on a Pod, see README.md's "RunPod pod
image" section -- requires a queue-based Serverless endpoint to declare a
handler at container startup, even when the endpoint itself is never meant
to serve real traffic. It just echoes its input back; nothing in this repo
calls it, and no real workload should ever be routed through it.
"""

import runpod


def handler(job):
    return {"echo": job.get("input")}


runpod.serverless.start({"handler": handler})
