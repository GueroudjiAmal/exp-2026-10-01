"""OPAL chain experiment: attention provenance via vllm-attn-connector + Flowcept.

The experiment replays the OPAL benchmark's multi-round chains through one vLLM
engine, capturing per-decode-step attention over the prompt as Flowcept
provenance. Each chain is a conversation whose history is the model's OWN
answers, so attention at round N is paid over text the model itself produced at
rounds 1..N-1 -- which is the thing worth measuring.
"""

__all__ = ["chains", "ranges", "answers", "plan", "runner", "report"]
