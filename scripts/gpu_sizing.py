#!/usr/bin/env python
"""What settings fit on THIS NVIDIA card? Reads the dry run, answers in flags.

    python scripts/gpu_sizing.py                      # detect the GPU
    python scripts/gpu_sizing.py --vram-gb 24         # ask about a hypothetical

The binding constraint for this workload is the KV cache, not the weights.
Attention capture does not change that: the connector's probe buffers are
`max_num_seqs x heads x head_size` per layer (tens of MB) and its scratch is one
`[q_heads, prompt_len]` float32 row per request (under a MB). Both are noise
next to KV. What capture DOES cost is time, not memory.

KV bytes per token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes

For Qwen3-4B-Instruct-2507 that is 2 x 36 x 8 x 128 x 2 = 147456 B = 144 KiB,
which is high for a 4B model -- 36 layers with 8 KV heads each. A chain's last
round reaches ~5.1k tokens, so one long sequence alone holds ~720 MiB of KV.

Numbers come from `results/plan/plan.json`, so run `make plan` first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# (layers, kv_heads, head_dim, params_billions) per model we have a shape for.
SHAPES = {
    # (layers, kv_heads, head_dim, params_billions) -- from each model's
    # config.json. KV per token is 2*layers*kv_heads*head_dim*2 bytes, which is
    # what actually bounds concurrency; parameter count only bounds whether the
    # weights fit at all.
    "Qwen/Qwen3-4B-Instruct-2507": (36, 8, 128, 4.02),
    "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit": (36, 8, 128, 4.02),
    "Qwen/Qwen3-14B": (40, 8, 128, 14.8),
    "Qwen/Qwen3-32B": (64, 8, 128, 32.8),
    # MoE: 30B total, ~3B active per token. Weights are sized by the total
    # (they all have to be resident), but compute and therefore speed follow
    # the active count -- which is why this is the interesting one on a
    # bandwidth-limited part like GB10.
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (48, 4, 128, 30.5),
    "Qwen/Qwen2.5-72B-Instruct": (80, 8, 128, 72.7),
    # Qwen3.5. All hybrid (full attention every `full_attention_interval`
    # layers, gated-delta-net in between) and all multimodal, so their shape
    # lives under config.json's `text_config`. head_dim is 256, which is why
    # KV per token stays low despite the layer counts -- 2 or 4 KV heads.
    #
    # The layer count here is the TOTAL; the connector scores only the
    # full-attention subset (10 of 40 on 35B-A3B). That does not change KV
    # sizing, which the engine does over all attention layers.
    "Qwen/Qwen3.5-4B": (32, 4, 256, 5.0),
    "Qwen/Qwen3.5-9B": (32, 4, 256, 10.0),
    "Qwen/Qwen3.5-27B": (64, 4, 256, 28.0),
    "Qwen/Qwen3.5-35B-A3B": (40, 2, 256, 36.0),
    "Qwen/Qwen3.5-122B-A10B": (48, 2, 256, 125.0),
    "Qwen/Qwen2.5-7B-Instruct": (28, 4, 128, 7.62),
    "facebook/opt-125m": (12, 12, 64, 0.125),
}
BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "awq-4bit": 0.6}   # AWQ incl. scales


def detect_vram_gb() -> float | None:
    """Usable device memory, in GiB.

    torch first, nvidia-smi second. On a unified-memory part -- Grace Hopper,
    Grace Blackwell / DGX Spark -- `nvidia-smi --query-gpu=memory.total` returns
    `[N/A]`, because there is no discrete framebuffer to report: the CPU and GPU
    share one physical pool. torch still reports what the device can address, so
    it is both more accurate and more portable here.
    """
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gib = props.total_memory / 2**30
            cc = f"sm_{props.major}{props.minor}"
            extra = (f"  [+{torch.cuda.device_count() - 1} more GPU(s); sizing "
                     f"for one]" if torch.cuda.device_count() > 1 else "")
            print(f"detected: {props.name}  ({gib:.1f} GiB, {cc}){extra}")
            if gib > 90:
                print("  note: this looks like unified memory shared with the CPU.")
                print("        gpu_memory_utilization is a fraction of the WHOLE")
                print("        pool, so a high value starves the host. Start low")
                print("        (--util 0.4) and raise it once a run is stable.")
            return gib
    except Exception:
        pass

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return None
    rows = [r for r in out.strip().splitlines() if r.strip()]
    if not rows:
        return None
    mib, name = rows[0].split(",", 1)
    try:
        gib = int(mib.strip()) / 1024
    except ValueError:
        # "[N/A]" -- unified memory, and torch was not importable to ask
        # properly. Guessing here would silently size the run wrong.
        print(f"detected: {name.strip()}, but it reports memory.total="
              f"{mib.strip()!r}.\n"
              f"  That is normal on a unified-memory part. Run this inside the\n"
              f"  GPU venv so torch can be asked instead:\n"
              f"    .venv-gpu/bin/python scripts/gpu_sizing.py\n"
              f"  or state it yourself:  --vram-gb 110", file=sys.stderr)
        return None
    print(f"detected: {name.strip()}  ({gib:.1f} GiB)"
          + (f"  [+{len(rows) - 1} more GPU(s); sizing for one]" if len(rows) > 1 else ""))
    return gib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--vram-gb", type=float, default=None)
    ap.add_argument("--plan", type=Path, default=HERE / "results" / "plan" / "plan.json")
    ap.add_argument("--util", type=float, default=0.90,
                    help="gpu_memory_utilization vLLM will be given")
    ap.add_argument("--quant", choices=["bf16", "awq-4bit"], default=None,
                    help="default: awq-4bit if the model name says so, else bf16")
    args = ap.parse_args()

    if args.model not in SHAPES:
        print(f"no recorded shape for {args.model}. Add it to SHAPES "
              f"(layers, kv_heads, head_dim, params_B) from its config.json.",
              file=sys.stderr)
        return 2
    layers, kv_heads, head_dim, params_b = SHAPES[args.model]
    quant = args.quant or ("awq-4bit" if "awq" in args.model.lower() else "bf16")

    vram = args.vram_gb if args.vram_gb is not None else detect_vram_gb()
    if vram is None:
        print("no NVIDIA GPU detected and no --vram-gb given.", file=sys.stderr)
        return 2

    if not args.plan.is_file():
        print(f"no plan at {args.plan} -- run `make plan` first.", file=sys.stderr)
        return 2
    plan = json.loads(args.plan.read_text())
    proj = plan["projection"]
    p_max = proj["prompt_tokens_max"]
    p_med = proj["prompt_tokens_median"]
    gen = plan["config"]["max_tokens"]

    kv_per_token = 2 * layers * kv_heads * head_dim * 2        # bytes, bf16 cache
    weights_gib = params_b * 1e9 * BYTES_PER_PARAM[quant] / 2**30
    # vLLM's own activations, CUDA context, NCCL buffers and the probe's
    # persistent query buffers. Measured headroom, not a derived figure --
    # treat it as the reason not to run at utilization 0.98.
    overhead_gib = 1.2
    budget_gib = vram * args.util - weights_gib - overhead_gib

    print(f"\nmodel        {args.model}  ({quant})")
    print(f"  weights    {weights_gib:.2f} GiB")
    print(f"  KV/token   {kv_per_token / 1024:.0f} KiB   "
          f"({layers} layers x {kv_heads} kv heads x {head_dim})")
    print(f"workload     max {p_max} + {gen} tok, median {p_med} + {gen}")
    print(f"  one max-length sequence holds "
          f"{(p_max + gen) * kv_per_token / 2**20:.0f} MiB of KV")

    print(f"\nat gpu_memory_utilization={args.util} on {vram:.1f} GiB:")
    if budget_gib <= 0.25:
        print(f"  KV budget  {budget_gib:.2f} GiB  -- DOES NOT FIT")
        print(f"\n  {args.model} in {quant} needs about "
              f"{weights_gib + overhead_gib + 1:.1f} GiB before any KV cache.")
        if quant == "bf16":
            print("  Try the 4-bit build instead:")
            print("    --model cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
        return 1

    kv_tokens = int(budget_gib * 2**30 / kv_per_token)
    conc_max = max(1, kv_tokens // (p_max + gen))
    conc_med = max(1, kv_tokens // (p_med + gen))
    print(f"  KV budget  {budget_gib:.2f} GiB  = {kv_tokens:,} cached tokens")
    print(f"  concurrent {conc_max} sequences at the LONGEST round, "
          f"{conc_med} at the median")

    # max_num_seqs is a cap, not a reservation -- vLLM schedules within the
    # pool regardless. Setting it near the long-round figure keeps the engine
    # from admitting a wave it then has to preempt, and preemption is the one
    # path the connector's `restarts` counter exists to flag.
    seqs = max(1, min(64, conc_max))
    graphs = budget_gib > 2.5      # graph pools need room the KV cache wants
    print("\nsuggested flags:")
    print(f"  --model {args.model} \\")
    print(f"  --max-model-len {plan['config']['max_model_len']} \\")
    print(f"  --max-num-seqs {seqs} \\")
    print(f"  --gpu-memory-utilization {args.util}", end="")
    print(" \\\n  --cuda-graphs" if graphs else "", end="")
    print()
    if not graphs:
        print("\n  CUDA graphs left OFF: the capture pool plus graph pools do not")
        print("  fit here. Note the connector's e2e covers the graph path"
              " separately,\n  so correctness is not at stake -- only speed.")
    if conc_max < 4:
        print(f"\n  Only {conc_max} long sequences fit at once, so the later rounds")
        print("  of the batch will serialise. Expect the run to be dominated by")
        print("  rounds 5-7. `--limit` a subset if you only need a sample.")
        if quant == "bf16":
            print("  The 4-bit build frees ~4.9 GiB for KV and would raise this a lot:")
            print("    --model cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
