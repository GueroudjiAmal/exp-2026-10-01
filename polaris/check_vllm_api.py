#!/usr/bin/env python
"""Is the installed vLLM new enough for vllm-attn-connector?

    python polaris/check_vllm_api.py             # static: safe on a login node
    python polaris/check_vllm_api.py --import    # also import it (needs a GPU node)
    python polaris/check_vllm_api.py --quiet     # exit code only

Exit 0 = usable, 1 = not.

By default this answers WITHOUT importing vllm. That is deliberate: Polaris
login nodes have no GPU, and `import vllm` there drags in torch and platform
detection for reasons unrelated to the question -- so an import failure would be
indistinguishable from "your version is too old" at exactly the point in setup
where the difference matters. `importlib.util.find_spec` locates the package
without running its __init__, and the sources are then read as text. Version
comes from package metadata, also without importing.

Use --import inside a job, where importing is itself a meaningful test.

TWO extension points are checked, because this connector uses two and they
appeared at different times:

1. The v1 KV connector base. The connector imports KVConnectorWorkerMetadata
   and SupportsHMA from it, and reports completion through
   build_connector_worker_meta.
       <= 0.25  none of it -- import fails outright
       0.26     all but requires_kv_delivery
       >= 0.27  everything -- the floor

2. The ATTENTION BACKEND REGISTRY. The probe subclasses the active backend and
   re-registers it via `AttentionBackendEnum` + `register_backend`. Nothing in
   the KV connector API implies that registry exists, so a vLLM that passes
   check 1 can still fail here -- and the failure mode is quiet rather than
   loud: `install_probe()` returns False, the connector still emits
   structurally valid records, and every number in them is zero.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# (relative source path, symbols that must appear in it, what it is for)
REQUIREMENTS = [
    (Path("distributed/kv_transfer/kv_connector/v1/base.py"),
     ("KVConnectorWorkerMetadata", "SupportsHMA",
      "build_connector_worker_meta", "request_finished_all_groups"),
     "KV connector v1 API"),
    (Path("v1/attention/backends/registry.py"),
     ("AttentionBackendEnum", "register_backend"),
     "attention backend registry (the query probe)"),
]

TOO_OLD = (
    "vllm-attn-connector needs >= 0.27.0 (tested on 0.27.x, 0.28.0 is the\n"
    "default pin here). The ALCF conda module ships far below that. Install a\n"
    "newer one into the venv:\n"
    "    pip install --ignore-installed vllm==0.28.0"
)


def locate() -> tuple[Path | None, str]:
    """Find the installed vllm package directory without importing it."""
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception as e:          # a broken install can raise here
        return None, f"cannot locate vllm: {type(e).__name__}: {e}"
    if spec is None or not spec.origin:
        return None, "vllm is not installed"
    try:
        version = importlib.metadata.version("vllm")
    except Exception:               # noqa: BLE001
        version = "unknown"
    return Path(spec.origin).parent, version


def check(do_import: bool = False) -> tuple[bool, str]:
    pkg, version = locate()
    if pkg is None:
        return False, version       # version holds the error message here

    for rel, symbols, what in REQUIREMENTS:
        src_file = pkg / rel
        if not src_file.is_file():
            return False, (f"vllm {version}: no {rel} in {pkg}, so the "
                           f"{what} is absent entirely.\n" + TOO_OLD)
        src = src_file.read_text(encoding="utf-8", errors="replace")
        missing = [s for s in symbols if s not in src]
        if missing:
            return False, (f"vllm {version}: the {what} is missing {missing}.\n"
                           + TOO_OLD)

    if do_import:
        # Only meaningful where vLLM can actually load: a GPU node.
        try:
            import vllm  # noqa: F401
            from vllm.distributed.kv_transfer.kv_connector.v1 import base  # noqa: F401
            from vllm.v1.attention.backends.registry import (  # noqa: F401
                AttentionBackendEnum,
                register_backend,
            )
        except Exception as e:      # noqa: BLE001
            return False, (
                f"vllm {version} has the right API but does not import here: "
                f"{type(e).__name__}: {e}\n"
                "On a login node that is expected (no GPU) -- drop --import.\n"
                "In a job it is real: most likely the CUDA wheels vs the driver."
            )
        # FLASH_ATTN is what the probe overrides by default. If it will not
        # resolve, install_probe() returns False and the capture silently
        # records zeros -- so ask now rather than discover it in the output.
        try:
            AttentionBackendEnum["FLASH_ATTN"].get_class()
        except Exception as e:      # noqa: BLE001
            return False, (f"vllm {version}: FLASH_ATTN does not resolve on this "
                           f"node ({type(e).__name__}: {e}).\nThe probe would not "
                           f"install and the capture would be all zeros.")
        return True, (f"vllm {version}: connector API and backend registry "
                      f"complete, imports cleanly, FLASH_ATTN resolves")

    return True, f"vllm {version}: connector API and backend registry complete (static)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="also import vllm; only valid on a GPU node")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ok, message = check(args.do_import)
    if not args.quiet:
        print(("OK: " if ok else "FAIL: ") + message,
              file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
