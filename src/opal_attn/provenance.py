"""Reading the connector's SafeTensors files, with or without vLLM installed.

`vllm_attn_connector.store` has no vLLM dependency of its own -- it falls back
to the stdlib logger -- but importing it runs the package `__init__`, which
imports `connector`, which needs vLLM. On a CPU-only machine that fails.

That matters because of how this experiment is split: capture happens on
Polaris, analysis happens on a laptop. Requiring a CUDA vLLM install just to
read a file back would mean every look at the results needs an allocation. So
when the package will not import, load `store.py` directly off disk.

The import is tried first and preferred, so on a machine that HAS vLLM this is
exactly `vllm_attn_connector.load_provenance` and not a reimplementation of it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_store_module():
    try:
        from vllm_attn_connector import load_provenance, open_lazy
        return load_provenance, open_lazy
    except Exception:
        pass

    for root in _candidate_roots():
        store = Path(root) / "vllm_attn_connector" / "store.py"
        if not store.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_opal_attn_store", store)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_opal_attn_store"] = mod
        spec.loader.exec_module(mod)        # no vLLM import inside store.py
        return mod.load, mod.open_lazy

    raise ImportError(
        "cannot find vllm_attn_connector's store.py. Either install the package, "
        "or point OPAL_CONNECTOR_SRC at its src/ directory."
    )


def _candidate_roots() -> list[str]:
    here = Path(__file__).resolve().parents[2]      # the experiment directory
    roots = [os.environ.get("OPAL_CONNECTOR_SRC", "")]
    roots += [p for p in sys.path if p]
    roots.append(str(here.parent / "vllm-attn-connector" / "src"))
    return [r for r in roots if r]


_store_load, _store_open_lazy = _load_store_module()


def _rebase(record: dict, run_dir) -> dict:
    """A copy of `record` whose uri points inside `run_dir`.

    The connector records an ABSOLUTE `file://` uri, resolved on the machine
    that wrote it, and `store.load` reads that path verbatim. The moment a run
    directory is rsynced anywhere -- pulled off Polaris, pushed to a colleague,
    even just moved on the same disk -- that path stops existing and every read
    fails with a confusing FileNotFoundError naming /lus/eagle.

    Files are written to `out_dir/<connector-workflow-id>/<request>_g<n>.safetensors`,
    so the last TWO path components are stable wherever the directory lands.
    Match on those first, then fall back to the bare filename, which covers a
    directory someone flattened by hand.

    The uri is rewritten only when it does not resolve, so a run being analysed
    in place is untouched.
    """
    desc = (record or {}).get("attention_stats") or {}
    uri = desc.get("uri")
    if not uri or run_dir is None:
        return record
    path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
    if path.is_file():
        return record

    run_dir = Path(run_dir)
    for cand in (run_dir / path.parent.name / path.name, run_dir / path.name):
        if cand.is_file():
            break
    else:
        hits = list(run_dir.rglob(path.name))
        if not hits:
            raise FileNotFoundError(
                f"{path.name} is referenced by the provenance but is not under "
                f"{run_dir}. The record's uri is {uri}, which does not exist "
                f"here either -- the tensor files were probably not copied "
                f"alongside the JSONL (see scripts/push_results.sh --no-tensors)."
            )
        cand = hits[0]

    patched = dict(record)
    patched["attention_stats"] = {**desc, "uri": cand.resolve().as_uri()}
    return patched


def load_provenance(record: dict, run_dir=None):
    """Read a record's arrays, resolving the file relative to `run_dir`.

    `run_dir` is what makes a moved run readable; omit it to behave exactly
    like `vllm_attn_connector.load_provenance`.
    """
    return _store_load(_rebase(record, run_dir))


def open_lazy(record: dict, run_dir=None):
    """`safe_open` handle for one tensor, with the same rebasing."""
    return _store_open_lazy(_rebase(record, run_dir))


__all__ = ["load_provenance", "open_lazy"]
