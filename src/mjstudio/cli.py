"""Headless command line tools."""
from __future__ import annotations

import argparse
import sys


def validate_main(argv: list[str] | None = None) -> int:
    from .core.compiler import compile_doc, validate_steps
    from .core.doc import MjcfDoc, MjcfError
    ap = argparse.ArgumentParser(description="Compile and briefly simulate an MJCF model")
    ap.add_argument("file")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args(argv)
    try:
        doc = MjcfDoc.load(a.file)
    except MjcfError as ex:
        print(f"ERROR: {ex}")
        return 2
    res = compile_doc(doc)
    if not res.ok:
        print("COMPILE ERROR:", res.pretty_error(doc))
        return 1
    m = res.model
    print(f"OK: compiled in {res.elapsed_ms:.1f} ms; nbody={m.nbody} njnt={m.njnt} nq={m.nq} ngeom={m.ngeom} "
          f"nu={m.nu} nsensor={m.nsensor}")
    issues = validate_steps(m, a.steps)
    for i in issues:
        print("WARNING:", i)
    return 0 if not issues else 3


if __name__ == "__main__":
    sys.exit(validate_main())
