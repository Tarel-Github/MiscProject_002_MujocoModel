"""화면 없이 쓰는 명령줄 도구."""
from __future__ import annotations

import argparse
import sys


def validate_main(argv: list[str] | None = None) -> int:
    from .core.compiler import compile_doc, validate_steps
    from .core.doc import MjcfDoc, MjcfError
    ap = argparse.ArgumentParser(description="MJCF 모델을 컴파일하고 잠깐 시뮬레이션해 검증합니다")
    ap.add_argument("file", help="MJCF 파일 경로")
    ap.add_argument("--steps", type=int, default=200, help="시뮬레이션 스텝 수 (기본 200)")
    a = ap.parse_args(argv)
    try:
        doc = MjcfDoc.load(a.file)
    except MjcfError as ex:
        print(f"오류: {ex}")
        return 2
    res = compile_doc(doc)
    if not res.ok:
        print("컴파일 오류:", res.pretty_error(doc))
        return 1
    m = res.model
    print(f"성공: 컴파일 {res.elapsed_ms:.1f} ms; nbody={m.nbody} njnt={m.njnt} nq={m.nq} ngeom={m.ngeom} "
          f"nu={m.nu} nsensor={m.nsensor}")
    issues = validate_steps(m, a.steps)
    for i in issues:
        print("경고:", i)
    return 0 if not issues else 3


if __name__ == "__main__":
    sys.exit(validate_main())
