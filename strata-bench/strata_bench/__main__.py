"""strata-bench command line.

  python3 -m strata_bench run --base URL --config RUN-CONFIG.json --origin setup-default|custom-N
                              --label LABEL --out DIR [--sizes probe,medium,long,xlong[,gprobe]] [--identity IDENTITY.json]
  python3 -m strata_bench validate RESULT.json [...]

Exit codes of run: 0 done, 2 preflight refused (nothing measured), 3 incomplete (partial result kept).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import VERSION, core, validate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="strata_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure a running server")
    r.add_argument("--base", required=True)
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--origin", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--protocol", default=core.PROTOCOL, choices=[core.PROTOCOL])
    r.add_argument("--sizes", default="probe,medium,long,xlong")
    r.add_argument("--identity", type=Path)
    r.add_argument("--probe-offset", type=int, default=0)
    r.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"), help="default $STRATA_API_KEY; never written")
    v = sub.add_parser("validate", help="check result files for public export")
    v.add_argument("files", nargs="+", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "run":
        sizes = [s for s in a.sizes.split(",") if s]
        bad = [s for s in sizes if s not in ("probe", "gprobe") and s not in core.SIZES]
        if bad:
            p.error(f"unknown sizes: {bad}")
        cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
        ident = json.loads(a.identity.read_text(encoding="utf-8")) if a.identity else None
        return core.run(a.base.rstrip("/"), cfg, a.origin, sizes, a.label, a.out, ident,
                        probe_offset=a.probe_offset, api_key=a.api_key)
    rc = 0
    for f in a.files:
        probs = validate.problems(json.loads(f.read_text(encoding="utf-8")))
        print(f"{f}: {'OK' if not probs else 'REJECTED'}")
        for x in probs:
            print("  -", x)
        rc |= bool(probs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
