"""Public-export gate. A result.json may be published only if: it is a complete strata-bench/v1 result of the public
protocol with its canonical prompt hashes, every key at every depth is in the public schema (schema.PUBLIC), and no
value looks like a local path, a private network address or a secret. It does not prove that a free-text value is
harmless: keep free text out."""
import json
import re

from . import schema

LEAKS = [
    (re.compile(r"\b[A-Za-z]:[\\/]"), "a Windows path"),
    (re.compile(r"(?<![<\w.:/])/[A-Za-z0-9_.-]+/"), "an absolute path"),
    (re.compile(r"\b(10|127)(\.\d{1,3}){3}\b|\b192\.168(\.\d{1,3}){2}\b|\b172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2}\b"),
     "a private network address"),
    (re.compile(r"\\\\[\w.-]+\\"), "a network share"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,})"), "a token"),
]


def problems(doc: dict) -> list:
    out = schema.check(doc, schema.PUBLIC)
    if doc.get("schema") != "strata-bench/v1":
        out.append("schema is not strata-bench/v1")
    proto = doc.get("protocol") or {}
    if proto.get("id") != "standard-v1":
        out.append(f"protocol {proto.get('id')!r} is not public")
    for name, h in (proto.get("prompt_sha256") or {}).items():
        if schema.STANDARD_V1_PROMPTS.get(name) != h:
            out.append(f"prompt {name} is not standard-v1's")
    from .core import PROBE_SYSTEM, sha256          # the canonical probe system prompt
    if proto.get("probe_system_sha256") not in (None, sha256(PROBE_SYSTEM.encode())):
        out.append("probe system prompt is not standard-v1's")
    from .core import GPROBE_CHARS, GPROBE_SYSTEM   # the canonical gprobe, when the result has one
    gp = proto.get("gprobe") or {}
    if gp and (gp.get("system_sha256") != sha256(GPROBE_SYSTEM.encode())
               or gp.get("chars") != {str(k): v for k, v in GPROBE_CHARS.items()}):
        out.append("gprobe is not standard-v1's")
    if (doc.get("validity") or {}).get("status") not in ("ok", "warnings"):
        out.append(f"result is not complete (status {(doc.get('validity') or {}).get('status')!r})")
    text = json.dumps(doc)
    for rx, what in LEAKS:
        m = rx.search(text)
        if m:
            out.append(f"contains {what}: {m.group(0)!r}")
    return out
