"""Measurement core: preflight, calls, per-call persistence, result.json (schema strata-bench/v1).

Attach mode only: the caller starts and stops the server. Standard library only.
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from . import SCHEMA, VERSION
from .schema import IDENTITY, OBSERVED, PUBLIC, RUN, SERVER_FIELDS, SHARED_KEYS, STANDARD_V1_PROMPTS, prune

PROMPTS = Path(__file__).parent / "prompts"
SIZES = ("short", "medium", "long", "xlong")
PROTOCOL = "standard-v1"
REUSE_LIMIT = 64          # more reused prompt tokens than this on a size run means the prompt was not read fresh
SECRET_ENV = re.compile(r"(?i)(api_?key|secret|passw(or)?d|token)$")

PROBE_SYSTEM = ("You are a code reviewer for a C++ inference engine. Classify each line the user sends. Answer with "
                "exactly one word: SAFE if the line has no obvious correctness risk, RISKY if it does. Rules: " +
                " ".join(f"Rule {i}: " + r for i, r in enumerate([
                    "atomics with relaxed ordering that publish data to another thread are RISKY",
                    "unchecked integer narrowing from 64-bit to 32-bit sizes is RISKY",
                    "pointer arithmetic past the end of a buffer is RISKY",
                    "a lock acquired without a matching release on every path is RISKY",
                    "plain arithmetic, logging, comments and declarations are SAFE",
                    "reads of constants or configuration values are SAFE",
                    "a busy-wait loop without a pause or yield instruction is RISKY",
                    "a division whose divisor can be zero is RISKY",
                    "a function call whose error result is ignored is RISKY",
                    "standard containers used within their size are SAFE"], 1)) +
                " Think about memory ordering, bounds, error handling and thread safety, but never explain: one word "
                "only. " * 3)
PROBE_CALLS, PROBE_SKIP = 30, 2
# gprobe (opt-in, --sizes ...,gprobe): short-request probes shaped like an agent's classifier calls - a short fixed
# system prompt and a ~150/300/450-token excerpt of the public xlong.txt, a different excerpt per call, 1 token out.
# It measures the prompt path below Strata's 1024-token streamed walk, which the line probes barely touch.
GPROBE_SYSTEM = ("You review one excerpt of C++ from an inference engine. Answer with exactly one word: SAFE if the "
                 "excerpt has no obvious correctness risk, RISKY if it does. Risky means a data race, an out-of-bounds "
                 "access, an ignored error or an integer overflow. One word only, no explanation.")
GPROBE_CHARS = {150: 525, 300: 1050, 450: 1575}   # ~3.5 characters per token for this code


class PreflightError(Exception):
    pass


class MeasureError(Exception):
    pass


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_path(v: str) -> bool:
    return "\\" in v or "/" in v or v.lower().endswith((".gguf", ".bin", ".exe", ".json"))


def secret_env(k: str) -> bool:
    """Credentials in env: STRATA_API_KEY, and outside the engine's STRATA_ prefix any *_TOKEN / *_API_KEY /
    *_SECRET / *_PASSWORD (HF_TOKEN, OPENAI_API_KEY). Engine knobs like STRATA_INDEXER_PER_TOKEN stay."""
    return k.upper() == "STRATA_API_KEY" or (not k.upper().startswith("STRATA_") and bool(SECRET_ENV.search(k)))


def depath(v):
    if isinstance(v, dict):
        return {k: depath(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [depath(x) for x in v]
    return "<path>" if isinstance(v, str) and is_path(v) else v


def scrub_config(cfg: dict) -> dict:
    """args, env and the server fields a start uses, with every path replaced by the role of the flag that carries
    it (<pack>, <native>, ...) and every secret-looking value redacted. host, port, log, exe and api_key stay out."""
    args, out = [str(x) for x in cfg.get("args", [])], []
    for i, a in enumerate(args):
        prev = args[i - 1] if i else ""
        out.append(f"<{prev.lstrip('-')}>" if is_path(a) and prev.startswith("--") else "<path>" if is_path(a) else a)
    env = {k: "<redacted>" if secret_env(k) else depath(str(v)) for k, v in sorted((cfg.get("env") or {}).items())}
    server = prune(depath({k: cfg[k] for k in SERVER_FIELDS if k in cfg}), PUBLIC["config"]["server"])
    return {"args": out, "env": env, "server": server}


def config_block(cfg: dict, origin: str, verified: bool, shared=None) -> dict:
    s = scrub_config(cfg)
    if shared is not None:
        s["server_shared"] = shared
    return {**s, "config_id": sha256(json.dumps(s, sort_keys=True).encode())[:16], "config_origin": origin,
            "config_origin_verified": verified}


def flag(cfg: dict, name: str):
    a = cfg.get("args", [])
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else None


API_KEY = {"value": None}     # set by run(); sent as a bearer token, never written


def http_json(url: str, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"}
    if API_KEY["value"]:
        headers["authorization"] = "Bearer " + API_KEY["value"]
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def basename(p) -> str:
    return Path(str(p or "").replace("\\", "/")).name


def preflight(base: str, cfg: dict) -> tuple:
    """The server must be the config's: same model file and same context, else nothing is measured."""
    try:
        props = http_json(base + "/props", timeout=30)
    except ValueError as e:
        raise PreflightError(f"/props is not JSON: {e}")
    warnings = []
    if not isinstance(props.get("build_info"), str) or not props.get("build_info"):
        raise PreflightError("/props has no build_info: the engine version is unknown")
    if not props.get("model_path"):
        raise PreflightError("/props has no model_path: the model is unknown")
    served = basename(props.get("model_path"))
    want = flag(cfg, "--native")
    if want and served != basename(want):
        raise PreflightError(f"server model {served!r} is not the config's {basename(want)!r}")
    if not want:
        warnings.append("model check skipped: no --native in the config")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    want_ctx = flag(cfg, "--max-context")
    if want_ctx:
        try:
            ok = int(n_ctx) == int(want_ctx) and int(n_ctx) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise PreflightError(f"server context {n_ctx!r} is not the config's {want_ctx}")
    try:                                        # a server with an api_key refuses /v1 without it
        http_json(base + "/v1/models", timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PreflightError("the server wants an API key (--api-key or STRATA_API_KEY)")
    shared = None
    try:                                        # chat settings the server applies to requests that omit them
        s = http_json(base + "/settings", timeout=30)
        shared = prune({k: v for k, v in (s.get("defaults") or {}).items() if k in SHARED_KEYS},
                       PUBLIC["config"]["server_shared"])
        for k in ("reasoning_effort", "experimental_speed_projection"):
            if shared.get(k) not in (None, False, ""):
                warnings.append(f"server shared setting {k}={shared[k]!r} applies to the size runs")
    except (urllib.error.HTTPError, ValueError):
        warnings.append("server shared settings unknown (no /settings)")
    return {"build_info": props.get("build_info"), "model_file": served, "n_ctx": n_ctx, "shared": shared}, warnings


def check_prompts() -> dict:
    """Every prompt file must match SHA256SUMS byte for byte (a CRLF checkout or an edit is a different protocol)."""
    sums = dict(STANDARD_V1_PROMPTS)           # pinned in the code: an edited SHA256SUMS does not change the protocol
    for name, h in sums.items():
        got = sha256((PROMPTS / name).read_bytes())
        if got != h:
            raise PreflightError(f"prompt {name} does not match SHA256SUMS ({got[:12]} != {h[:12]}): "
                                 "not standard-v1 (a CRLF checkout?)")
    return sums


def call(base: str, messages: list, max_tokens: int, extra=None) -> dict:
    body = {"messages": messages, "temperature": 0, "seed": 42, "max_tokens": max_tokens, **(extra or {})}
    t0 = time.perf_counter()
    d = http_json(base + "/v1/chat/completions", body)
    wall = time.perf_counter() - t0
    ch = d["choices"][0]
    msg, finish = ch.get("message", {}), ch.get("finish_reason")
    t = d.get("timings") or {}
    return {"request_sha256": sha256(json.dumps(messages, ensure_ascii=False).encode()),
            "wall_ms": round(wall * 1000, 1), "timings": prune(t, RUN["timings"]),
            "usage": prune(d.get("usage"), RUN["usage"]),
            "tokens": {"prompt": t.get("prompt_n"), "reused": t.get("cache_n"), "generated": t.get("predicted_n")},
            "reasoning_chars": len(msg.get("reasoning_content") or ""), "content_chars": len(msg.get("content") or ""),
            "finish_reason": finish,
            "output_sha256": sha256(json.dumps([msg.get("reasoning_content") or "", msg.get("content") or "",
                                                finish]).encode())}


def require(r: dict, keys: tuple, what: str):
    missing = [k for k in keys if not isinstance(r["timings"].get(k), (int, float))]
    if missing:
        raise MeasureError(f"{what}: the server returned no {', '.join(missing)}")


def summary(runs: list, key: str):
    v = [r["timings"][key] for r in runs if isinstance(r["timings"].get(key), (int, float))]
    return {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None


def core_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def merge_identity(block: str, observed: dict, ident: dict, warnings: list) -> dict:
    """Identity adds declared facts (hashes, hardware); it never overrides what the server reported, and keys outside
    the public schema are dropped."""
    out = dict(observed)
    spec = IDENTITY.get(block, {})
    for k, v in (ident.get(block) or {}).items():
        if k not in spec:
            warnings.append(f"identity {block}.{k} dropped: not in the public schema")
            continue
        clean = prune(v, spec[k])
        if clean != v:
            warnings.append(f"identity {block}.{k}: parts outside the public schema dropped")
        v = clean
        if k in OBSERVED.get(block, set()):
            if v != observed.get(k):
                warnings.append(f"identity {block}.{k}={v!r} differs from the server's {observed.get(k)!r}: kept the "
                                "server's")
        else:
            out[k] = v
    return out


class Result:
    """The result.json, rewritten after every call (atomic replace), so a failure keeps what was measured."""

    def __init__(self, path: Path, doc: dict):
        self.path, self.doc = path, doc

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def run(base: str, cfg: dict, origin: str, sizes: list, label: str, out_dir: Path, identity=None,
        runs_n=3, warmup=1, max_tokens=256, probe_offset=0, api_key=None, log=print) -> int:
    if not re.fullmatch(r"setup-default|custom-\d+", origin):
        raise SystemExit("--origin must be setup-default or custom-N")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.result.json"
    if path.exists():
        log(f"refusing to overwrite {path}")
        return 2
    API_KEY["value"] = api_key
    try:
        sums = check_prompts()
        server, warnings = preflight(base, cfg)
    except (PreflightError, OSError) as e:
        log(f"preflight refused: {e}")
        return 2
    ident = identity or {}
    verified = bool((ident.get("config") or {}).get("origin_verified")) and origin == "setup-default"
    res = Result(path, {
        "schema": SCHEMA,
        # flat copies for dashboards that read the legacy run_bench.py layout (no paths: model_path is the file name)
        "label": label, "build_info": server["build_info"], "model_path": server["model_file"],
        "n_ctx": server["n_ctx"],
        "runner": {"name": "strata-bench", "version": VERSION, "commit": core_commit()},
        "submission": {"label": label, "date": datetime.now().astimezone().isoformat(timespec="seconds")},
        "machine": merge_identity("machine", {}, ident, warnings),
        "engine": merge_identity("engine", {"build_info": server["build_info"]}, ident, warnings),
        "model": merge_identity("model", {"file": server["model_file"], "n_ctx": server["n_ctx"]}, ident, warnings),
        "config": {**config_block(cfg, origin, verified, server["shared"]),
                   **({"calibration": merge_identity("calibration", {}, ident, warnings)}
                      if ident.get("calibration") else {})},
        "protocol": {"id": PROTOCOL, "prompt_sha256": sums, "probe_system_sha256": sha256(PROBE_SYSTEM.encode()),
                     "sampling": {"temperature": 0, "seed": 42, "cache_prompt": "false on sizes"},
                     "max_tokens": max_tokens, "warmup": warmup, "runs": runs_n,
                     "probe": {"calls": PROBE_CALLS, "skip_first": PROBE_SKIP, "max_tokens": 1, "offset": probe_offset,
                               "cache_prompt": "server default (the shared system prefix may be reused)"},
                     "reasoning_sent": {"sizes": "server default", "probe": "enable_thinking false"},
                     "unique_line": "[strata-bench <size> <warmup|run> <i>]"},
        "runs": [], "sizes": {},
        "validity": {"status": "running", "warnings": warnings},
    })
    d = res.doc
    res.save()
    try:
        for size in sizes:
            if size == "probe":
                measure_probe(base, res, probe_offset, log)
                continue
            if size == "gprobe":
                d["protocol"]["gprobe"] = {"system_sha256": sha256(GPROBE_SYSTEM.encode()), "chars": GPROBE_CHARS,
                                           "source": "xlong.txt", "calls": PROBE_CALLS, "skip_first": PROBE_SKIP,
                                           "max_tokens": 1}
                measure_gprobe(base, res, log)
                continue
            text = (PROMPTS / f"{size}.txt").read_text(encoding="utf-8")
            done = []
            for kind, n in (("warmup", warmup), ("run", runs_n)):
                for i in range(1, n + 1):
                    msg = [{"role": "user", "content": f"[strata-bench {size} {kind} {i}]\n" + text}]
                    r = {"size": size, "kind": kind, "index": i,
                         **call(base, msg, max_tokens, {"cache_prompt": False})}
                    d["runs"].append(r)
                    res.save()
                    if kind != "run":
                        continue
                    require(r, ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_per_second"),
                            f"{size} run {i}")
                    done.append(r)
                    reused = r["tokens"]["reused"]
                    if reused is None:
                        d["validity"]["warnings"].append(f"{size} run {i}: reused tokens unknown (no cache_n)")
                    elif reused > REUSE_LIMIT:
                        d["validity"]["warnings"].append(f"{size} run {i}: {reused} prompt tokens reused, the prompt "
                                                         "was not read fresh")
                    t = r["timings"]
                    log(f"{label} {size} run {i}: prompt {t.get('prompt_n')} (+{t.get('cache_n')} reused) "
                        f"{t.get('prompt_per_second')} tok/s, decode {t.get('predicted_n')} tok "
                        f"{t.get('predicted_per_second')} tok/s")
            d["sizes"][size] = {"prompt_tps": summary(done, "prompt_per_second"),
                                "decode_tps": summary(done, "predicted_per_second"),
                                "prompt_ms": summary(done, "prompt_ms")}
            res.save()
    except Exception as e:                      # keep what was measured, say why it stopped
        d["validity"].update(status="incomplete", error=f"{type(e).__name__}: {e}"[:300])
        res.save()
        log(f"incomplete: {e}")
        return 3
    d["validity"]["status"] = "ok" if not d["validity"]["warnings"] else "warnings"
    res.save()
    log(f"written {path}")
    return 0


def measure_probe(base: str, res: Result, offset: int, log):
    """Classification probes: one fixed system prompt, one public code line per call, 1-token answer; the first
    PROBE_SKIP calls warm the shared system prefix and stay out of the summary."""
    d = res.doc
    lines = [l.strip() for l in (PROMPTS / "probe_lines.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for i in range(PROBE_CALLS):
        msg = [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": lines[(offset + i) % len(lines)]}]
        r = {"size": "probe", "kind": "warmup" if i < PROBE_SKIP else "run", "index": i + 1,
             **call(base, msg, 1, {"chat_template_kwargs": {"enable_thinking": False}})}
        d["runs"].append(r)
        res.save()
        if i >= PROBE_SKIP:
            require(r, ("prompt_n", "prompt_ms"), f"probe {i + 1}")
            p, c = r["tokens"]["prompt"], r["tokens"]["reused"]
            if p is not None and p <= 4:          # prompt_n is what was read: ~nothing read = the whole call reused
                d["validity"]["warnings"].append(f"probe {i + 1}: only {p} prompt tokens read ({c} reused), the "
                                                 "whole call came from the cache")
        rows.append(r)
    kept = rows[PROBE_SKIP:]
    reused = [r["tokens"]["reused"] for r in kept if r["tokens"]["reused"] is not None]
    d["sizes"]["probe"] = {"prompt_ms": summary(kept, "prompt_ms"),
                           "wall_ms": {"median": statistics.median(r["wall_ms"] for r in kept)},
                           "reused_median": statistics.median(reused) if reused else None}
    s = d["sizes"]["probe"]
    log(f"probe: prompt_ms median {s['prompt_ms']['median']}, reused median {s['reused_median']}")


def gprobe_excerpt(code: str, size: int, i: int) -> str:
    start = 2000 + (size * 37 + i * 1777) % (len(code) - 4000)
    return code[start:start + GPROBE_CHARS[size]]


def measure_gprobe(base: str, res: Result, log):
    """Agent-shaped probes at ~150/300/450 tokens (see GPROBE_SYSTEM); per size the first PROBE_SKIP calls warm the
    shared system prefix and stay out of the summary."""
    d = res.doc
    code = (PROMPTS / "xlong.txt").read_text(encoding="utf-8")
    for size in GPROBE_CHARS:
        rows = []
        for i in range(PROBE_CALLS):
            msg = [{"role": "system", "content": GPROBE_SYSTEM}, {"role": "user", "content": gprobe_excerpt(code, size, i)}]
            r = {"size": f"gprobe{size}", "kind": "warmup" if i < PROBE_SKIP else "run", "index": i + 1,
                 **call(base, msg, 1, {"chat_template_kwargs": {"enable_thinking": False}})}
            d["runs"].append(r)
            res.save()
            if i >= PROBE_SKIP:
                require(r, ("prompt_n", "prompt_ms"), f"gprobe{size} {i + 1}")
            rows.append(r)
        kept = rows[PROBE_SKIP:]
        reused = [r["tokens"]["reused"] for r in kept if r["tokens"]["reused"] is not None]
        read = [r["tokens"]["prompt"] for r in kept if r["tokens"]["prompt"] is not None]
        d["sizes"][f"gprobe{size}"] = {"prompt_ms": summary(kept, "prompt_ms"),
                                       "wall_ms": {"median": statistics.median(r["wall_ms"] for r in kept)},
                                       "reused_median": statistics.median(reused) if reused else None,
                                       "read_median": statistics.median(read) if read else None}
        res.save()
        g = d["sizes"][f"gprobe{size}"]
        log(f"gprobe{size}: prompt_ms median {g['prompt_ms']['median']}, read {g['read_median']}, "
            f"reused {g['reused_median']}")
