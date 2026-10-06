"""The public result schema, recursive. The same spec prunes what the core writes (unknown keys are dropped) and
gates what may be exported (unknown keys are rejected).

Spec forms: SCALAR (str/int/float/bool/None), HEX (hex digest), a dict {key: spec} (only those keys), [spec] (a list
of spec), Map(key_regex, spec) (a dynamic mapping whose keys match key_regex).
"""
import re

SCALAR, HEX = "scalar", "hex"


class Map:
    def __init__(self, key_regex: str, spec):
        self.key, self.spec = re.compile(key_regex), spec


STANDARD_V1_PROMPTS = {
    "medium.txt": "e4c2b6ad18b235d3f0466f0aeb1256612c533188c75b2c5c6741ea6e5463b1c6",
    "long.txt": "2b54045914a7bd362c93587902dc16af4d589cea3493cd0301a116df8495844a",
    "xlong.txt": "acf504f5230c14fdb6e6ca5d8125a9dabd4176d8f5d09882b5a3aed4eee98b83",
    "short.txt": "2cd69728112a942401591b3d19141361105a8b2e9195ae19c007b593d70e9402",
    "probe_lines.txt": "dfa9547550e81cca80b085f57ceb4614fe0c4fba040ddf96fe8d70bb68dde2db",
}
SERVER_FIELDS = ("reasoning_budget_tokens", "sampling", "draft_vocab", "fit_max_tokens", "gpu", "layer_split")
SHARED_KEYS = ("reasoning_effort", "temperature", "top_p", "top_k", "seed", "max_tokens",
               "experimental_speed_projection")

SUMMARY = {"median": SCALAR, "min": SCALAR, "max": SCALAR}
GPU = {"name": SCALAR, "vram_gb": SCALAR, "arch": SCALAR, "pcie": SCALAR, "power_limit_w": SCALAR, "role": SCALAR,
       "display": SCALAR}
TIMINGS = {k: SCALAR for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
                               "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
                               "draft_n", "draft_n_accepted")}
USAGE = {"prompt_tokens": SCALAR, "completion_tokens": SCALAR, "total_tokens": SCALAR,
         "prompt_tokens_details": {"cached_tokens": SCALAR},
         "completion_tokens_details": {"reasoning_tokens": SCALAR}}
SAMPLING = {"temperature": SCALAR, "top_p": SCALAR, "top_k": SCALAR, "min_p": SCALAR, "seed": SCALAR,
            "max_tokens": SCALAR, "repeat_penalty": SCALAR, "presence_penalty": SCALAR}

# keys an --identity file may add (declared by the caller), per block
IDENTITY = {
    "machine": {"machine_id": SCALAR, "os": SCALAR, "wsl": SCALAR, "container": SCALAR, "gpus": [GPU], "cpu": SCALAR,
                "cpu_isa": SCALAR, "cpu_isa_used": SCALAR, "cores": SCALAR, "ram_gb": SCALAR, "ram_type": SCALAR,
                "ram_speed_mts": SCALAR, "driver": SCALAR, "cuda": SCALAR, "hip": SCALAR, "hardware_hash": HEX},
    "engine": {"version": SCALAR, "commit": SCALAR, "binary_sha256": HEX, "source": SCALAR, "build_options": [SCALAR],
               "backend": SCALAR, "server_py_version": SCALAR, "build_info": SCALAR},
    "model": {"family": SCALAR, "quant": SCALAR, "repo": SCALAR, "revision": SCALAR,
              "files_sha256": Map(r"^[\w.-]+$", HEX), "storage": SCALAR, "mtp": SCALAR, "draft_vocab": SCALAR,
              "expert_profile": SCALAR, "hashes_verified_on_server": SCALAR, "file": SCALAR, "n_ctx": SCALAR},
    "calibration": {"ran": SCALAR, "kept_params": Map(r"^--?[\w-]+$", SCALAR), "pool_workers_asked": SCALAR,
                    "pool_workers_resolved": SCALAR, "sweep_summary": SCALAR, "state": SCALAR},
    "config": {"origin_verified": SCALAR},
}
OBSERVED = {"engine": {"build_info"}, "model": {"file", "n_ctx"}}

RUN = {"size": SCALAR, "kind": SCALAR, "index": SCALAR, "request_sha256": HEX, "wall_ms": SCALAR, "timings": TIMINGS,
       "usage": USAGE, "tokens": {"prompt": SCALAR, "reused": SCALAR, "generated": SCALAR},
       "reasoning_chars": SCALAR, "content_chars": SCALAR, "finish_reason": SCALAR, "output_sha256": HEX}

PUBLIC = {
    "schema": SCALAR, "label": SCALAR, "build_info": SCALAR, "model_path": SCALAR, "n_ctx": SCALAR,
    "runner": {"name": SCALAR, "version": SCALAR, "commit": SCALAR},
    "submission": {"label": SCALAR, "date": SCALAR},
    "machine": IDENTITY["machine"],
    "engine": IDENTITY["engine"],
    "model": IDENTITY["model"],
    "config": {"args": [SCALAR], "env": Map(r"^[A-Za-z_][A-Za-z0-9_]*$", SCALAR),
               "server": {"reasoning_budget_tokens": SCALAR, "sampling": SAMPLING, "draft_vocab": SCALAR,
                          "fit_max_tokens": SCALAR, "gpu": SCALAR, "layer_split": [SCALAR]},
               "server_shared": {k: SCALAR for k in SHARED_KEYS},
               "config_id": HEX, "config_origin": SCALAR, "config_origin_verified": SCALAR,
               "calibration": IDENTITY["calibration"]},
    "protocol": {"id": SCALAR, "prompt_sha256": Map(r"^[\w.-]+\.txt$", HEX), "probe_system_sha256": HEX,
                 "sampling": {"temperature": SCALAR, "seed": SCALAR, "cache_prompt": SCALAR},
                 "max_tokens": SCALAR, "warmup": SCALAR, "runs": SCALAR,
                 "probe": {"calls": SCALAR, "skip_first": SCALAR, "max_tokens": SCALAR, "offset": SCALAR,
                           "cache_prompt": SCALAR},
                 "gprobe": {"system_sha256": HEX, "chars": Map(r"^(150|300|450)$", SCALAR), "source": SCALAR,
                            "calls": SCALAR, "skip_first": SCALAR, "max_tokens": SCALAR},
                 "reasoning_sent": {"sizes": SCALAR, "probe": SCALAR}, "unique_line": SCALAR},
    "runs": [RUN],
    "sizes": Map(r"^(short|medium|long|xlong|probe|gprobe150|gprobe300|gprobe450)$",
                 {"prompt_tps": SUMMARY, "decode_tps": SUMMARY, "prompt_ms": SUMMARY, "wall_ms": SUMMARY,
                  "reused_median": SCALAR, "read_median": SCALAR}),
    "validity": {"status": SCALAR, "warnings": [SCALAR], "error": SCALAR},
}

_HEX = re.compile(r"^[0-9a-f]{8,64}$")


def check(value, spec, where="", problems=None) -> list:
    """Every key and value that does not fit spec, as 'where: why'."""
    problems = [] if problems is None else problems
    if spec == SCALAR:
        if not (value is None or isinstance(value, (str, int, float, bool))):
            problems.append(f"{where}: not a scalar")
    elif spec == HEX:
        if not (isinstance(value, str) and _HEX.match(value)):
            problems.append(f"{where}: not a hex digest")
    elif isinstance(spec, list):
        if not isinstance(value, list):
            problems.append(f"{where}: not a list")
        else:
            for i, v in enumerate(value):
                check(v, spec[0], f"{where}[{i}]", problems)
    elif isinstance(spec, Map):
        if not isinstance(value, dict):
            problems.append(f"{where}: not an object")
        else:
            for k, v in value.items():
                if not spec.key.match(str(k)):
                    problems.append(f"{where}.{k}: key not allowed")
                else:
                    check(v, spec.spec, f"{where}.{k}", problems)
    elif isinstance(spec, dict):
        if not isinstance(value, dict):
            problems.append(f"{where}: not an object")
        else:
            for k, v in value.items():
                if k not in spec:
                    problems.append(f"{where}.{k}: not in the public schema".lstrip("."))
                else:
                    check(v, spec[k], f"{where}.{k}".lstrip("."), problems)
    return problems


def prune(value, spec):
    """value with everything outside spec removed (None when value itself does not fit)."""
    if spec == SCALAR:
        return value if value is None or isinstance(value, (str, int, float, bool)) else None
    if spec == HEX:
        return value if isinstance(value, str) and _HEX.match(value) else None
    if isinstance(spec, list):
        return [prune(v, spec[0]) for v in value] if isinstance(value, list) else None
    if isinstance(spec, Map):
        return ({k: prune(v, spec.spec) for k, v in value.items() if spec.key.match(str(k))}
                if isinstance(value, dict) else None)
    if isinstance(spec, dict):
        return {k: prune(v, spec[k]) for k, v in value.items() if k in spec} if isinstance(value, dict) else None
    return None
