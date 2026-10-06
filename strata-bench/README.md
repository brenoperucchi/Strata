# strata-bench (core)

Measures a running Strata server with fixed public prompts and writes one `result.json` (schema `strata-bench/v1`)
per configuration. Standard library only, Python 3.10+.

```
python3 -m strata_bench run --base http://127.0.0.1:18299 --config strata-swift.json \
    --origin setup-default --label 0139-swift-default --out results/ [--sizes probe,medium,long,xlong]
python3 -m strata_bench validate results/*.result.json      # public-export gate
```

- Attach mode: the caller starts and stops the server. The core refuses to measure if the server's model file or
  context differs from the config's (exit 2), never overwrites a result, and saves after every call (exit 3 keeps a
  partial, marked incomplete).
- `--origin` is `setup-default` (the config the release's setup writes) or `custom-N`; the config's `args`/`env` are
  stored with paths replaced by roles (`<pack>`, `<native>` ...) and hashed into `config_id`.
- Protocol `standard-v1`: prompts in `strata_bench/prompts` (hashes in `SHA256SUMS`), temperature 0, seed 42,
  256-token cap, 1 warm-up + 3 runs, a constant first line per (size, run); probes: fixed system prompt, 30 one-token
  calls, the first 2 skipped.
- `gprobe` (opt-in, add it to `--sizes`): 30 calls at each of ~150/300/450 tokens shaped like an agent's classifier
  request - a short fixed system prompt and a different excerpt of `xlong.txt` per call, 1 token out, the first 2
  skipped. It measures the prompt path below Strata's 1024-token streamed walk, which the line probes barely touch.
  Results go to `sizes.gprobe150/300/450` (`prompt_ms`, `wall_ms`, `read_median`, `reused_median`).
- Flat `label` / `build_info` / `model_path` (file name only) / `n_ctx` at the top and `sizes{}` keep the `prompt_tps` / `decode_tps` / `prompt_ms` medians that existing dashboards read.

Tests: `python3 -m unittest discover -s tests`

Run it from this directory (`cd strata-bench`). The prompts are excerpts of the Strata source; see
`strata_bench/prompts/README.md` for how each one was cut. It lives in this fork, not in upstream Strata, and is not
maintained by the Strata author.
