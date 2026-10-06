Prompts built from the public Strata source at tag v0.1.32 (MIT), so anyone can rebuild them:

- short.txt: one fixed sentence.
- medium.txt: an instruction line, a blank line, then `git show v0.1.32:src/core/expert_source.cpp | head -c 9000`.
- long.txt: an instruction line, a blank line, then `git show v0.1.32:src/core/expert_source.cpp | head -c 50000`.
- xlong.txt: an instruction line, a blank line, then `git show v0.1.32:src/program/generate.cpp | head -c 100000`.
- probe_lines.txt: one code line per row, taken from `src/kernels/cpu/pool.cpp` at v0.1.37. Each probe call sends
  the fixed system prompt `PROBE_SYSTEM` (in `strata_bench/core.py`, hashed as `protocol.probe_system_sha256`) and one
  of these lines as the user message, with a 1-token output cap.

SHA256SUMS has the hashes of the exact files used; the core refuses to run if a file differs (for example after a
CRLF checkout: `.gitattributes` marks these files `-text`). These are checked copies; there is no generator yet.
