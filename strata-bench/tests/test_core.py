import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from strata_bench import core, schema, validate
from strata_bench.__main__ import main

CFG = {"exe": "E:\\strata-0139\\engine\\strata.exe", "args": [
    "--pack", "E:\\strata-data\\packs\\swift", "--native", "E:\\m\\Swift-IQ3_XXS-00001-of-00002.gguf",
    "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768", "--kv", "int8"],
    "env": {"STRATA_IQ_MT_MIN": "1"}}
MODEL = "E:\\m\\Swift-IQ3_XXS-00001-of-00002.gguf"
H64 = "ab" * 32


class Fake(BaseHTTPRequestHandler):
    """A Strata server.py stand-in: prompt_n is what was read, cache_n what was reused."""
    st = {}

    @classmethod
    def reset(cls):
        cls.st = {"model": MODEL, "n_ctx": 32768, "build": "Strata 0.1.39", "reused": 0, "probe_read": 20,
                  "probe_reused": 400, "fail_after": None, "no_timings": False, "calls": [], "shared": {},
                  "settings": True, "api_key": None}

    def log_message(self, *a):
        pass

    def reply(self, d, code=200):
        b = json.dumps(d).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self):
        k = Fake.st["api_key"]
        return not k or self.headers.get("authorization") == "Bearer " + k

    def do_GET(self):
        s = Fake.st
        if self.path == "/props":
            p = {"model_path": s["model"], "default_generation_settings": {"n_ctx": s["n_ctx"]} if s["n_ctx"] else {}}
            if s["build"]:
                p["build_info"] = s["build"]
            return self.reply(p)
        if not self.authed():
            return self.reply({"error": "unauthorized"}, 401)
        if self.path == "/settings" and s["settings"]:
            return self.reply({"shared": True, "defaults": s["shared"]})
        if self.path == "/v1/models":
            return self.reply({"data": []})
        self.reply({}, 404)

    def do_POST(self):
        s = Fake.st
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if not self.authed():
            return self.reply({"error": "unauthorized"}, 401)
        s["calls"].append(body)
        if s["fail_after"] is not None and len(s["calls"]) > s["fail_after"]:
            return self.reply({}, 500)
        msg = {"choices": [{"message": {"content": "SAFE", "reasoning_content": "hmm"}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 1, "secret_extra": "x"}}
        if s["no_timings"]:
            return self.reply(msg)
        probe = body["messages"][0]["role"] == "system"
        read, reused = (s["probe_read"], s["probe_reused"]) if probe else (100, s["reused"])
        msg["timings"] = {"prompt_n": read, "cache_n": reused, "prompt_ms": 50.0, "prompt_per_second": 2000.0,
                          "predicted_n": 1, "predicted_ms": 5.0, "predicted_per_second": 200.0, "private_case": "X"}
        self.reply(msg)


class CoreTest(unittest.TestCase):
    def setUp(self):
        Fake.reset()
        self.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp)

    def go(self, sizes=("medium",), origin="setup-default", label="t", cfg=CFG, identity=None, api_key=None):
        return core.run(self.base, cfg, origin, list(sizes), label, self.tmp, identity, api_key=api_key,
                        log=lambda *a: None)

    def doc(self, label="t"):
        return json.loads((self.tmp / f"{label}.result.json").read_text())

    # --- the normal path
    def test_full_run_identity_panel_fields_and_export(self):
        self.assertEqual(self.go(("probe", "medium")), 0)
        d = self.doc()
        self.assertEqual((d["label"], d["build_info"], d["model_path"], d["n_ctx"]),
                         ("t", "Strata 0.1.39", "Swift-IQ3_XXS-00001-of-00002.gguf", 32768))
        self.assertEqual(d["config"]["config_origin"], "setup-default")
        self.assertFalse(d["config"]["config_origin_verified"])
        self.assertEqual(d["config"]["args"][1], "<pack>")
        self.assertEqual(d["config"]["args"][3], "<native>")
        self.assertEqual(d["sizes"]["medium"]["prompt_tps"]["median"], 2000.0)
        self.assertEqual(d["sizes"]["probe"]["prompt_ms"]["median"], 50.0)
        self.assertNotIn("prompt_tps", d["sizes"]["probe"])
        self.assertEqual(len([r for r in d["runs"] if r["size"] == "probe"]), core.PROBE_CALLS)
        self.assertEqual(d["runs"][0]["reasoning_chars"], 3)
        self.assertEqual(d["runs"][0]["usage"], {"completion_tokens": 1})          # unknown usage key pruned
        self.assertNotIn("private_case", d["runs"][0]["timings"])                   # unknown timing key pruned
        self.assertEqual(d["validity"]["status"], "ok")                             # a normal probe is not "reused"
        self.assertEqual(validate.problems(d), [])

    def test_gprobe_three_sizes_distinct_excerpts_and_export(self):
        self.assertEqual(self.go(("gprobe",)), 0)
        d = self.doc()
        self.assertEqual(sorted(d["sizes"]), ["gprobe150", "gprobe300", "gprobe450"])
        calls = Fake.st["calls"]
        self.assertEqual(len(calls), 3 * core.PROBE_CALLS)
        self.assertTrue(all(c["messages"][0]["content"] == core.GPROBE_SYSTEM and c["max_tokens"] == 1 for c in calls))
        for n, size in enumerate(core.GPROBE_CHARS):
            users = [c["messages"][1]["content"] for c in calls[n * core.PROBE_CALLS:(n + 1) * core.PROBE_CALLS]]
            self.assertEqual({len(u) for u in users}, {core.GPROBE_CHARS[size]})
            self.assertEqual(len(set(users)), core.PROBE_CALLS)                      # a different excerpt per call
        self.assertEqual(d["sizes"]["gprobe300"]["read_median"], 20)
        self.assertEqual(validate.problems(d), [])
        d["protocol"]["gprobe"]["system_sha256"] = H64
        self.assertIn("gprobe is not standard-v1's", validate.problems(d))

    def test_unique_line_is_constant_per_size_and_run(self):
        self.go()
        firsts = [c["messages"][0]["content"].split("\n", 1)[0] for c in Fake.st["calls"]]
        self.assertEqual(firsts, ["[strata-bench medium warmup 1]", "[strata-bench medium run 1]",
                                  "[strata-bench medium run 2]", "[strata-bench medium run 3]"])
        self.assertTrue(all(c["cache_prompt"] is False for c in Fake.st["calls"]))
        d = self.doc()
        self.assertEqual(len({r["request_sha256"] for r in d["runs"]}), 4)

    # --- preflight
    def test_preflight_refusals_measure_nothing(self):
        cases = [("model", "E:\\m\\Other.gguf"), ("n_ctx", 65536), ("n_ctx", None), ("build", None),
                 ("model", "")]
        for i, (k, v) in enumerate(cases):
            Fake.reset()
            Fake.st[k] = v
            self.assertEqual(self.go(label=f"p{i}"), 2, (k, v))
            self.assertFalse((self.tmp / f"p{i}.result.json").exists())
            self.assertEqual(Fake.st["calls"], [])

    def test_refuses_overwrite(self):
        self.go()
        self.assertEqual(self.go(), 2)

    def test_api_key(self):
        Fake.st["api_key"] = "sk-secret-123456789"
        self.assertEqual(self.go(label="a"), 2)
        self.assertEqual(self.go(label="b", api_key="sk-secret-123456789"), 0)
        self.assertNotIn("sk-secret", (self.tmp / "b.result.json").read_text())

    def test_changed_prompt_refused(self):
        tmp = self.tmp / "prompts"
        shutil.copytree(core.PROMPTS, tmp)
        (tmp / "medium.txt").write_bytes((tmp / "medium.txt").read_bytes().replace(b"\n", b"\r\n", 1))
        with mock.patch.object(core, "PROMPTS", tmp):
            self.assertEqual(self.go(), 2)

    # --- measurement validity
    def test_failure_keeps_partial_result(self):
        Fake.st["fail_after"] = 2
        self.assertEqual(self.go(), 3)
        d = self.doc()
        self.assertEqual(d["validity"]["status"], "incomplete")
        self.assertEqual(len(d["runs"]), 2)

    def test_missing_timings_is_incomplete(self):
        Fake.st["no_timings"] = True
        self.assertEqual(self.go(), 3)
        self.assertEqual(self.doc()["validity"]["status"], "incomplete")

    def test_size_reuse_is_flagged(self):
        Fake.st["reused"] = 500
        self.go()
        self.assertEqual(self.doc()["validity"]["status"], "warnings")

    def test_probe_full_reuse_flagged(self):
        Fake.st["probe_read"], Fake.st["probe_reused"] = 0, 420
        self.go(("probe",))
        self.assertEqual(self.doc()["validity"]["status"], "warnings")

    # --- config identity
    def test_server_fields_keep_values_and_change_config_id(self):
        ids = {core.config_block({**CFG, "reasoning_budget_tokens": n}, "custom-1", False)["config_id"]
               for n in (0, 512, 4096)}
        self.assertEqual(len(ids), 3)
        b = core.config_block({**CFG, "fit_max_tokens": 1024}, "custom-1", False)
        self.assertEqual(b["server"]["fit_max_tokens"], 1024)
        a = core.config_block(CFG, "custom-1", False)["config_id"]
        self.assertEqual(a, core.config_block({**CFG, "port": 9, "log": "E:\\x.log"}, "custom-1", False)["config_id"])

    def test_env_secrets_redacted_engine_knobs_kept(self):
        cfg = {**CFG, "env": {"HF_TOKEN": "hf_abcdefghijk", "OPENAI_API_KEY": "x", "STRATA_API_KEY": "y",
                              "STRATA_INDEXER_PER_TOKEN": "1", "STRATA_GDN_KEYHEAD": "2"}}
        env = core.scrub_config(cfg)["env"]
        self.assertEqual([env[k] for k in ("HF_TOKEN", "OPENAI_API_KEY", "STRATA_API_KEY")], ["<redacted>"] * 3)
        self.assertEqual((env["STRATA_INDEXER_PER_TOKEN"], env["STRATA_GDN_KEYHEAD"]), ("1", "2"))

    def test_shared_settings_recorded_in_config_id(self):
        self.go(label="a")
        Fake.st["shared"] = {"reasoning_effort": "low", "unknown": "z"}
        self.go(label="b")
        a, b = self.doc("a"), self.doc("b")
        self.assertNotEqual(a["config"]["config_id"], b["config"]["config_id"])
        self.assertEqual(b["config"]["server_shared"], {"reasoning_effort": "low"})
        self.assertEqual(b["validity"]["status"], "warnings")

    def test_identity_cannot_override_observed_and_is_pruned(self):
        ident = {"engine": {"build_info": "Strata 0.1.32", "binary_sha256": H64},
                 "model": {"file": "OTHER.gguf", "n_ctx": 65536, "repo": "https://huggingface.co/u/m"},
                 "machine": {"ram_type": "DDR4", "hostname": "DESKTOP-1",
                             "gpus": [{"name": "RTX 5090", "private_case": "PRIVATE_CANARY"}]}}
        self.go(identity=ident)
        d = self.doc()
        self.assertEqual(d["engine"]["build_info"], "Strata 0.1.39")
        self.assertEqual(d["engine"]["binary_sha256"], H64)
        self.assertEqual(d["model"]["file"], "Swift-IQ3_XXS-00001-of-00002.gguf")
        self.assertEqual(d["machine"], {"ram_type": "DDR4", "gpus": [{"name": "RTX 5090"}]})
        self.assertNotIn("PRIVATE_CANARY", json.dumps(d))
        self.assertEqual(d["validity"]["status"], "warnings")
        self.assertEqual(validate.problems(d), [])                 # the HF URL is not an "absolute path"

    def test_bad_origin_rejected(self):
        with self.assertRaises(SystemExit):
            self.go(origin="tuned")

    # --- export gate
    def test_validator_rejects_private_and_leaks_at_any_depth(self):
        self.go()
        d = self.doc()
        bad = [{"note": "x"}, {"engine": {"exe": "E:\\strata\\strata.exe"}},
               {"machine": {"hostname": "DESKTOP-7Q2K1"}},
               {"machine": {"gpus": [{"name": "RTX 5090", "private_case": "P"}]}},
               {"sizes": {"medium": {**d["sizes"]["medium"], "private_case": "P"}}},
               {"runs": [{**d["runs"][0], "timings": {"private_case": "P"}}]},
               {"protocol": {**d["protocol"], "sampling": {"private_case": "P"}}},
               {"protocol": {**d["protocol"], "id": "private-x"}},
               {"protocol": {**d["protocol"], "prompt_sha256": {"medium.txt": H64}}},
               {"submission": {"label": "E:/strata-data/x"}}, {"model": {"file": "/opt/models/x.gguf"}},
               {"validity": {"status": "incomplete"}},
               {"config": {**d["config"], "env": {"K": "sk-live-abcdefghij"}}}]
        for b in bad:
            self.assertTrue(validate.problems({**d, **b}), b)
        self.assertEqual(validate.problems({**d, "machine": {"os": "Windows 10.0.26200"}}), [])

    def test_schema_prune_and_check_agree(self):
        v = {"gpus": [{"name": "a", "x": 1}], "ram_type": "DDR4", "zzz": 1}
        self.assertEqual(schema.check(schema.prune(v, schema.IDENTITY["machine"]), schema.IDENTITY["machine"]), [])

    def test_cli(self):
        cfg = self.tmp / "c.json"
        cfg.write_text(json.dumps(CFG))
        rc = main(["run", "--base", self.base, "--config", str(cfg), "--origin", "custom-2", "--label", "c",
                   "--out", str(self.tmp), "--sizes", "medium"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.doc("c")["config"]["config_origin"], "custom-2")
        self.assertEqual(main(["validate", str(self.tmp / "c.result.json")]), 0)


if __name__ == "__main__":
    unittest.main()
