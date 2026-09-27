"""CLI wiring, offline (the transport is faked at the runner)."""
import json

from conftest import FakeTransport, ok_reply

from deepinfra_runner import cli, runner


def _fake_transport(monkeypatch, content="OK"):
    tr = FakeTransport(lambda body, n: ok_reply(body, content=content))
    monkeypatch.setattr(runner, "RequestsTransport", lambda pool=0: tr)
    return tr


def test_items_run_writes_parsed_output(tmp_path, monkeypatch):
    tr = _fake_transport(monkeypatch, content="Bonjour.")
    items = tmp_path / "in.jsonl"
    items.write_text('{"id":"a","text":"hi"}\n{"id":"b","text":"yo"}\n')
    prompt = tmp_path / "p.txt"
    prompt.write_text("Reponds en un mot.")
    out = tmp_path / "out.jsonl"

    rc = cli.main(["--items", str(items), "--prompt", str(prompt), "--out", str(out),
                   "--model", "deepseek-ai/DeepSeek-V4-Flash", "--obs", str(tmp_path / "o.jsonl")])
    assert rc == 0 and tr.n == 2
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["output"] for r in rows] == ["Bonjour.", "Bonjour."]   # parse hook is wired
    assert all(r["error"] is None for r in rows)
    # the system prompt reached the request
    assert tr.calls[0][2]["messages"][0] == {"role": "system", "content": "Reponds en un mot."}

    tr2 = _fake_transport(monkeypatch)                              # rerun: served from cache
    assert cli.main(["--items", str(items), "--prompt", str(prompt), "--out", str(out),
                     "--model", "deepseek-ai/DeepSeek-V4-Flash", "--obs", str(tmp_path / "o.jsonl")]) == 0
    assert tr2.n == 0


def test_status_runs(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPINFRA_RUNNER_OBS", str(tmp_path / "none.jsonl"))
    assert cli.main(["status"]) == 0
    printed = capsys.readouterr().out
    assert "api key: set" in printed and "test-key-not-real" not in printed
