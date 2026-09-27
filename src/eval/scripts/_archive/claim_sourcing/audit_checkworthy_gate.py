"""Audit the dev-500 extraction's checkworthy gate + opinion leakage (claim-level, cross-family judge).

Reads the final claims parquet (post-normalize), sends each post + its claims to a judge model
from a different family than the extractor (default Llama-3.3-70B), and tags each claim with the
known failure modes so we can measure the gate's miscalibration in BOTH directions:

  - own_voice_opinion : the outlet's own opinion/characterization/normative take emitted as a bare
                        factual claim (the false-nudge source)
  - unbound_attribution: the post attributes the content to a speaker/group but the claim states it
                        as bare fact (frame/content split)
  - speakerless_attribution: typed attribution with no identifiable attributee
  - trivia            : entertainment/sports/celebrity/lifestyle trivia (no public import)
  - substantive       : concrete claim whose truth carries public/political weight

Writes a per-claim parquet + prints leak rates overall / by cw / by outlet, with examples.

  cd src && uv run python eval/scripts/claim_sourcing/audit_checkworthy_gate.py \
      [-i eval/data/survey_claims/dev500_claims.parquet] [-o <out>.parquet] [--limit N]
"""
import argparse, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parents[3]
BASE = "https://api.deepinfra.com/v1/openai"
KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("DEEPINFRA_API_KEY="))
USAGE = {"in": 0, "out": 0, "cost": 0.0}

SYSTEM = """You audit claims that were extracted from a news outlet's social-media post for a fact-checking pipeline. For each extracted claim, judge it against the ORIGINAL POST and answer five booleans:

- own_voice_opinion: true if the claim is the OUTLET'S OWN opinion, prediction, characterization, or normative/evaluative statement ("X screams for Y", "states can restrain rogue judges", "let's stop calling them elite") presented as if it were a checkable factual claim. A concrete factual proposition (an event, action, ruling, number, quote) is false here, even if charged or partisan. A verifiable-but-contested empirical claim (e.g. "an increasing number of Ivy League students cannot read") is NOT opinion.
- unbound_attribution: true if the POST clearly attributes this content to a speaker or group ("fans claim...", "critics say...", "X said...") but the extracted claim states the content as bare fact, dropping the attribution. Routine anonymous news sourcing ("sources say", "reportedly") does NOT count - that is deliberate policy.
- speakerless_attribution: true if the claim is typed "attribution" but names no identifiable speaker (person, org, group) - e.g. the outlet's own sarcasm or a scare quote with no source.
- trivia: true if the claim is entertainment, celebrity, sports, lifestyle, or human-interest trivia whose truth carries no political or public weight (award nominations, a wedding, a viral animal). Public disorder, crime, deaths, public safety, and anything with civic consequences are NOT trivia.
- substantive: true if the claim is concrete/checkable AND its truth carries political, civic, or public weight (elections, courts, wars, public safety, public health, economy, geopolitics). Hedged phrasing ("what appears to be...") does not disqualify an otherwise substantive claim.

Return only JSON: {"claims": [{"i": <claim index>, "own_voice_opinion": bool, "unbound_attribution": bool, "speakerless_attribution": bool, "trivia": bool, "substantive": bool}]}
One entry per claim, in order."""


def _obj(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def judge(model, post_text, handle, claims):
    lines = "\n".join(f"{i}. [{c['type']}] {c['claim']}" for i, c in enumerate(claims))
    user = f"POST by @{handle}:\n{post_text}\n\nEXTRACTED CLAIMS:\n{lines}"
    payload = {"model": model, "temperature": 0, "max_tokens": 1500,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                USAGE["in"] += u.get("prompt_tokens", 0); USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += u.get("estimated_cost") or 0
                return _obj(d["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(2 * 2 ** a)
    return {}


FLAGS = ["own_voice_opinion", "unbound_attribution", "speakerless_attribution", "trivia", "substantive"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(SRC / "eval/data/survey_claims/dev500_claims.parquet"))
    ap.add_argument("-o", "--output", default=str(SRC / "eval/data/survey_claims/dev500_gate_audit.parquet"))
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="cap on posts (0 = all)")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if "removed_reason" in df.columns:   # v3 schema: audit the KEPT claims (the ledger candidates)
        df = df[df["removed_reason"].isna()]
    groups = list(df.groupby("post_id", sort=False))
    if args.limit:
        groups = groups[:args.limit]
    print(f"auditing {sum(len(g) for _, g in groups)} claims / {len(groups)} posts ({args.model})...", flush=True)
    t0 = time.time()

    def run(item):
        pid, g = item
        recs = g.to_dict("records")
        o = judge(args.model, recs[0]["post_text"], recs[0]["handle"], recs)
        out = {int(e.get("i", -1)): e for e in (o.get("claims") or []) if isinstance(e, dict)}
        rows = []
        for i, r in enumerate(recs):
            e = out.get(i, {})
            rows.append({**{k: r[k] for k in ("post_id", "claim_id", "handle", "domain", "ng_score",
                                              "register", "lean", "post_text", "claim", "type", "checkworthy")},
                         **{f: bool(e.get(f)) for f in FLAGS},
                         "judged": i in out})
        return rows

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        all_rows = [r for rows in ex.map(run, groups) for r in rows]
    out = pd.DataFrame(all_rows)
    out.to_parquet(args.output, index=False)
    dt = time.time() - t0
    print(f"DONE in {dt:.0f}s. judged {int(out.judged.sum())}/{len(out)} claims -> {args.output}")
    print(f"  tokens: {USAGE['in']:,} in / {USAGE['out']:,} out; cost ${USAGE['cost']:.3f}")

    cw = out[out.checkworthy]
    ncw = out[~out.checkworthy]
    print(f"\n=== GATE AUDIT (cw=True n={len(cw)}, cw=False n={len(ncw)}) ===")
    print("LEAKS INTO the verify ledger (cw=True but...):")
    for f in ("own_voice_opinion", "unbound_attribution", "trivia"):
        n = int(cw[f].sum())
        print(f"  {f:24s} {n:4d}  ({100*n/max(1,len(cw)):.1f}% of cw=True)")
    n = int(cw.speakerless_attribution.sum())
    print(f"  {'speakerless_attribution':24s} {n:4d}  (typing bug, any cw)")
    n = int((ncw.substantive & ~ncw.own_voice_opinion).sum())
    print(f"FALSE NEGATIVES (cw=False but substantive, not opinion): {n} ({100*n/max(1,len(ncw)):.1f}% of cw=False)")

    print("\nopinion-leak (cw=True & own_voice_opinion) by outlet:")
    leak = cw[cw.own_voice_opinion]
    if len(leak):
        t = leak.groupby("handle").size().sort_values(ascending=False)
        tot = cw.groupby("handle").size()
        for h, n in t.items():
            print(f"  @{h:20s} {n}/{tot[h]} of its cw claims")
    print("\nEXAMPLES - opinion leak (cw=True):")
    for _, r in leak.head(12).iterrows():
        print(f"  @{r.handle}: [{r.type}] {r.claim[:110]}")
    print("\nEXAMPLES - substantive but cw=False:")
    for _, r in ncw[ncw.substantive & ~ncw.own_voice_opinion].head(12).iterrows():
        print(f"  @{r.handle}: [{r.type}] {r.claim[:110]}")
    print("\nEXAMPLES - trivia with cw=True:")
    for _, r in cw[cw.trivia].head(8).iterrows():
        print(f"  @{r.handle}: [{r.type}] {r.claim[:110]}")


if __name__ == "__main__":
    main()
