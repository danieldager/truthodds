"""Light filter + dedup pass over the harvested claim pool.

Combines the harvest batches, (1) semantically DEDUPS near-identical claims (MiniLM cosine, keep the
most complete representative), then (2) FILTERS each survivor with an LLM checkworthiness judge — keep
specific, checkable, self-contained factual/attributed claims; drop opinion, vague, superlative/
prediction, or context-dependent ones. Every claim's decision is persisted (auditable). Produces the
clean claim_pool.csv (for Stage-3 verification) + claim_pool_audit.csv (full record).
"""
import os, sys, re, json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, "src")
from eval import ace

OUTDIR = "src/eval/data/survey_claims"
INPUTS = [(OUTDIR + "/claims_smoke.csv", "smoke_32free"), (OUTDIR + "/claims_newfree.csv", "newfree_13"),
          (OUTDIR + "/claims_deep.csv", "deep_scarce"), (OUTDIR + "/claims_headline.csv", "headline_all"),
          (OUTDIR + "/claims_bulk.csv", "bulk")]
DUP_COS = 0.80          # cosine >= this => treat as the same claim (near-dup / same-event)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SMOKE = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >0 => classify only first K (validate)

JUDGE = (
    "You screen candidate claims for a fact-checking study. Each claim becomes a synthetic tweet whose "
    "factual accuracy is then verified against evidence. Decide KEEP or DROP.\n"
    "KEEP if the claim is BOTH:\n"
    " (a) a specific, checkable factual assertion — a concrete event, number, action, or an ATTRIBUTED "
    "statement ('X said/accused/announced Y') that can be judged true/false against evidence; AND\n"
    " (b) self-contained — understandable alone, no dangling reference ('the case', 'the ruling', an "
    "unintroduced name/pronoun).\n"
    "DROP if the claim is mainly: opinion / value-judgment / characterization ('is corrupt', 'deserves "
    "little credit', 'price gouging'); vague or non-falsifiable; a superlative or future prediction with "
    "no concrete verifiable core ('largest in history', 'will happen'); or not self-contained.\n"
    "Attributed claims are KEEP (verifiable as attributions). Reply ONLY with JSON: "
    '{"keep": true|false, "type": "checkable_fact|attributed|opinion|vague|superlative_prediction|not_selfcontained", "reason": "<=12 words"}'
)


def judge(claim):
    resp = ace._client.chat.completions.create(
        model=ace.MODEL, temperature=0.0, max_tokens=200,
        messages=[{"role": "system", "content": JUDGE}, {"role": "user", "content": f"CLAIM: {claim}"}],
    )
    raw = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"keep": True, "type": "checkable_fact", "reason": "parse-fail-default-keep"}
    try:
        d = json.loads(m.group(0))
        return {"keep": bool(d.get("keep", True)), "type": d.get("type", "?"), "reason": d.get("reason", "")}
    except Exception:
        return {"keep": True, "type": "checkable_fact", "reason": "parse-fail-default-keep"}


def main():
    frames = []
    for path, batch in INPUTS:
        if os.path.exists(path):
            df = pd.read_csv(path); df["batch"] = batch
            frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    if "extraction_source" not in d.columns:
        d["extraction_source"] = "body"
    d["extraction_source"] = d["extraction_source"].fillna("body")  # smoke/newfree/deep were body-only
    d["claim"] = d["claim"].astype(str)
    d = d[d["claim"].str.len() > 10].reset_index(drop=True)
    print(f"loaded {len(d)} claims from {len(frames)} batches", flush=True)

    # --- 1. semantic dedup ---
    print("embedding + clustering near-dups...", flush=True)
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(d["claim"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb, emb)
    n = len(d)
    group = [-1] * n
    gid = 0
    for i in range(n):
        if group[i] != -1:
            continue
        group[i] = gid
        for j in range(i + 1, n):
            if group[j] == -1 and float(sim[i][j]) >= DUP_COS:
                group[j] = gid
        gid += 1
    d["dup_group"] = group
    # representative per group = the LONGEST claim (most complete)
    d["is_rep"] = False
    for g, idx in d.groupby("dup_group").groups.items():
        rep = d.loc[idx, "claim"].str.len().idxmax()
        d.loc[rep, "is_rep"] = True
    dup_clusters = d[d.dup_group.map(d.dup_group.value_counts()) > 1]
    print(f"dedup: {n} claims -> {d.is_rep.sum()} unique ({n - d.is_rep.sum()} merged as dups)", flush=True)

    # --- 2. LLM checkworthiness filter (representatives only) ---
    reps = d[d.is_rep].copy()
    if SMOKE:
        reps = reps.head(SMOKE)
    print(f"judging {len(reps)} representatives...", flush=True)
    decisions = []
    for k, (i, r) in enumerate(reps.iterrows(), 1):
        v = judge(r["claim"])
        decisions.append((i, v["keep"], v["type"], v["reason"]))
        if k % 20 == 0:
            print(f"  {k}/{len(reps)}", flush=True)
    dec = {i: (kp, tp, rs) for i, kp, tp, rs in decisions}
    d["filter_keep"] = d.index.map(lambda i: dec.get(i, (None, None, None))[0])
    d["filter_type"] = d.index.map(lambda i: dec.get(i, (None, None, None))[1])
    d["filter_reason"] = d.index.map(lambda i: dec.get(i, (None, None, None))[2])
    # a non-representative is dropped as a duplicate
    d["decision"] = d.apply(
        lambda r: ("kept" if r["filter_keep"] else "dropped_filter") if r["is_rep"] else "dropped_dup", axis=1)

    if SMOKE:
        print(d[d.is_rep].head(SMOKE)[["cell", "outlet", "claim", "filter_keep", "filter_type", "filter_reason"]]
              .to_string(index=False), flush=True)
        return

    pool = d[d["decision"] == "kept"].copy()
    # dup metadata on the kept representative
    grp_outlets = d.groupby("dup_group")["outlet"].apply(lambda s: sorted(set(s))).to_dict()
    pool["dup_count"] = pool["dup_group"].map(d.dup_group.value_counts())
    pool["dup_outlets"] = pool["dup_group"].map(lambda g: ", ".join(grp_outlets[g]))
    cols = ["cell", "outlet", "claim", "risk_reason", "filter_type", "extraction_source", "dup_count", "dup_outlets", "url", "headline", "batch"]
    pool[cols].to_csv(OUTDIR + "/claim_pool.csv", index=False)
    audit_cols = ["cell", "outlet", "claim", "decision", "filter_keep", "filter_type", "filter_reason",
                  "dup_group", "is_rep", "batch", "url"]
    d[audit_cols].to_csv(OUTDIR + "/claim_pool_audit.csv", index=False)

    print("\n=== RESULT ===", flush=True)
    print(f"{n} raw -> {int(d.is_rep.sum())} after dedup -> {len(pool)} after filter", flush=True)
    print("\ndrop reasons:", flush=True)
    print(d[d.decision != "kept"]["decision"].value_counts().to_string(), flush=True)
    print("\ndropped-by-filter types:", flush=True)
    print(d[d.decision == "dropped_filter"]["filter_type"].value_counts().to_string(), flush=True)
    print("\nkept per cell:", flush=True)
    print(pool["cell"].value_counts().sort_index().to_string(), flush=True)
    if len(dup_clusters):
        print(f"\n{dup_clusters.dup_group.nunique()} dup-clusters, e.g.:", flush=True)
        for g in dup_clusters.dup_group.unique()[:4]:
            members = d[d.dup_group == g]["claim"].tolist()
            print("  * " + " || ".join(m[:60] for m in members), flush=True)
    print(f"\nwrote {OUTDIR}/claim_pool.csv + claim_pool_audit.csv", flush=True)


if __name__ == "__main__":
    main()
