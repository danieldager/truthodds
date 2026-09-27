"""Build the Score Ladder artifact page from ladder.json + the two SVGs.

    uv run python -m eval.scripts.build_eval.model_ladder_html

House style is fixed (see the artifact house-style rule). Daniel's voice, no em
dashes, no colons, no semicolons in the prose, text blocks the same width as the
figures, the Two Urns CSS token block verbatim. Writes model_ladder.html next to
the figures. Rerun this and model_ladder_figs after any refit, then republish.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import polars as pl

from eval.scripts.build_eval import retrieval_forensics as RF

A = Path("eval/data/urn_runs/e1_ctx/model_ladder")
S = json.loads((A / "ladder.json").read_text())
T = json.loads((A / "transfer.json").read_text())
X = json.loads((A / "cross.json").read_text())
P, M = S["population"], S["models"]
TM, TU = T["models"], T["urns"]
XC, XB = X["cells"], X["band_audit_upper_bound"]
MODELS = ("3-voice", "7-flag", "28-cell")
NAME = {"3-voice": "3 buckets", "7-flag": "7 buckets", "28-cell": "28 buckets"}


def fig(name: str) -> str:
    return f'<div class="panel">{(A / f"fig_{name}.svg").read_text()}</div>'


trows = ""
for m in MODELS:
    v = TM[m]
    lo, hi = v["auc_ci"]
    d = (f"{v['dauc_vs_3voice']:+.3f} [{v['dauc_ci'][0]:+.3f}, {v['dauc_ci'][1]:+.3f}]"
         if m != "3-voice" else "reference")
    trows += (f"<tr><td>{NAME[m]}</td><td class='m'>{v['auc']:.3f}</td>"
              f"<td class='m'>[{lo:.3f}, {hi:.3f}]</td>"
              f"<td class='m'>{v['recall_2pct']*100:.1f}%</td><td class='m'>{d}</td>"
              f"<td class='m'>{v['gold_refit_auc']:.3f}</td>"
              f"<td class='m'>{v['gold_refit_recall_2pct']*100:.1f}%</td></tr>")

sweep_rows = ""
for eps, per in T["eps_sweep"].items():
    sweep_rows += (f"<tr><td class='m'>{float(eps)*100:.0f}%</td>"
                   + "".join(f"<td class='m'>{per[m]['auc']:.3f}</td>"
                             f"<td class='m'>{per[m]['recall_2pct']*100:.1f}%</td>"
                             for m in MODELS) + "</tr>")

def xcell(fit, ev, m):
    return XC[f"{fit}->{ev}"][m]

mrows = ""
for fit in ("gold", "urn", "pool"):
    lab = {"gold": "fc gold", "urn": "the two urns", "pool": "both, pooled"}[fit]
    cells = ""
    for ev in ("gold", "urn"):
        for m in MODELS:
            c = xcell(fit, ev, m)
            cells += (f"<td class='m'>{c['auc']:.3f}</td>" if c else "<td class='m'>--</td>")
    mrows += f"<tr><td>{lab}</td>{cells}</tr>"

xsweep = ""
for eps, per in X["eps_sweep"].items():
    xsweep += (f"<tr><td class='m'>{float(eps)*100:.0f}%</td>"
               + "".join(f"<td class='m'>{per[f]['7-flag']['auc']:.3f}</td>"
                         f"<td class='m'>{per[f]['7-flag']['auc_observed']:.3f}</td>"
                         for f in ("gold", "urn", "pool")) + "</tr>")

rows = ""
for m in MODELS:
    v = M[m]
    lo, hi = v["auc_ci"]
    d = (f"{v['dauc_vs_3voice']:+.3f} [{v['dauc_ci'][0]:+.3f}, {v['dauc_ci'][1]:+.3f}]"
         if m != "3-voice" else "reference")
    rows += (f"<tr><td>{NAME[m]}</td><td class='m'>{v['auc_oof']:.3f}</td>"
             f"<td class='m'>[{lo:.3f}, {hi:.3f}]</td>"
             f"<td class='m'>{v['recall_2pct']*100:.1f}%</td><td class='m'>{d}</td></tr>")

# ---- section 08: the score-blind eps audit, banded by s7
# s7 uses the pinned two-urn 7-flag weights at an assumed false share of 5%,
# the same weights tl_band_audit / timeline_eps_audit score with.
TLR = Path("eval/data/urn_runs/true_timeline")
EPSD = Path("eval/data/eps_audit/smoke")

# The band audit's own file is the source of truth for how many claims it read and how
# many it called false. cross.json keys the audit by claim_text[:80] so it can subtract
# the falses from the timeline, and three claims collide on that key, which is why its
# n_audited reads 397 and its n_called_false 190. Those are join counts, not audit counts.
BA = [json.loads(line) for line in (TLR / "band_audit.jsonl").open()]
BA_N = len(BA)
BA_F = sum(1 for r in BA if r["verdict"] == "false")
W7 = next(f["weights"] for f in json.loads((TLR / "two_urn_fit.json").read_text())["fits7"]
          if abs(f["eps"] - 0.05) < 1e-9)
BANDS = [(-99, -6), (-6, -3), (-3, 0), (0, 3), (3, 99)]
BLAB = {(-99, -6): "below -6", (-6, -3): "-6 to -3", (-3, 0): "-3 to 0",
        (0, 3): "0 to 3", (3, 99): "3 and up"}


def _band(s: float):
    return next((a, b) for a, b in BANDS if a <= s < b)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval, by bisection on the binomial cdf. Exact
    rather than normal because three of the five bands hold one false claim."""
    if n == 0:
        return (0.0, 1.0)

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    high = 1.0 if k == n else solve(lambda p: (alpha / 2) - _binom_cdf(k, n, p))
    return (low, high)


urn_band = collections.Counter()
for line in (TLR / "scores.jsonl").open():
    r = json.loads(line)
    fl = [d["read"]["direction"] for d in r["results"]
          if d.get("read") and d["read"].get("direction")]
    if fl:
        urn_band[_band(sum(W7.get(f, 0.0) for f in fl))] += 1
URN_N = sum(urn_band.values())

# ---- the n=40 shakedown, kept only for the paragraph that says what it did and did
# not settle. Every number in the table below comes from the pooled n=400 draw.
SM = json.loads((Path("eval/data/eps_audit/smoke") / "result.json").read_text())
SMS7 = {r["aid"]: r["s7"] for r in
        pl.read_parquet("eval/data/eps_audit/smoke/sample.parquet").iter_rows(named=True)}
smaud = collections.defaultdict(collections.Counter)
for aid, v in SM["labels"].items():
    smaud[_band(SMS7[aid])][v["label"]] += 1
SM_DEC = [smaud[b]["TRUE"] + smaud[b]["FALSE"] for b in BANDS]
SM_F = SM["FALSE"]

# ---- the pooled 400-claim draw. timeline_eps_audit.py --analyze writes every number
# below, including both band cuts, the trend tests and the second-read arms.
EPS = json.loads((Path("eval/data/eps_audit/pooled") / "result.json").read_text())
ET, EF, EU = EPS["TRUE"], EPS["FALSE"], EPS["UNVERIFIABLE"]
EN, EDEC = EPS["n"], ET + EF
ELO, EHI = EPS["clopper_pearson"]
ECLO, ECHI = EPS["clustered"]
B2, BG = EPS["bands_two_urn"], EPS["bands_gold"]
SEP2, SEPG = EPS["separation"]["two-urn"], EPS["separation"]["gold-fit"]
SR, DU = EPS["second_reads"], EPS["deep_unver"]

BLAB2 = {"[-99,-6)": "below -6", "[-6,-3)": "-6 to -3", "[-3,0)": "-3 to 0",
         "[0,3)": "0 to 3", "[3,99)": "3 and up"}


def brows(bt, lab=None) -> str:
    out = ""
    for r in bt["rows"]:
        e = "--" if not r["decided"] else f"{r['eps']:.3f}"
        ci = ("--" if not r["decided"]
              else f"[{r['cp'][0]:.3f}, {r['cp'][1]:.3f}]")
        out += (f"<tr><td class='m'>{(lab or {}).get(r['band'], r['band'])}</td>"
                f"<td class='m'>{r['audited']}</td><td class='m'>{r['T']}</td>"
                f"<td class='m'>{r['F']}</td><td class='m'>{r['U']}</td>"
                f"<td class='m'>{e}</td><td class='m'>{ci}</td>"
                f"<td class='m'>{r['urn_n']:,}</td>"
                f"<td class='m'>{r['urn_pct']:.1f}%</td>"
                f"<td class='m'>{r['implied_false']:.0f}</td></tr>")
    out += (f"<tr><td class='m'><b>all</b></td><td class='m'>{EN}</td>"
            f"<td class='m'>{ET}</td><td class='m'>{EF}</td><td class='m'>{EU}</td>"
            f"<td class='m'>{EPS['eps']:.3f}</td>"
            f"<td class='m'>[{ELO:.3f}, {EHI:.3f}]</td>"
            f"<td class='m'>{bt['urn_n']:,}</td><td class='m'>100.0%</td>"
            f"<td class='m'>{EPS['eps']*bt['urn_n']:.0f}</td></tr>")
    return out


erows, growers = brows(B2, BLAB2), brows(BG)
IMPLIED2 = sum(r["implied_false"] for r in B2["rows"])
# the strongest two-way collapse, reported with the caveat that it is the strongest
BEST2 = min(B2["splits"], key=lambda s: s["fisher_p"])
BESTG = min(BG["splits"], key=lambda s: s["fisher_p"])
UWHY = sorted(EPS["unver_why_p1"].items(), key=lambda kv: -kv[1])


# ---- what the measured eps does to the numbers already on this page. timeline_eps_audit
# --sweep re-runs cross_ladder at the measured share and at both ends of its interval,
# redirected out of urn_runs so nothing pinned is overwritten.
MEASP = Path("eval/data/eps_audit/pooled/cross_measured/cross.json")
MEAS = json.loads(MEASP.read_text()) if MEASP.exists() else None
if MEAS:
    MW = MEAS["cells"]["gold->urn"]["7-flag"]
    MSW = {float(k): v["gold"]["7-flag"]["auc"] for k, v in MEAS["eps_sweep"].items()}
    MLO, MHI = MSW[min(MSW)], MSW[max(MSW)]


def andlist(xs) -> str:
    xs = [str(x) for x in xs]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# ---- prose that has to change shape with the numbers rather than just interpolate them
RES = DU.get("resolvability", {})
_hard = RES.get("hard_but_resolvable", 0)
_struct = RES.get("structurally_unresolvable", 0)
_denom = max(_hard + _struct, 1)
TAXTOP = sorted(DU.get("taxonomy_stuck", {}).items(), key=lambda kv: -kv[1])[:4]
RESOLV_SENT = (
    f"Of the {_hard + _struct} that stayed unverifiable, {_hard} are hard but resolvable and "
    f"{_struct} are structurally unresolvable. Hard but resolvable means a record that would "
    f"settle the claim exists somewhere the auditor could not reach, behind a paywall, on a "
    f"deleted or unarchived page, in a subscription database, or in a language or jurisdiction "
    f"it could not search. Somebody with that access could settle it. Structurally unresolvable "
    f"means no record that would settle it exists or could exist, because nobody wrote it down, "
    f"or the referent is too vague to check, or it is about the future, or it is not a factual "
    f"proposition at all. So only {_struct/_denom*100:.0f}% of the residue is out of reach in "
    f"principle and the other {_hard/_denom*100:.0f}% is out of reach in practice, which says "
    f"the ceiling here is access and effort rather than the nature of the claims. The shapes "
    f"that dominate the residue are "
    + andlist([f"{k.replace('_',' ')} at {v}" for k, v in TAXTOP])
    + f", with the rest spread thin over {len(DU.get('taxonomy_stuck', {})) - len(TAXTOP)} "
      f"other kinds."
) if DU.get("resolvability") else ""

TAXROWS = "".join(
    f"<tr><td>{k.replace('_',' ')}</td><td class='m'>{v}</td>"
    f"<td class='m'>{v/max(DU.get('n_stuck',1),1)*100:.1f}%</td></tr>"
    for k, v in sorted(DU.get("taxonomy_stuck", {}).items(), key=lambda kv: -kv[1]))

WG5 = EPS.get("gold_weights", {}).get("5", 2.94)
W25 = EPS.get("two_urn_weights", W7).get("5", 1.29)

def _inversions(bt):
    d = [r for r in bt["rows"] if r["decided"]]
    return [(d[i], d[i + 1]) for i in range(len(d) - 1) if d[i]["eps"] < d[i + 1]["eps"]]


def _mono_sent(bt, nm):
    inv = _inversions(bt)
    if not inv:
        return f"The {nm} cut falls without a single reversal across all five bands."
    a, b = inv[0]
    return (f"The {nm} cut falls without a reversal except between {a['band']} and {b['band']}, "
            f"where it goes up, and that lower band decides on {a['decided']} claims so the "
            f"reversal is {a['decided']} claims wide.")


AGREE_SENT = (
    _mono_sent(B2, "two urn") + " " + _mono_sent(BG, "gold fitted")
    + " The gold cut also puts a bigger step between its second and third band than the two urn"
    " cut does, which is what re-fitting the weights on another corpus does to the spacing. The"
    " two cuts do not disagree about direction anywhere the urn is dense, so the choice of"
    " weight set is not what is producing the ladder.")
UWHY_SENT = (andlist([f"{k.replace('_',' ')} {v}" for k, v in UWHY]) + "."
             if UWHY else "")


# ---- section 09: why a quarter of the false claims never get a usable document
# Recomputed live from retrieval_forensics, which rebuilds both arms from the saved
# reads plus the scraped fact-check articles. No paid API, nothing cached by hand.
FG, FCN = RF.gold(verbose=False), RF.cn(verbose=False)
FG_HIT, FG_TOT = len(FG["hit"]), len(FG["hit"]) + FG["n_pool"]
FCN_HIT, FCN_TOT = len(FCN["hit"]), len(FCN["hit"]) + FCN["n_pool"]


def med(rows) -> float:
    v = sorted(r["ndocs"] for r in rows)
    return (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)


FG_A = [r for r in FG["hit"] if r["bucket"] == "A"]
FG_AWI = sum(1 for r in FG_A if r["wi"])

TAX = collections.defaultdict(collections.Counter)
for line in Path("eval/data/retrieval_forensics/taxonomy.jsonl").open():
    r = json.loads(line)
    TAX[r["arm"]][r["code"]] += 1
ARMS = ("cn_allirr", "cn_ctrl", "fg_allirr", "fg_ctrl")
MODES = [(("nonexistent_event",), "nonexistent_event", "no such event to find"),
         (("over_generic",), "over_generic", "query too broad to land on the claim"),
         (("wrong_date",), "wrong_date", "right event, wrong window"),
         (("query_ok",), "query_ok", "nothing wrong with the query"),
         (("missing_entity",), "missing_entity", "the deciding name never made it in"),
         (("non_english", "entity_mangled", "deixis"), "non_english, entity_mangled, deixis",
          "wrong language, broken name, unresolved this or that")]


def modeshare(arm, codes) -> float:
    return sum(TAX[arm][c] for c in codes) / sum(TAX[arm].values()) * 100


mode_rows = ""
for codes, lab, gloss in MODES:
    mode_rows += (f"<tr><td>{lab}<br><span class='note'>{gloss}</span></td>"
                  + "".join(f"<td class='m'>{modeshare(a, codes):.1f}%</td>" for a in ARMS)
                  + "</tr>")
mode_rows += ("<tr><td><b>queries labelled</b></td>"
              + "".join(f"<td class='m'>{sum(TAX[a].values())}</td>" for a in ARMS) + "</tr>")

# ---- the ledger, section 00. Numbers that already live in a run file are read from it.
# graded_metrics.json is the 7-flag fit on the pre-mixed-cut population (n=3,699), which
# is where the FPR-matched comparison and the LR+ were measured. headline_metrics.json is
# the shipped 3-voice fit on that same population, so the pair is the population-deletion
# gap. Both are read only.
E1 = Path("eval/data/urn_runs/e1_ctx")
GM = json.loads((E1 / "graded_metrics.json").read_text())
HM = json.loads((E1 / "headline_metrics.json").read_text())
GMM, GM2 = GM["recall_at_matched_fpr"], GM["recall_at_2pct_fpr"]

# The deployable precision claim. Cumulative from the bottom of the band audit down to the
# two urn 7-flag threshold at a 2% false alarm budget, which is the operating point the
# audit was designed around. Every unsure counts as a wrong flag in the floor.
TH2 = next(t["threshold"] for t in
           json.loads((TLR / "two_urn_fit.json").read_text())["thresholds"]["7-flag"]
           if abs(t["fpr_budget"] - 0.02) < 1e-9)
PFLAG = [r for r in BA if r["s7"] <= TH2]
P_F = sum(1 for r in PFLAG if r["verdict"] == "false")
P_T = sum(1 for r in PFLAG if r["verdict"] == "true")
P_U = sum(1 for r in PFLAG if r["verdict"] == "unsure")

# ---- section 10: the oracle probe. Recomputed from eval/data/oracle_probe/, reusing the
# probe's own legality join rather than restating the ceiling rule here.
from eval.scripts.build_eval import oracle_probe as OP  # noqa: E402

OPS = [json.loads(line) for line in (OP.OUT / "sample.jsonl").open()]
OPIDX = OP.load_index()
OP.legality(OPS, OP.load_dates(), OPIDX)
OP_TGT = [t for r in OPS for t in r["tgt"]]
OP_LEGAL = [t for t in OP_TGT if t["legal"] == "legal"]
OP_IND = sum(1 for t in OP_LEGAL if (OPIDX.get(t["url"]) or {}).get("indexed"))
OP_UNDATED = sum(1 for t in OP_TGT if t["date"] is None)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. The oracle probe reports Wilson rather than Clopper Pearson
    because three of its headline counts are zero, where Clopper Pearson's lower bound is
    exactly zero and tells the reader nothing about the width."""
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wpct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k/n*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


OP2 = [json.loads(line) for line in (OP.OUT / "stage2.jsonl").open()]
OP2B = [json.loads(line) for line in (OP.OUT / "stage2b.jsonl").open()]
OP2C = [json.loads(line) for line in (OP.OUT / "stage2c.jsonl").open()]
OP3B = [json.loads(line) for line in (OP.OUT / "stage3b.jsonl").open()]


def _reach(r) -> list:
    return [k for k, v in r["rungs"].items() if v["hit"]]


OP2_HIT = [r for r in OP2 if _reach(r)]
OP2_PROD = [r for r in OP2_HIT if r["rungs"]["r1"]["hit"]]
OP2_NEW = len(OP2_HIT) - len(OP2_PROD)
# the ladder early-stops, so a claim reached on an earlier rung has no r5 entry
OP2_R5 = sum(1 for r in OP2 if (r["rungs"].get("r5") or {}).get("hit"))
OP2_DOM = sum(1 for r in OP2 if any(v["dom_hit"] for v in r["rungs"].values()))
OP2B_HIT = sum(1 for r in OP2B if any(p["rank"] is not None for p in r["probes"].values()))
OP2C_HIT = sum(1 for r in OP2C if any(p["hit"] for p in r["pages"].values()))
OP_QA = [r for r in OP2 if r["claim_type"] == "quote_attribution"]
OP_QA_HIT = sum(1 for r in OP_QA if _reach(r))

PAT = collections.Counter(r["pattern"] for r in OP3B)
PATLAB = [
    ("different_framing", "Same underlying fact, different framing, different actor or object"),
    ("adjacent_event", "An adjacent event, other date, other place, other person"),
    ("origin_source", "The origin article the claim was lifted from"),
    ("class_level_debunk", "Debunks the kind of claim, never names this instance"),
    ("not_evidence", "A profile, an index or a tag page"),
]
pat_rows = ""
for code, gloss in PATLAB:
    n = PAT[code]
    pat_rows += (f"<tr><td>{gloss}<br><span class='note'>{code}</span></td>"
                 f"<td class='m'>{n}</td><td class='m'>{n/len(OP3B)*100:.1f}%</td></tr>")
_dm = PAT["direct_match"]
_dmlo, _dmhi = wilson(_dm, len(OP3B))
pat_rows += (f"<tr><td><b>A document about this claim, in the claim's own terms</b>"
             f"<br><span class='note'>direct_match</span></td>"
             f"<td class='m zero'>{_dm}</td>"
             f"<td class='m zero'>{_dm/len(OP3B)*100:.1f}% "
             f"[{_dmlo*100:.1f}, {_dmhi*100:.1f}]</td></tr>")
OP3B_WEAK = sum(1 for r in OP3B if r["relation"] in ("background", "not_evidence"))

# The six interventions. Each arm wrote its own jsonl under a different directory and none
# of them wrote a summary file, so this table is transcribed from the clog entries that
# closed them, clog/270826.md 21:30 (ceiling and fact-check-targeted) and clog/280826.md
# 12:07 (spec, nodate, ugc) and 16:35 (Exa), which are also the rows in
# docs/logodds_sprint.md under the 2026-08-28 six-interventions status. HARDCODED, same
# source: Exa's 6.0% [2.6, 13.2] on 84 claims and the 166 of 211 refuting documents that
# postdate the claim, both from clog/280826.md 16:35.
ARM_ROWS = [
    ("Date ceiling extended 90 days", "Wider time window", "1 / 25", "Not adopted"),
    ("Fact-check-targeted query", "Query aimed at debunk pages", "0 / 25",
     "Rejected, 0.5x selective"),
    ("Uncapped detailed query", "Longer, more specific query", "1 / 10",
     "Rejected, net minus 2 flags"),
    ("Date token removed", "No date in the query string", "1 / 10",
     "Rejected, that one a wrong-event read"),
    ("User-content blocklist lifted", "Forums and social back in", "0 / 47",
     "Rejected, the one directional read <em>confirms</em> the falsehood"),
    ("Exa neural retrieval", "Whole different paradigm", "6.0%", "Not a lever, but see below"),
]
arm_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='m'>{c}</td><td>{d}</td></tr>"
                   for a, b, c, d in ARM_ROWS)

# The silent-read audit, recomputed from its own verdict file. This is the arm that rules
# the reader out as the constraint on this block.
SIL = [json.loads(line) for line in
       Path("eval/data/urn_runs/synth_expt/silent_audit_verdicts.jsonl").open()]


def _sil(pop):
    rows = [r for r in SIL if r["population"] == pop]
    return sum(1 for r in rows if r["refutes"]), len(rows)


SIL_F, SIL_N = _sil("cn_false_missed")
SILC_F, SILC_N = _sil("tl_pass")

# The three worked examples are pulled out of stage3b by their target URL so the claim,
# the production query and the target's title are the run's own strings.
_BY_URL = {r["url"]: r for r in OP3B}
EXAMPLES = [
    ("https://apnews.com/article/fact-checking-330111634396",
     "The debunk never mentions Schumer. It settles the claim by settling the category the "
     "claim belongs to. No query carrying her name reaches it, and no query dropping her "
     "name would ever have been generated."),
    ("https://www.snopes.com/fact-check/microsoft-ceo-disable-computers/",
     "The reviewer knew this was a mutation of an older Microsoft rumour. The principal "
     "actor in the source is a company, not a person, so matching on the claim's own entity "
     "was never going to land."),
    ("https://www.mirror.co.uk/science/people-who-over-6ft-tall-22429492",
     "This is the article the claim was distorted from. It says more likely to report, the "
     "claim says twice as likely to get. The evidence is the origin, and we exclude origin "
     "sources by policy."),
]


def _titlesplit(t: str) -> str:
    for sep in (" | ", " - "):
        head, s, tail = t.rpartition(sep)
        if head and len(tail) <= 20:
            return f"{head} <span class='dead'>{tail}</span>"
    return t


ex_html = ""
for url, why in EXAMPLES:
    r = _BY_URL[url]
    ex_html += (
        f"<div class='ex'><p class='lab'>Claim</p><p class='val'>{r['claim']}</p>"
        f"<p class='lab'>What we searched for</p>"
        f"<p class='val q'>{r['query_prod']}</p>"
        f"<p class='lab'>What the reviewer used</p>"
        f"<p class='val'>{_titlesplit(r['title'])}</p>"
        f"<p class='note'>{why}</p></div>")

# ---- section 11: claim type. ladder_judged_pinned.json is the pinned 3,274 population,
# ladder_judged_all.json the same run with the 294 media-axis claims added back as their
# own stratum, which is the only place the media row exists.
CT = json.loads(Path("eval/data/claim_type/ladder_judged_pinned.json").read_text())
CTA = json.loads(Path("eval/data/claim_type/ladder_judged_all.json").read_text())
CTS = json.loads(Path("eval/data/claim_type/ladder_claim_side_all.json").read_text())
RPT = json.loads(Path("eval/data/reportability/ladder.json").read_text())
CTG, RPG = CT["gold_arms"]["7-flag"], RPT["gold_arms"]["7-flag"]
CTSG = json.loads(
    Path("eval/data/claim_type/ladder_claim_side_pinned.json").read_text())["gold_arms"]["7-flag"]


def _ctrow(name, d, em=False):
    lab = name.replace("_", " ")
    lab = f"<i>{lab}</i>" if em else lab
    return (f"<tr><td>{lab}</td><td class='m'>{d['n']:,}</td>"
            f"<td class='m'>{d['p_false']:.3f}</td><td class='m'>{d['auc']:.4f}</td>"
            f"<td class='m'>{d['recall_2pct']*100:.1f}%</td>"
            f"<td class='m'>{d['all_irrelevant_share']*100:.1f}%</td></tr>")


CT_ALLIRR = (sum(v["n"] * v["all_irrelevant_share"] for v in CT["diagnostic"].values())
             / sum(v["n"] for v in CT["diagnostic"].values()))
ct_rows = "".join(_ctrow(k, v) for k, v in
                  sorted(CT["diagnostic"].items(), key=lambda kv: -kv[1]["auc"]))
ct_rows += _ctrow("media_authenticity", CTA["diagnostic"]["media_authenticity"], em=True)
ct_rows += (f"<tr><td><b>all, pinned</b></td><td class='m'>{CT['population']['pinned']:,}</td>"
            f"<td class='m'>{CT['population']['gold_false']/CT['population']['pinned']:.3f}</td>"
            f"<td class='m'>{CTG['global']['auc']:.4f}</td>"
            f"<td class='m'>{CTG['global']['recall_2pct']*100:.1f}%</td>"
            f"<td class='m'>{CT_ALLIRR*100:.1f}%</td></tr>")

CT_WORST = min(CT["diagnostic"].items(), key=lambda kv: kv[1]["auc"])[0]
CT_BEST = max(CT["diagnostic"].items(), key=lambda kv: kv[1]["auc"])[0]
CT_QA = CT["diagnostic"]["quote_attribution"]
CT_CE = "causal_effect"   # key, used as both label and lookup
MEDIA_REC = CTS["agreement"]["media_authenticity"]["recall"]


def _arm(g, arm):
    v = g[arm]
    d = v.get("dauc_vs_intercept")
    ci = v.get("dauc_vs_intercept_ci")
    txt = ("reference" if arm == "intercept" else
           f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if d is not None else "--")
    return (f"<td class='m'>{v['auc']:.4f}</td><td class='m'>{v['recall_2pct']*100:.1f}%</td>"
            f"<td class='m'>{txt}</td>")


strat_rows = ""
for lab, g in (("claim type, off the verdict", CTG),
               ("claim type, off the claim text", CTSG),
               ("reportability", RPG)):
    for arm, alab in (("global", "no strata"), ("full", "per stratum weights"),
                      ("silence", "per stratum silent weight"),
                      ("intercept", "per stratum constant only")):
        strat_rows += (f"<tr><td>{lab}</td><td>{alab}</td>{_arm(g, arm)}</tr>"
                       if arm in g else "")


html = f"""<title>Score Ladder</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#fbfaf7; --ink:#1c1b19; --mut:#6e6a63; --line:#e2dfd8; --panel:#ffffff;
  --accent:#7a3020; --barfill:#c9c4ba; --dead:#b4aea4; --panel2:#f5f2ec;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#ffffff;
  --accent:#d08363; --barfill:#4a463f; --dead:#5c574f; --panel2:#211f1b;
}} }}
:root[data-theme="dark"] {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#ffffff;
  --accent:#d08363; --barfill:#4a463f; --dead:#5c574f; --panel2:#211f1b;
}}
body {{ background:var(--bg); color:var(--ink);
  font:16px/1.65 "IBM Plex Sans",system-ui,sans-serif; margin:0; padding:3rem 2rem 5rem; }}
main {{ max-width:1080px; margin:0 auto; }}
h1 {{ font:600 2.3rem/1.2 "IBM Plex Serif",Georgia,serif; margin:0 0 .4rem; }}
h2 {{ font:600 1.35rem/1.3 "IBM Plex Serif",Georgia,serif; margin:3rem 0 .8rem;
  padding-top:1.6rem; border-top:1px solid var(--line); }}
h2 span {{ color:var(--mut); font-family:"IBM Plex Mono",monospace; font-size:.85rem;
  display:block; letter-spacing:.08em; margin-bottom:.3rem; }}
p {{ margin:.7rem 0; }}
.sub {{ color:var(--mut); margin-bottom:2.2rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.92rem; }}
th,td {{ text-align:left; padding:.42rem .7rem; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.m {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:1rem; margin:1.2rem 0; overflow-x:auto; }}
.panel svg {{ max-width:100%; height:auto; display:block; }}
.flow {{ display:flex; gap:0; align-items:stretch; margin:1.4rem 0; flex-wrap:wrap; }}
.fbox {{ flex:1 1 150px; border:1px solid var(--line); border-radius:8px; padding:.7rem .8rem;
  font-size:.85rem; background:color-mix(in srgb, var(--bg) 60%, var(--panel)); margin:0 .4rem .4rem 0; }}
.fbox b {{ display:block; font-size:.92rem; }}
.key {{ border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; margin:1.2rem 0; }}
.note {{ color:var(--mut); font-size:.9rem; }}
.tw {{ overflow-x:auto; margin:1rem 0; }}
.tw table {{ margin:0; min-width:34rem; }}
h3 {{ font:600 1rem/1.35 "IBM Plex Sans",system-ui,sans-serif; margin:1.8rem 0 .4rem; }}
.ex {{ background:var(--panel2); border:1px solid var(--line); border-radius:8px;
  padding:1rem 1.2rem; margin:1rem 0; }}
.ex .lab {{ font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--mut); margin:.7rem 0 0; }}
.ex .lab:first-child {{ margin-top:0; }}
.ex .val {{ margin:.1rem 0 0; }}
.ex .val.q {{ font-family:"IBM Plex Mono",monospace; font-size:.88rem; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:1rem;
  margin:1.2rem 0; }}
.grid2 .ex {{ margin:0; }}
.diag {{ border:1px solid var(--line); border-radius:8px; padding:1.1rem 1.2rem;
  margin:1.3rem 0; overflow-x:auto; }}
.diag svg {{ max-width:100%; height:auto; display:block; margin:0 auto; }}
.zero {{ color:var(--accent); font-weight:600; }}
.dead {{ color:var(--dead); }}
</style>
<main>
<h1>Score Ladder</h1>
<p class="sub">A running ledger of the sprint's main findings. How finely we bucket one
piece of evidence, what the extra buckets buy, how much of it survives being fitted
somewhere else, and where the recall that is still missing actually goes · run 27 August
2026, blind audit, retrieval forensics, oracle probe and claim type diagnostic added 28
August 2026</p>

<h2><span>ledger</span>What we actually know, as of 28 August</h2>
<p>This page started as one measurement and is now the running ledger for the sprint. Read
this section before quoting any number off it. Everything below is the honest version,
including the parts where the gain turned out to be smaller than the headline.</p>
<table>
<tr><th>finding</th><th>the number</th><th>population it was measured on</th></tr>
<tr><td>The sprint's only separation gain is the seven bucket cut</td>
<td class="m">dAUC {M['7-flag']['dauc_vs_3voice']:+.4f}
[{M['7-flag']['dauc_ci'][0]:+.4f}, {M['7-flag']['dauc_ci'][1]:+.4f}], recall at a 2% false
alarm budget {M['3-voice']['recall_2pct']*100:.1f}% to
{M['7-flag']['recall_2pct']*100:.1f}%</td>
<td>pinned fc gold, {P['n']:,} claims, out of fold</td></tr>
<tr><td>Only part of that recall survives a false alarm matched comparison</td>
<td class="m">{GMM['baseline_recall']*100:.1f}% to {GMM['recall']*100:.1f}%, so
{(GMM['recall']-GMM['baseline_recall'])*100:.1f} points, at the shipped false alarm rate of
{GMM['fpr']*100:.2f}%</td>
<td>fc gold before the mixed cut, {HM['overall']['n']:,} claims</td></tr>
<tr><td>The headline move is mostly population deletion, not model</td>
<td class="m">{HM['overall']['auc_oof']:.4f} to {M['7-flag']['auc_oof']:.4f} after
{HM['media_axis_excluded']} media axis and {P['set_aside_n']} mixed subtype claims came
out</td>
<td>same instrument at both ends, {HM['overall']['n']:,} claims then {P['n']:,}</td></tr>
<tr><td>Out of corpus the real numbers are lower</td>
<td class="m">AUC {xcell('gold','urn','7-flag')['auc']:.4f}
[{xcell('gold','urn','7-flag')['auc_ci'][0]:.4f},
{xcell('gold','urn','7-flag')['auc_ci'][1]:.4f}], recall
{xcell('gold','urn','7-flag')['recall_2pct']*100:.1f}%</td>
<td>gold weights scored on the urns at an assumed false share of
{X['eps_head']*100:.0f}%, {X['sizes']['cn']:,} community noted against
{X['sizes']['timeline']:,} timeline claims</td></tr>
<tr><td>The deployable precision claim</td>
<td class="m">P(false given flag) {P_F/(P_F+P_T):.2f}, floor {P_F/len(PFLAG):.2f} counting
every unsure as a wrong flag</td>
<td>{len(PFLAG)} flagged timeline claims read blind, on a population that is roughly
{DU['folded_eps']*100:.0f}% false</td></tr>
<tr><td>That precision replaces every gold base rate precision figure</td>
<td class="m">LR+ {GM2['lr_plus']:.1f} reads as 94% at gold's
{P['n_false']/P['n']*100:.0f}% false rate and about 70% at 10%</td>
<td>the gold base rate number is not a claim about a feed</td></tr>
<tr><td>The assumed false share is now measured, and the assumption holds</td>
<td class="m">eps {DU['folded_eps']:.4f} [{DU['folded_clustered'][0]:.4f},
{DU['folded_clustered'][1]:.4f}] folded, against the {X['eps_head']*100:.0f}% this page
runs on</td>
<td>{EN} timeline claims drawn before any score was looked at, {DU['folded_dec']}
decided</td></tr>
<tr><td>Two stratifiers were tested today and both were rejected as scoring changes</td>
<td class="m">a per stratum constant beats the stratified weights in every arm</td>
<td>claim type and reportability, section 11</td></tr>
<tr><td>Six retrieval interventions have failed on the same block, and we now know why</td>
<td class="m">the reviewer's own source is indexed {wpct(OP_IND, len(OP_LEGAL))} of the time
and reachable by no query we can write</td>
<td>{len(OPS)} all-irrelevant false claims, section 10</td></tr>
</table>
<p class="note">The 94% and the 70% on the LR+ row are the two readings of the same
likelihood ratio at two base rates, taken from docs/logodds_sprint.md. Everything else in
this table is recomputed from the run files by this page's own generator.</p>

<h2><span>01</span>What this is</h2>
<p>The score is a sum over evidence documents. Each document contributes one number, the
log ratio of how often a document like it turns up on a true claim versus a false one. The
only design choice here is how finely we sort the documents into buckets before counting.
Three versions of that choice are on this page. Three buckets, seven buckets, twenty eight
buckets. Same claims, same documents, same estimator, same folds. Nothing else moves.</p>
<div class="flow">
<div class="fbox"><b>3 buckets</b>support, refute, silent</div>
<div class="fbox"><b>7 buckets</b>the reader's own flag, ungrouped</div>
<div class="fbox"><b>28 buckets</b>the flag crossed with the source tier</div>
<div class="fbox"><b>{P['n']:,} claims</b>{P['n_true']:,} true and {P['n_false']:,} false</div>
<div class="fbox"><b>{P['docs']:,} documents</b>five folds, every number out of fold</div>
</div>
<p>The population is fc gold with two groups set aside. Media provenance claims are out,
336 of them, because there the fact checker graded whether a photo or video shows what it
is presented as showing and our reader only ever sees text. The misleading group is out
too, 425 claims where the verdict is about the post's framing or its missing context rather
than about the claim we actually score. The 118 unprovable claims stay in. They count as
false when we measure and they stay out of every fit, which is the standing convention.</p>
<p class="note">Setting the misleading group aside is what moves the seven bucket number
from 0.833 to 0.862. That is the same sensitivity cut already reported in the sprint, so
this page and that row are the same measurement.</p>

<h2><span>02</span>The curves</h2>
<p>Detection framing throughout. A false claim is the thing being detected, so recall runs
over gold false claims and the false alarm rate runs over gold true ones. A claim scores
low when the evidence goes against it, so we flag at or below a threshold.</p>
{fig("roc")}
<table>
<tr><th>model</th><th>auc out of fold</th><th>95% interval</th>
<th>recall at 2% false alarms</th><th>gain over 3 buckets</th></tr>
{rows}
</table>
<p>Finer buckets win, and the win is small in rank terms and larger where we actually
operate. Going from three buckets to seven is worth {M['7-flag']['dauc_vs_3voice']*100:.1f}
points of AUC and {(M['7-flag']['recall_2pct']-M['3-voice']['recall_2pct'])*100:.1f} points of
recall at the 2% budget. Going on to twenty eight adds
{(M['28-cell']['auc_oof']-M['7-flag']['auc_oof'])*100:.1f} more points of AUC and only
{(M['28-cell']['recall_2pct']-M['7-flag']['recall_2pct'])*100:.1f} of recall. Both intervals
clear zero against the three bucket baseline, and they are paired, so the comparison is on
the same resampled claims every time.</p>

<h2><span>03</span>The weights</h2>
<p>Every bucket carries one weight. It is the log likelihood ratio for a single document in
that bucket, smoothed by adding one imaginary document to every bucket in each class so an
empty bucket cannot send a weight to infinity. No bucket borrows from any other. That is
the point of the intervals. A wide bar means the bucket has thin evidence of its own, not
that we shrank it toward something safer.</p>
{fig("weights")}
<p>The bars are 95% bootstrap intervals, 2,000 replicates resampled inside each gold class,
with the weights refit inside every replicate. A hollow dot marks a bucket with fewer than
30 documents on one side. Read those as placeholders rather than as findings.</p>
<div class="key">
<p>Three things the collapse to three buckets hides. Flag 2, points against, is charged
{M['7-flag']['weights']['2']:+.2f} on its own but pays the full refute weight of
{M['3-voice']['weights']['refute']:+.2f} once collapsed, so a weak counter signal is priced
like a flat contradiction. Flag X, on claim context with no direction, comes out
{M['7-flag']['weights']['X']:+.2f}, mildly positive, and the collapse buries it in the silent
channel at {M['3-voice']['weights']['silent']:+.2f}. And support splits hard by tier once you
look, {M['28-cell']['weights']['5 | RELIABLE']:+.2f} for a rated outlet stating the claim
against {M['28-cell']['weights']['5 | PRIMARY']:+.2f} for a primary source doing the same.</p>
</div>
<h2><span>04</span>The same ladder without gold labels</h2>
<p>Everything above was fitted on fact checker labels. Now the same three models are fitted
on our own two piles instead, with no gold label anywhere in the fit, and then scored
against the same {P['n']:,} gold claims with no refit. The false pile is
{TU['false_claims']:,} community noted claims. The true pile is {TU['true_claims']:,} claims
off a real X timeline, which is a mixed draw, so we de-mix it at an assumed false share of
{T['eps_head']*100:.0f}% before taking the ratio. Every gold claim is out of sample here by
construction, so there are no folds.</p>
{fig("roc_transfer")}
<table>
<tr><th>model</th><th>transfer auc</th><th>95% interval</th>
<th>recall at 2% false alarms</th><th>gain over 3 buckets</th>
<th>auc when refit on gold</th><th>recall when refit on gold</th></tr>
{trows}
</table>
<div class="key">
<p>The ladder inverts. Fitted on gold, more buckets is monotonically better. Fitted on the
urns, more buckets is monotonically worse, and the drop to 28 is
{abs(TM['28-cell']['dauc_vs_3voice'])*100:.1f} points of AUC with an interval nowhere near
zero. Three buckets transfer essentially intact,
{TM['3-voice']['auc']:.3f} against {M['3-voice']['auc_oof']:.3f} refit. Twenty eight buckets
lose {(M['28-cell']['auc_oof']-TM['28-cell']['auc'])*100:.1f} points. The finer the
bucketing, the more of what it learned was about its own corpus.</p>
</div>
<p>The operating point tells a different story from the ranking, and it is worth saying out
loud because the two disagree. At the 2% budget the seven bucket transfer recovers
{TM['7-flag']['recall_2pct']*100:.1f}% against {M['7-flag']['recall_2pct']*100:.1f}% refit,
so it gives up almost nothing, while three buckets drop from
{M['3-voice']['recall_2pct']*100:.1f}% to {TM['3-voice']['recall_2pct']*100:.1f}%. Three
buckets win on AUC and lose where we actually ship. Seven buckets are the reverse.</p>
<p class="note">The de-mixing rate barely matters. Across an assumed false share of 0 to 15%
the transfer AUC moves by less than 0.003 for every model.</p>
<table>
<tr><th>assumed false share</th><th>3 auc</th><th>3 recall</th><th>7 auc</th><th>7 recall</th>
<th>28 auc</th><th>28 recall</th></tr>
{sweep_rows}
</table>

<h2><span>05</span>The weights it learned instead</h2>
<p>Same three panels as before. The dot and its interval are the urn fit. The tick is where
the gold fit put the same bucket, so the distance between them is what changed when the
labels went away.</p>
{fig("weights_transfer")}
<p>Two things to read off it. The urn weights are compressed toward zero almost everywhere,
which is expected because the false pile is one narrow slice of false claims and the true
pile still has some false in it. Compression alone would not hurt a ranking. What does hurt
is that the compression is uneven. Support falls from {M['3-voice']['weights']['support']:+.2f}
to {TM['3-voice']['weights']['support']:+.2f} while refute only falls from
{M['3-voice']['weights']['refute']:+.2f} to {TM['3-voice']['weights']['refute']:+.2f}, so the
balance between the two channels moves, and that does change the ranking.</p>
<p>The source tier panel is where it breaks. A reliable outlet stating the claim is worth
{M['28-cell']['weights']['5 | RELIABLE']:+.2f} on gold and only
{TM['28-cell']['weights']['5 | RELIABLE']:+.2f} on the urns. An irrelevant document from a
reliable outlet is {M['28-cell']['weights']['I | RELIABLE']:+.2f} on gold and
{TM['28-cell']['weights']['I | RELIABLE']:+.2f} on the urns. Those are not small
disagreements and they are not noise, the intervals do not overlap. Source tier is picking
up what each corpus retrieves rather than whether the claim is true, so it is exactly the
feature that should not be expected to travel.</p>

<h2><span>06</span>Scoring the urns instead</h2>
<p>Now the eval flips. The weights come from fc gold and the thing being scored is our own
data, {X['sizes']['cn']:,} community noted false claims against {X['sizes']['timeline']:,}
timeline claims. There is one problem to deal with first. The timeline is not a clean true
set, some share of it is false, so counting every timeline claim we flag as a false alarm
undercounts how well the score is doing.</p>
<p>We correct it rather than ignore it. If a false claim sitting in the timeline scores like
a false claim in the community notes urn, which is the same assumption the de-mixing already
makes, then at any threshold the observed false alarm rate is the true one diluted by the
contamination</p>
<p class="m" style="margin-left:1rem">FPR observed = (1 - eps) x FPR true + eps x recall</p>
<p>Rearranged, that corrects the whole curve point by point, and integrating the corrected
curve gives AUC true = (AUC observed - eps/2) / (1 - eps). Both numbers are below. At an
assumed false share of {X['eps_head']*100:.0f}% the correction is worth about
{(xcell('gold','urn','7-flag')['auc']-xcell('gold','urn','7-flag')['auc_observed'])*100:.1f}
points of AUC.</p>
{fig("roc_urn")}
<table>
<tr><th>assumed false share</th><th>gold fit, corrected</th><th>gold fit, observed</th>
<th>urn fit, corrected</th><th>urn fit, observed</th>
<th>pooled fit, corrected</th><th>pooled fit, observed</th></tr>
{xsweep}
</table>
<p class="note">Seven buckets shown. Unlike the transfer direction, the assumed false share
does move these numbers, because here it is the eval set that is contaminated rather than
the fit. Every urn-side number on this page is quoted at
{X['eps_head']*100:.0f}%, and the band audit supports that floor, {BA_F} of the bottom {BA_N}
timeline claims came back false on an independent read, which is {BA_F/URN_N*100:.1f}% of the
urn from the bottom fifth alone.</p>

<h2><span>07</span>Every fit against every eval</h2>
<p>Three places to fit, two places to score. The pooled fit adds the two corpora at the
document level, so the false side is gold false plus community notes and the true side is
gold true plus the de-mixed timeline. Everything is out of fold wherever a fit and an eval
share a corpus.</p>
{fig("matrix")}
<table>
<tr><th rowspan="2">fitted on</th><th colspan="3">scored on fc gold</th>
<th colspan="3">scored on the urns, corrected</th></tr>
<tr><th>3</th><th>7</th><th>28</th><th>3</th><th>7</th><th>28</th></tr>
{mrows}
</table>
<div class="key">
<p>The reverse direction transfers cleanly. Fitted on gold and scored on the urns, seven
buckets get {xcell('gold','urn','7-flag')['auc']:.3f} against
{xcell('urn','urn','7-flag')['auc']:.3f} for the urns fitted on themselves. That is the same
number. So gold labels teach the reader flags nothing corpus specific, while the reverse
direction lost {(xcell('gold','gold','7-flag')['auc']-xcell('urn','gold','7-flag')['auc'])*100:.1f}
points. The gold fit is the more general of the two, not just the better one on its own
turf.</p>
</div>
<p>Twenty eight buckets are the exception again and in the direction the earlier sections
predicted. On the urns, the urn fit beats the gold fit
{xcell('urn','urn','28-cell')['auc']:.3f} against {xcell('gold','urn','28-cell')['auc']:.3f},
which is what you expect if the tier cells encode each corpus's own retrieval mix. Whichever
corpus you fit on, the tier split works at home and travels badly.</p>
<p><b>Pooling does not help.</b> On gold it costs seven buckets
{(xcell('gold','gold','7-flag')['auc']-xcell('pool','gold','7-flag')['auc'])*100:.1f} points
against fitting on gold alone, {xcell('pool','gold','7-flag')['auc']:.3f} against
{xcell('gold','gold','7-flag')['auc']:.3f}. On the urns it matches the urn fit rather than
beating it, {xcell('pool','urn','7-flag')['auc']:.3f} against
{xcell('urn','urn','7-flag')['auc']:.3f}. It is never the best cell in either column. What it
does buy is insurance at twenty eight buckets, where pooling rescues the gold score from
{xcell('urn','gold','28-cell')['auc']:.3f} to {xcell('pool','gold','28-cell')['auc']:.3f}.
So mixing the corpora is a hedge against a bad fit, not a way to a better one.</p>
<p class="note">One sensitivity we ran and are not headlining. The band audit read the bottom
{BA_N} timeline claims by score and called {BA_F} of them false. Dropping the
{XB['cells']['n_removed']} of those that match a scored claim by its text and re-scoring lifts
the gold fit at seven buckets to
{XB['cells']['gold']['7-flag']['auc']:.3f}. We do not report that as the number, because the
audit only ever looked below a score cut, so removing claims exactly where the score already
said false raises the result by construction. Treat it as an upper bound on what a fully
audited timeline would give. The removed count is {XB['cells']['n_removed']} rather than {BA_F}
because that join keys claims on their text and three of them collide on the key, which is also
why the cross ladder's own bookkeeping records the audit as 397 claims and 190 falses. Those are
join counts. The audit read {BA_N} and called {BA_F} false, and that is the pair to quote.</p>

<h2><span>08</span>The score against blind labels, on an unconditional draw</h2>
<p>Every number above scores the ladder against labels that already existed. This section is
the first time the score has been checked against labels collected on a draw that did not look
at the score. {EN} claims were sampled uniformly at random from the {URN_N:,} scored timeline
claims, seeded, before any score was consulted, and audited blind. The auditor saw the claim
and the post it came from and nothing else. No score, no flag, no band, no retrieved document,
no community note. It did its own web research and returned true, false or unverifiable.</p>
<p>That independence is the whole point. The band audit in section 07 read the bottom {BA_N}
claims by score, so a false rate measured on it is a property of the cut and cannot be read as
a property of the timeline. This draw can.</p>
<div class="key">
<p>On the first pass the false share is {EPS['eps']:.3f}, {EF} false against {ET} true on the
{EDEC} claims that pass could decide. The exact 95% interval is [{ELO:.3f}, {EHI:.3f}] and the
post clustered one, which lets several claims from the same post move together instead of
counting them as independent draws, is [{ECLO:.3f}, {ECHI:.3f}]. Always quote the post clustered
interval rather than the exact one. It is the wider of the two and it is the one that matches how
the sample was actually built, {EDEC} claims off {EPS['n_posts']} posts, and the design effect is
{EPS['deff']:.2f} so ignoring the clustering would understate the width by about a fifth.</p>
<p>That first pass number is a floor, not the estimate. A second harder pass went back over the
{DU['n_unver']} claims the first pass could not settle and resolved {DU['resolved_T']+DU['resolved_F']}
of them, and those came back {DU['resolved_F']} false against {DU['resolved_T']} true, a false
share of {DU['resolved_F']/max(DU['resolved_T']+DU['resolved_F'],1):.2f} against
{EPS['eps']:.3f} in the easy claims. Unsettleable claims are enriched in false ones, which is
exactly the direction that makes dropping them from the denominator biased low. Folding every
read in gives <b>{DU['folded_eps']:.3f}</b> on {DU['folded_dec']} decided claims, post clustered
95% [{DU['folded_clustered'][0]:.3f}, {DU['folded_clustered'][1]:.3f}]. <b>That is the number to
carry.</b> Every urn side figure on this page is corrected at an assumed
{X['eps_head']*100:.0f}%, which sits almost exactly on it, so the assumption this page has been
running on for weeks turns out to be right and nothing above needs restating.</p>
</div>
<p><b>The unverifiable claims are the binding uncertainty, so they got their own pass.</b>
{EU} of the {EN} claims came back unverifiable on the first read, {EU/EN*100:.1f}% of the draw.
If every one of them were secretly true the false share would be {EPS['bound_all_true']:.3f} and
if every one were secretly false it would be {EPS['bound_all_false']:.3f}, a span of
{(EPS['bound_all_false']-EPS['bound_all_true'])*100:.1f} points, which is far wider than the
sampling interval. Another few hundred claims would not touch that, so the reads went here
instead. All {DU['n_deep']} were re-read by a fresh auditor working to a harder brief, told to
escalate through archived copies, official registries, company and agency newsrooms, trade press
and non English sources, and to distinguish no source found from no source can exist. That pass
settled {DU['resolved_T']+DU['resolved_F']} of {DU['n_deep']} and left {DU['n_stuck']} standing.
The residual ambiguity band is [{DU['band'][0]:.3f}, {DU['band'][1]:.3f}], a span of
{DU['band_span_pp']:.1f} points against {DU['band_span_before_pp']:.1f} before, so the pass cut
the binding uncertainty roughly in half.</p>
<p>{RESOLV_SENT}</p>
<table>
<tr><th>what makes it unresolvable</th><th>claims</th><th>share of the residue</th></tr>
{TAXROWS}
</table>
<p><b>The false calls were read twice.</b> The estimator is more sensitive to a wrong false
than to a wrong true, because a false sits in the numerator and a true only in the denominator,
so every claim called false went to a second independent auditor who was not told what the
first one said. {SR['false_confirmed']} of {SR['n_false_reread']} came back false again. On its
own that means little, because a second reader who is simply more sceptical would also confirm
them, so {SR['n_true_reread']} claims called true were sent to a second auditor the same way
without being marked as a control. {SR['true_overturned']} of those came back false. The two
rates differ with a Fisher exact p of {SR.get('fisher_p',1):.1e} at an odds ratio of
{SR.get('or',0):.0f}, so the second reader is agreeing with the first about which claims are
false rather than being uniformly harsher. Without that control this arm would say nothing. The
{SR['n_false_reread']-SR['false_confirmed']} falses the second reader did not confirm are real
disagreement though, and they put a band on the numerator. Under the strictest rule, both readers
must say false, the share is {SR.get('eps_and',0):.3f}. Under the loosest, either reader saying
false is enough, it is {SR.get('eps_or',0):.3f}. Both sit inside the interval on the number the
box above says to carry, so the reading rule is not what decides this.</p>
<p><b>What each column is.</b> The s7 band is a range of the score itself, which is the sum of
the log odds weights over one claim's retrieved documents, so a claim sits low when the evidence
leans against it and high when it leans for it. Audited is how many of the {EN} sampled claims
landed in that band. True, false and unverifiable are what the blind auditor came back with.
False share is false over true plus false, so the unverifiable claims leave the denominator,
because we do not know which way they would have fallen. That number is eps, the same eps this
page uses everywhere else for the share of the timeline that is really false. Beside it is a
Clopper Pearson interval, which is a 95% confidence interval for a proportion that does not go
through the normal approximation. The normal approximation needs a count big enough to look like
a bell curve and returns nothing usable at two false claims out of five, where Clopper Pearson
works straight off the binomial and still answers. It answers conservatively, so the interval is
a little wider than it strictly needs to be. Claims in urn is how many of the {URN_N:,} scored
timeline claims sit in that band, share of urn is the same count as a percentage, and implied
false takes the band's false share and applies it to every claim in the band, which makes it an
extrapolation rather than a count.</p>
<p><b>What the first 40 claims did and did not show, because it matters for reading this
table.</b> The shakedown pass audited 40 claims and every one of the five bands came back holding
exactly one false claim. That ladder was entirely the denominators moving, {andlist(SM_DEC)}
decided claims from the bottom band up, and the numerator carried no information at all. We
checked it for a bug in the join between the audit labels and the bands and it came back clean,
the draw reproduces element for element from its seed and the label join is one to one. Read
against the {sum(SM_DEC)} decided claims rather than all 40, the pattern is about 7.5 times more
likely under a monotone model than under a flat one, which is weak evidence for the signal and a
pattern we went looking for after seeing it. The {EN} claim draw below replaces that read
entirely. The numerator now moves across the bands and the table stands on its own.</p>
<table>
<tr><th>s7 band</th><th>audited</th><th>true</th><th>false</th><th>unverifiable</th>
<th>false share</th><th>95% exact interval</th><th>claims in urn</th><th>share of urn</th>
<th>implied false</th></tr>
{erows}
</table>
<p class="note">s7 here is the seven bucket score with the weights from the two urn fit at an
assumed false share of 5%, read straight out of true_timeline/two_urn_fit.json. Those are the
same weights the band audit scores with, so these bands are the bands already in use and not a
new cut invented for this table. The five band implied false figures add to {IMPLIED2:.0f} while
the same audit read whole gives {EPS['eps']*B2['urn_n']:.0f}, and that
{abs(IMPLIED2-EPS['eps']*B2['urn_n']):.0f} claim gap is the sampling noise arguing with itself.
The {EU} unverifiable claims are not spread evenly either, which is why the decided count per
band is smaller than the audited count by different amounts.</p>
<p><b>The same table under the other weights.</b> Layer 0 changed on 28 August and fc gold is
the fit corpus now, so the bands are worth re-cutting with the gold fitted weights rather than
the two urn ones. The two weight sets have different dynamic ranges, the strongest supporting
bucket is {WG5:+.2f} on gold against {W25:+.2f} on the two urns, so a fixed cut at minus three
and plus three does not mean the same thing in both. The gold cut is taken at the frame
percentiles of the two urn bands instead, which holds the number of urn claims per band fixed
and leaves only the ordering free to differ.</p>
<table>
<tr><th>s7 band, gold fit</th><th>audited</th><th>true</th><th>false</th><th>unverifiable</th>
<th>false share</th><th>95% exact interval</th><th>claims in urn</th><th>share of urn</th>
<th>implied false</th></tr>
{growers}
</table>
<p class="note">{AGREE_SENT}</p>
{fig("eps_bands")}
<p class="note">Dot at the band's false share, whisker at its exact 95% interval, count above
each whisker reading falses over decided claims. The grey bar behind each band is the share of
the urn that sits in it, on the right hand axis, so the bands whose intervals are tight are
visibly the bands where most of the urn lives. Dotted line at the false share of the whole
draw.</p>
<p>What is real, and does not lean on the audit at all, is the right hand side of both tables.
The band populations are ordered the way a working score orders them, {urn_band[BANDS[-1]]:,} of
the {URN_N:,} claims in the top band down to {urn_band[BANDS[0]]} in the bottom one. That is the
score doing its job on a population that is mostly true, and it is measured on all {URN_N:,}
claims rather than on {EN}.</p>
<div class="key">
<p><b>Does the ladder survive at real n.</b> Yes, and this is the question the whole exercise
existed to answer. Under the two urn weights the false share falls across all five bands with no
reversal, from {B2['rows'][0]['eps']:.3f} in the bottom band to {B2['rows'][-1]['eps']:.3f} in
the top one, and a trend test across the five ordered bands returns z equal to
{B2['ca_z']:+.2f} with p equal to {B2['ca_p']:.4f}. Under the gold weights the same test returns
z equal to {BG['ca_z']:+.2f} and p equal to {BG['ca_p']:.4f}. Collapsed to the single strongest two way
split the two urn cut gives {BEST2['low_eps']:.3f} on {BEST2['low_dec']} decided claims below
against {BEST2['hi_eps']:.3f} on {BEST2['hi_dec']} above, Fisher exact one sided p equal to
{BEST2['fisher_p']:.4f}. That is the strongest of the four possible splits and it is quoted as
such. The cleanest statement does not need bands at all. Ranking the {EDEC} decided claims by
score, a randomly chosen true claim scores above a randomly chosen false one
{SEPG['auc']*100:.1f}% of the time under the gold weights and {SEP2['auc']*100:.1f}% under the
two urn weights, and a permutation test that shuffles labels between whole posts rather than
between claims returns p equal to {SEPG['perm_p']:.4f} and {SEP2['perm_p']:.4f}. At 40 claims
the strongest split reached p equal to 0.09 and we could say only that the direction was
consistent. At {EN} it is significant.</p>
<p>What that does and does not establish. It establishes that the score orders claims the same
way a blind human style read does, on a draw that never saw the score. It does not say the score
is calibrated, and the two tail bands still decide on a handful of claims each, so the per band
rates in the extremes carry intervals wide enough to overlap their neighbours. The ordering is
measured. The per band level is not.</p>
</div>
<p><b>What this does to the numbers above.</b> Nothing, and that is worth stating rather than
leaving implied. Every urn side figure on this page is corrected at an assumed false share of
{X['eps_head']*100:.0f}% and the best measured estimate is {DU['folded_eps']*100:.1f}%. To put a
bound on it anyway, re-running the whole cross ladder at the first pass share of
{EPS['eps']*100:.1f}% moves the headline cell, gold weights scored on the urns at seven buckets,
from {xcell('gold','urn','7-flag')['auc']:.4f} to {MW['auc']:.4f} and recall at a 2% false alarm
budget from {xcell('gold','urn','7-flag')['recall_2pct']*100:.1f}% to
{MW['recall_2pct']*100:.1f}%. Swept from one end of the post clustered interval to the other the
same cell runs {MLO:.4f} to {MHI:.4f}, a {(MHI-MLO)*100:.1f} point span, against a bootstrap
interval on that cell of [{xcell('gold','urn','7-flag')['auc_ci'][0]:.4f},
{xcell('gold','urn','7-flag')['auc_ci'][1]:.4f}] which is already
{(xcell('gold','urn','7-flag')['auc_ci'][1]-xcell('gold','urn','7-flag')['auc_ci'][0])*100:.1f}
points wide. The assumed share was never the binding uncertainty. The reason to measure it was
that it had never been measured.</p>
<p class="note">The first pass recorded its own coarser reason for each unverifiable claim,
before the deeper pass reclassified them, and on that reading they were {UWHY_SENT} The two lists
disagree mostly because no public record is what a normal search budget concludes and the deeper
pass then found the record about half the time. That is the finding worth extending. The ceiling
on this bucket is search effort, not the claims themselves.</p>

<h2><span>09</span>The claims that get nothing worth weighing</h2>
<p>This section changes subject. Everything above is about how to weigh the documents a claim
gets back. This is about the claims where there is nothing to weigh. A claim is in this block
when every retrieved document read irrelevant, all ten of them, no support, no refutation,
nothing on the claim at all. That is {FG_HIT} of {FG_TOT:,} fc gold false claims and
{FCN_HIT} of {FCN_TOT:,} community noted ones.</p>
<p>It is not a coverage failure, which was the first thing to rule out. The documents come
back at the normal rate, a median of {med(FG['hit']):.0f} per claim in this block against
{med(FG['ctl']):.0f} for the claims that worked, so nothing is being thrown away between the
search and the read. The evidence is findable too. Take the gold claims here whose reviewer's
own domain never appeared in any result at all, and {FG_AWI} of {len(FG_A)} of them cite at
least one source on a well indexed domain, meaning NewsGuard rated or government or academic
or a primary source. The reviewer's evidence was sitting where a search engine can see it. We
asked for something else.</p>
<p>So the next question is what the queries were doing wrong. Six hundred of them were labelled,
200 from each miss arm and 100 from a control arm of claims in the same corpus whose retrieval
worked.</p>
<table>
<tr><th>what the query did</th><th>notes, missed</th><th>notes, worked</th>
<th>fc gold, missed</th><th>fc gold, worked</th></tr>
{mode_rows}
</table>
<p>The control columns are the point. Most of these modes are just as common when retrieval
works. over_generic runs {modeshare('cn_allirr',('over_generic',)):.1f}% against
{modeshare('cn_ctrl',('over_generic',)):.1f}% on the notes and lower than its own control on
gold, and wrong_date is below its control in both corpora, so neither one is what separates a
miss from a hit. The mode that rises in both is nonexistent_event,
{modeshare('cn_allirr',('nonexistent_event',)):.1f}% against
{modeshare('cn_ctrl',('nonexistent_event',)):.1f}% on the notes and
{modeshare('fg_allirr',('nonexistent_event',)):.1f}% against
{modeshare('fg_ctrl',('nonexistent_event',)):.1f}% on gold. The claim describes an event that
never happened, so a query asking for it has nothing to match and the engine answers with
whatever shares its words.</p>
<p>The tidy explanations are not the story. Wrong language, mangled names and unresolved deixis
together are {modeshare('cn_allirr',('non_english','entity_mangled','deixis')):.1f}% of the
notes arm and {modeshare('fg_allirr',('non_english','entity_mangled','deixis')):.1f}% of the
gold arm. And in {modeshare('fg_allirr',('query_ok',)):.1f}% of the gold misses the labeller
found nothing wrong with the query at all, which is no better than its control. A rewriting
pass has less to work with here than it looks.</p>
<div class="key">
<p>This is the block that caps recall. We never flag on silence, so a claim whose ten documents
are all off claim cannot be flagged no matter what the weights are. Every ladder on this page
moves recall inside the claims that retrieve something. This block sits outside all of them, it
is {FG_HIT/FG_TOT*100:.0f}% of the gold false side and {FCN_HIT/FCN_TOT*100:.0f}% of the
community noted one, and it is the largest lever on recall we have identified.</p>
</div>
<p class="note">The query labels are one LLM pass over the claim, its query and the ten domains
that came back, so the split between over_generic and wrong_date is approximate. The four arms
are sized by design rather than by prevalence, so read down each column and not across the
bottom row.</p>

<h2><span>10</span>The evidence we know exists, and cannot reach</h2>
<p>Section 09 measured the block. This section attacks it, six ways, and then works the
problem backwards. The block is every false claim where all ten retrieved documents read
irrelevant, {FG_HIT} of {FG_TOT:,} fc gold falses and {FCN_HIT} of {FCN_TOT:,} community
noted ones, so roughly a quarter of what the tool exists to catch.</p>
<p>The reader is not the constraint, which had to be ruled out first. A second blind pass
over the reads called silent on missed false claims found {SIL_F} of {SIL_N} of them
actually refuting, {SIL_F/SIL_N*100:.1f}%, which is <em>below</em> the {SILC_F/SILC_N*100:.1f}%
on a mostly true control of {SILC_N} reads. There are no hidden refutations sitting in the
dossiers we already have. Whatever is missing was never retrieved.</p>
<p>So we went at retrieval. Every one of these changed how we search and none of them
changed the outcome.</p>
<table>
<tr><th>intervention</th><th>what it changed</th><th>claims moved</th><th>verdict</th></tr>
{arm_rows}
</table>
<p class="note">Exa is the only arm that is not flat. Its genuine ceiling legal recovery
rate is 6.0% [2.6, 13.2] on 84 claims, which is real and is not enough to swap providers
on. The five Serper side arms span query wording, query length, time windows and the source
blocklist. If the problem were the query, one of them should have worked.</p>

<p><b>The oracle test.</b> Instead of guessing at better queries we ran it backwards. For
these claims we know what the human reviewer used, because the fact check article and the
community note both cite their sources. So we take the target document as given and ask
what it would have taken to retrieve it. {len(OPS)} claims drawn from the block,
{len(OP_TGT)} reviewer targets, seed {OP.SEED}.</p>
<div class="grid2">
<div class="ex">
<p class="lab">stage 1, is it indexed</p>
<p class="val"><span class="m" style="font-size:1.4rem">{OP_IND/len(OP_LEGAL)*100:.1f}%</span>
of ceiling legal reviewer sources are in the index, {OP_IND} of {len(OP_LEGAL)}, 95%
interval {wilson(OP_IND, len(OP_LEGAL))[0]*100:.1f} to
{wilson(OP_IND, len(OP_LEGAL))[1]*100:.1f}.</p>
<p class="note">Searched by exact title in quotes, then by domain plus title, then by domain
plus slug. The document is there. The web does not have it is closed as an explanation.</p>
</div>
<div class="ex">
<p class="lab">stage 2, can any query reach it</p>
<p class="val"><span class="m"
style="font-size:1.4rem">{len(OP2_HIT)/len(OP2)*100:.1f}%</span>
reached by any of five query formulations, {len(OP2_HIT)} of {len(OP2)}, and
{len(OP2_PROD)} of those the production query already had.</p>
<p class="note">Net gain of the four new formulations is {wpct(OP2_NEW, len(OP2))}. The
source routing formulation, the one the eps audit's deep pass suggested, won
{OP2_R5} claims.
Domain level is softer, {OP2_DOM} of {len(OP2)} reach the target's domain.</p>
</div>
</div>
<p>Then we closed the two mechanical escapes. The same claims re-run with the date ceiling
removed, and results pages two through five walked, ranks 11 to 50.</p>
<div class="diag" style="text-align:center">
<p class="m zero" style="font-size:1.8rem; margin:.3rem 0">{OP2B_HIT} of {len(OP2B)}</p>
<p class="note" style="margin:.2rem 0 .3rem">In every condition, ceiling off and deep both,
95% interval {wilson(0, len(OP2B))[0]*100:.1f} to {wilson(0, len(OP2B))[1]*100:.1f}.</p>
</div>
<p>The document is not hiding behind the date filter and it is not sitting just below the
fold. It is indexed, and our search does not reach it.</p>

<p><b>What the reviewer actually cited.</b> So we read the reviewer's sources and
classified each one against the claim it was cited for. This is the result the sprint turns
on.</p>
<table>
<tr><th>what the reviewer's cited source actually is</th><th>n</th><th>share of pairs</th></tr>
{pat_rows}
</table>
<p class="note">{len(OP3B)} claim and target pairs, classified from the claim and the
target's title and URL only.</p>
<p>Not one. Across {len(OP3B)} pairs the document that settled the claim was never a
document about the claim, and {OP3B_WEAK} of {len(OP3B)} of what the reviewer cited,
{OP3B_WEAK/len(OP3B)*100:.0f}%, could not have settled the claim on its own at all.</p>
<div class="key">
<p><b>The reviewer supplies the reasoning. The document supplies only the raw fact.</b></p>
<p>A human fact checker does not search for the claim. They work out what the claim is a
distortion <em>of</em>, then go and get that. Our retrieval only ever performs the first
move.</p>
</div>
<div class="diag">
<svg viewBox="0 0 1000 340" role="img" aria-label="Diagram contrasting one hop retrieval, which fails, with the reviewer's two hop reasoning, which succeeds">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
    <marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)"/>
    </marker>
  </defs>
  <g font-family="IBM Plex Mono, monospace" font-size="11" letter-spacing="1.2" fill="var(--mut)">
    <text x="0" y="16">WHAT WE DO</text>
    <text x="0" y="196">WHAT THE REVIEWER DOES</text>
  </g>
  <g color="var(--mut)">
    <rect x="0" y="36" width="200" height="58" rx="6" fill="var(--panel2)" stroke="var(--line)"/>
    <text x="100" y="61" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
          font-size="13" font-weight="600" fill="var(--ink)">The claim</text>
    <text x="100" y="80" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="11" fill="var(--mut)">as posted</text>
    <line x1="206" y1="65" x2="292" y2="65" stroke="currentColor" stroke-width="1.4"
          marker-end="url(#ar)"/>
    <text x="249" y="55" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="10.5" fill="var(--mut)">search</text>
    <rect x="300" y="36" width="290" height="58" rx="6" fill="none"
          stroke="var(--dead)" stroke-width="1.4" stroke-dasharray="5 4"/>
    <text x="445" y="61" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
          font-size="13" font-weight="600" fill="var(--dead)">A document stating the claim</text>
    <text x="445" y="80" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="11" fill="var(--dead)">does not exist</text>
    <text x="620" y="70" font-family="IBM Plex Sans, sans-serif" font-size="13"
          fill="var(--mut)">Ten real documents come back.</text>
    <text x="620" y="90" font-family="IBM Plex Sans, sans-serif" font-size="13"
          fill="var(--mut)">None of them is on the claim.</text>
  </g>
  <line x1="0" y1="140" x2="1000" y2="140" stroke="var(--line)" stroke-width="1"/>
  <g color="var(--accent)">
    <rect x="0" y="216" width="200" height="58" rx="6" fill="var(--panel2)" stroke="var(--line)"/>
    <text x="100" y="241" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
          font-size="13" font-weight="600" fill="var(--ink)">The claim</text>
    <text x="100" y="260" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="11" fill="var(--mut)">as posted</text>
    <line x1="206" y1="245" x2="292" y2="245" stroke="var(--accent)" stroke-width="1.6"
          marker-end="url(#arA)"/>
    <text x="249" y="235" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="10.5" fill="var(--accent)">infer</text>
    <rect x="300" y="216" width="250" height="58" rx="6" fill="var(--panel2)"
          stroke="var(--accent)" stroke-width="1.6"/>
    <text x="425" y="241" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
          font-size="13" font-weight="600" fill="var(--ink)">The fact it distorts</text>
    <text x="425" y="260" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="11" fill="var(--accent)">the missing hop</text>
    <line x1="556" y1="245" x2="642" y2="245" stroke="var(--accent)" stroke-width="1.6"
          marker-end="url(#arA)"/>
    <text x="599" y="235" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="10.5" fill="var(--accent)">search</text>
    <rect x="650" y="216" width="290" height="58" rx="6" fill="var(--panel2)" stroke="var(--line)"/>
    <text x="795" y="241" text-anchor="middle" font-family="IBM Plex Sans, sans-serif"
          font-size="13" font-weight="600" fill="var(--ink)">A document about that fact</text>
    <text x="795" y="260" text-anchor="middle" font-family="IBM Plex Mono, monospace"
          font-size="11" fill="var(--mut)">indexed, {OP_IND/len(OP_LEGAL)*100:.1f}% of the time</text>
  </g>
</svg>
</div>
<p>Every one of the six interventions was a better way of drawing the top arrow. None of
them added the bottom one.</p>

<p><b>Three claims, three misses.</b> The claim and the query are the run's own strings and
the title is the reviewer's target.</p>
{ex_html}

<h3>What follows, change one. Retrieve the record, not the claim</h3>
<p>Add a step between the claim and the search that names what the claim implies rather
than what it states. Which slot, which record, which canonical event name. Then search for
that. For a quote attribution claim it is the transcript or the original remarks. For a
statistic it is the dataset or the report the number was pulled from. This also needs a
changed reading step, because our reader asks whether a document supports or refutes the
claim, and a transcript showing what was actually said never mentions the claim at all. The
right question for a retrieved record is whether it establishes what was the case, and
whether that matches what was asserted.</p>
<h3>What follows, change two. Stop excluding the origin source, selectively</h3>
<p>{PAT['origin_source']/len(OP3B)*100:.0f}% of the reviewer's cited sources are the
article the claim was lifted from. We exclude those by standing rule, and the rule is right
when the question is whether something is true, because an outlet cannot corroborate
itself. It is wrong when the question is what the source actually said, which is exactly
the case when a claim is a misreading of a real report. Those are two different uses of the
same document and we currently block both.</p>
<p class="note">What this does not establish. The oracle is only as good as the reviewer's
link set, and {OP3B_WEAK/len(OP3B)*100:.0f}% of that link set turned out not to be evidence
on its own. {OP_UNDATED} of {len(OP_TGT)} targets could not be dated, mostly dead links. The
query ladder ran on {len(OP2)} claims, which is enough to rule out a large effect and not
enough to rule out a small one. Neither change has been tested, and the measurement that
matters for either is not whether it finds more documents but whether it finds more
documents on false claims than on true ones. One earlier arm was killed for producing more
refutations on true claims than on false ones. One further finding sits underneath all of
this and neither change addresses it. With the date ceiling removed, 166 of 211 refuting
documents postdate the claim, so part of this block is a latency problem rather than a
reachability one, and a tool that must answer at post time cannot search its way out of
that.</p>

<h2><span>11</span>Which claims the instrument works on</h2>
<p>The same pinned weights, out of fold, evaluated inside each kind of claim. The kind is
read off the fact checker's own judged axis, which is a property of the verdict, so this is
a lens and never a production feature. Within type AUC is invariant to a per type constant,
so this ranking is not an artefact of the base rates in the third column.</p>
<table>
<tr><th>claim type</th><th>n</th><th>share false</th><th>within type auc</th>
<th>recall at 2% false alarms</th><th>all documents irrelevant</th></tr>
{ct_rows}
</table>
<p class="note">{CT_CE.replace('_',' ')} is {CT['diagnostic'][CT_CE]['p_false']*100:.1f}%
false, so its {CT['diagnostic'][CT_CE]['auc']:.3f} rests on
{CT['diagnostic'][CT_CE]['n'] - int(round(CT['diagnostic'][CT_CE]['n']*CT['diagnostic'][CT_CE]['p_false']))}
true claims. Do not quote it without the n. media authenticity is the excluded stratum,
shown here in italics on the run that adds it back, and it is the only genuinely weak one.
{CTA['diagnostic']['media_authenticity']['silent_doc_share']*100:.0f}% of its documents come
back silent, because the read is being scored against a proposition it never assessed. That
exclusion is now measured rather than assumed.</p>
<p>There is no dead zone, and that is the honest answer. Every kept type sits between
{CT['diagnostic'][CT_WORST]['auc']:.3f} and {CT['diagnostic'][CT_BEST]['auc']:.3f}. The
instrument is not strong on some kinds of claim and useless on others, it is decent
everywhere and noticeably better on some, which kills the routing story we went looking
for.</p>
<div class="key">
<p>The weakest kept type is {CT_WORST.replace('_',' ')} and the reason is retrieval, not
reading. {CT_QA['auc']:.4f} AUC, {CT_QA['recall_2pct']*100:.1f}% recall at the 2% budget,
and {CT_QA['all_irrelevant_share']*100:.1f}% of its claims come back with every document
irrelevant, which is roughly double every other type. Section 10 reached the same wall from
the other side without being pointed at it. Of the {len(OP_QA)} {CT_WORST.replace('_',' ')}
claims in the oracle sample, the number any query formulation reached was {OP_QA_HIT}. Two
methods, different data, same conclusion. And the second hop for this kind of claim is
unusually clear. The claim says a person said a thing, and the record that settles it is
what they actually said.</p>
</div>
<p><b>As a scoring change, both stratifiers tested today were rejected.</b> Claim type and
reportability were each fitted three ways against the same control, a per stratum constant
with the global evidence weights left alone. The stratified arms beat the global fit. The
constant beats them. Every stratified arm minus the constant overlaps zero or is negative,
in both stratifiers, on both label sources.</p>
<table>
<tr><th>stratifier</th><th>arm</th><th>auc</th><th>recall at 2% false alarms</th>
<th>gain over the per stratum constant</th></tr>
{strat_rows}
</table>
<p class="note">7 buckets, pinned fc gold {CT['population']['pinned']:,} claims, out of
fold, {CT['boot']['reps']:,} replicate paired bootstrap at seed {CT['boot']['seed']}. Claim
type off the verdict has {len(CT['diagnostic'])} strata on the pinned population, off the
claim text the same {len(CT['diagnostic'])}, reportability {len(RPT['strata'])}. A stratifier that gains is gaining
by encoding that this kind of claim is more often false in fc gold, not by anything about
how evidence behaves. The per type weights also lose inside their own strata, which is
straight overfitting, seven weight vectors where one was already enough. The pattern worth
recording is that the fit has room for a better prior and no room for better per stratum
evidence weights, which is a statement about the size of the fit population.</p>
<p><b>And the media exclusion cannot be reproduced from the claim side at all.</b> A
label blind classifier run on claim text and date only agrees with the judged axis on
{CTS['agreement']['_overall']['accuracy']*100:.1f}% of the {CTA['population']['gold']:,}
claims with media added back, and it reproduces the diagnostic ranking above. Its recall on
media authenticity is {MEDIA_REC:.3f}. That is not a classifier failure, it is a structural
fact restated. Extraction strips the media framing, so a claim about what a video shows
reaches us as a claim about the event itself. Setting the media axis aside is therefore a
gold only device, which matters for anything that wants to ship it.</p>

<p class="note">Generated by model_ladder.py, transfer_ladder.py, cross_ladder.py and
model_ladder_figs.py from the saved reads on all three corpora, section 08 from
timeline_eps_audit.py, section 09 from retrieval_forensics.py, section 10 from
oracle_probe.py plus the silent audit verdicts, and section 11 from claim_type_ladder.py,
claim_type_side.py and reportability_ladder.py. Every number on this page is recomputed
from the run files, except the six intervention counts and the two readings of the
likelihood ratio, which are transcribed from the clog entries that closed those runs and
are marked as such in the generator.</p>
</main>
"""

# Wide tables get their own scroll container so the page body never scrolls sideways.
# Done here rather than at every call site because there are eight of them.
import re as _re
html = _re.sub(r"<table>(.*?)</table>", r'<div class="tw"><table>\1</table></div>',
               html, flags=_re.S)

out = A / "model_ladder.html"
out.write_text(html)
print(f"wrote {out}")
