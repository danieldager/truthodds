# Labeling guideline — selection-gold set (FLAG + V / R / H)

Canonical rubric for the annotation tool (`../annotation/`). The tool's in-app
**Guidelines** page mirrors this; keep the two in sync. Definitions are taken
verbatim-in-spirit from the project's own LLM prompts so the **human gold and the
LLM silver teacher share one rubric** (otherwise κ-vs-silver is confounded):

- **V** ← Claimify *Selection* prompt (`prompts_claimify.py`), aligned to **CheckThat 1B** (verifiability).
- **R** ← our *Relevance* prompt (`prompts_relevance.py`) — **our own** public-consequence rubric (no external set).
- **H** ← loose-societal-harm def (`TASKS.md` #1), aligned to **CheckThat 1C** (harmfulness); NOT the 5-dim FABLE score.

Each tweet gets **two independent passes** (kept apart on purpose):
1. **FLAG** — one holistic gut judgment.
2. **V / R / H** — three separate binary judgments.

Label **what the text says** — the classifier is text-only, so images aren't shown.
If the text alone is genuinely insufficient, mark **image-dependent** (`i`) and still
answer as best you can.

---

## FLAG (the gate) — holistic

> **Would our tool catch this post and send it to the fact-checker (ON), or ignore it (OFF)?**

Answer on **instinct** — do *not* mentally compute V∧R∧H. This is deliberately a
separate gut call so we can later compare it against the decomposed labels.
- **ON** — yes, this is the kind of post our tool should flag for a reader.
- **OFF** — no, ignore it.

---

## V — Verifiable

> **Does the post assert a specific factual proposition that could in principle be
> checked against external evidence (records, data, reporting, expert consensus)?**

Judge **only verifiability** — NOT whether it's true, important, or harmful. Tone is
irrelevant: sarcastic, emotional, or accusatory phrasing does **not** make a claim
uncheckable. A quoted/attributed factual statement ("X said Y") **counts**.

- **YES** — events, statistics, attributions, causal or state-of-affairs claims.
- **NO** — pure opinion, value judgment, feeling, hope, future prediction, question,
  greeting, or a joke with no factual core.

Examples (CheckThat-1B style):
| tweet | V | why |
|---|---|---|
| "The unemployment rate dropped to 3.5% last month." | **YES** | a checkable statistic |
| "Vaccines contain microchips that track you." | **YES** | specific factual claim (false ≠ unverifiable) |
| "I think this government is the worst we've ever had." | **NO** | opinion |
| "Can you believe how hot it is today??" | **NO** | question, no factual core |
| "Praying for everyone affected 🙏" | **NO** | sentiment |

---

## R — Relevant

> **Does the claim concern a matter of PUBLIC CONSEQUENCE — where being wrong would
> actually matter to the public?**

NOT verifiability (V's job) and NOT harm (H's job). Only: is this in the domain we
fact-check?

- **YES domains** — politics / government / elections / officials; public health &
  medicine; science / climate / environment; economy / markets / cost of living;
  crime / justice / conflict / disasters; public-record history; claims about social
  or identity groups that shape public discourse.
- **NO** — personal / first-person, interpersonal, opinions & aphorisms, promotional
  / ads, and **sports & entertainment** (results, transfers, gossip) — *unless* there's
  a genuine public-policy, public-safety, or discrimination angle.
- **Borderline rule:** if it plausibly carries a public-consequence factual claim, keep
  it **YES**; reserve NO for the clearly personal / promotional / sports / entertainment.

Examples (our feed):
| tweet | R | why |
|---|---|---|
| "New EU law will ban new gas cars by 2035." | **YES** | public policy |
| "Measles cases are rising as vaccination rates fall." | **YES** | public health |
| "Amad Diallo gives Ivory Coast the lead in the 84th minute!" | **NO** | sports result, no public angle |
| "This new café downtown has the best croissants." | **NO** | personal / promotional |
| "Squeezie's new video just dropped." | **NO** | entertainment |

---

## H — Harmful

> **If believed and spread, could this claim cause SOCIETAL harm — erode trust in
> institutions, discourage protective behavior, stoke fear, scapegoat a group, or push
> conspiracy?**

Loose, **societal**, and **directional** — not physical injury. Judge the **harm
potential, NOT the truth** of the claim. (This is the binary CheckThat-1C-style harm,
not the 5-dimension FABLE score.)

- **YES** — erodes institutional trust, discourages protection, scapegoats a group,
  stokes fear, or advances a conspiracy narrative.
- **NO** — even a false or public claim can be harmless (trivia, an honest mistake with
  no societal stakes).

Examples (CheckThat-1C style):
| tweet | H | why |
|---|---|---|
| "Don't bother with seatbelts, airbags do all the work." | **YES** | discourages a protective behavior |
| "The election was rigged by the electoral commission." | **YES** | erodes institutional trust / conspiracy |
| "Migrants are the reason crime is rising." | **YES** | scapegoats a group |
| "Aspirin was first synthesized in 1897." | **NO** | verifiable but harmless |
| "The bakery changed its opening hours." | **NO** | no societal stakes |

---

## "Unsure"

Each of V / R / H has an **unsure** state (tap the key again). Use it sparingly — for
genuinely ambiguous cases — so the per-dimension unsure rate stays a meaningful signal.
The FLAG question is ON/OFF only.

## Image-dependent (`i`)

Many tweets reference an image we don't show. If the **text alone** can't carry the
judgment, toggle **image-dependent** and answer as best you can from the text. This is
metadata (not a label) — it lets us measure how much of the feed needs the image.
