"""Rule-based claim extraction for free prose (spec section 8.2, stage 1).

Detects feature mentions (raw names + glossary aliases), then assigns
direction cues, qualitative-level words and numbers to mentions using a
preceding-window rule ("handles.nfile (658.0, typical) also pushed toward
ransomware" attaches both cues to the nearest preceding mention).

A single direction label in a sentence applies to all mentions in it (natural
language: "X and Y both pushed toward ransomware"); when two conflicting
labels appear, proximity decides.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import yaml

from src.utils.config import Paths
from src.utils.logging import setup_logger

log = setup_logger("extract_rules")

DIRECTION_CUES = [
    # contrast/negation first: "rather than a malicious one" must not inherit
    # the "malicious" -> toward_ransomware default
    (r"\bnot\s+malicious\b", "toward_benign"),
    (r"\bnot\s+suspicious\b", "toward_benign"),
    (r"\brather\s+than\s+(?:a\s+|an\s+)?(?:malicious|suspicious)\b", "toward_benign"),
    (r"\brather\s+than\s+(?:a\s+|an\s+)?ransomware\b", "toward_benign"),
    (r"\blean(?:s|ing)?\s+(?:toward(?:s)?\s+)?benign\b", "toward_benign"),
    (r"\blean(?:s|ing)?\s+(?:toward(?:s)?\s+)?ransomware\b", "toward_ransomware"),
    (r"\bbenign\s+(?:signal|sign|indicator)s?\b", "toward_benign"),
    (r"\bno\s+indication\s+of\s+ransomware\b", "toward_benign"),
    (r"\btoward(?:s)?\s+ransomware\b", "toward_ransomware"),
    (r"\btoward(?:s)?\s+benign\b", "toward_benign"),
    (r"\b(?:supports?|suggests?|indicates?|points?\s+to|consistent\s+with|characteristic\s+of)\s+ransomware\b", "toward_ransomware"),
    (r"\b(?:supports?|suggests?|indicates?|points?\s+to|consistent\s+with|characteristic\s+of)\s+benign\b", "toward_benign"),
    (r"\bransomware[-\s]like\b|\bmalicious\b|\bsuspicious\b|\braises?\s+suspicion\b", "toward_ransomware"),
    (r"\bbenign[-\s]like\b|\breassuring\b|\bargues?\s+against\s+ransomware\b", "toward_benign"),
    (r"\bincreases?\s+the\s+likelihood\b|\braises?\s+the\s+likelihood\b", "toward_ransomware"),
    (r"\blowers?\s+the\s+likelihood\b|\bdecreases?\s+the\s+likelihood\b|\breduces?\s+the\s+likelihood\b", "toward_benign"),
]

LEVEL_CUES = [
    (r"\bvery\s+high\b|\bextremely\s+high\b|\bexceptionally\s+high\b", "very_high"),
    (r"\bvery\s+low\b|\bextremely\s+low\b|\bexceptionally\s+low\b", "very_low"),
    (r"\bhigh\b|\belevated\b|\braised\b|\blarge\b", "high"),
    (r"\blow\b|\breduced\b|\bsmall\b", "low"),
    # "average" is deliberately NOT a level word here: in this feature set it
    # almost always names a per-process average ("average handlers"), not a
    # qualitative level.
    (r"\btypical\b|\bnormal\b|\busual\b|\bunremarkable\b|\bmoderate\b", "typical"),
]

NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)([kKmM])?(?![\w])")
PERCENTILE_SUFFIX = re.compile(r"^\s*(?:th|st|nd|rd)?\s*(?:percentile|pct)", re.I)

# boundary classes exclude hyphens so "per-process count" does NOT match
# the pslist.nproc alias "process count". The trailing class still accepts a
# sentence-final period ("...mutex handles.") by rejecting a dot only when a
# word character follows it (dotted feature names like handles.nsection).
_LB = r"(?<![a-z0-9_.\-])"
_LA = r"(?![a-z0-9_\-]|\.(?=[a-z0-9_]))"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=1)
def _glossary() -> dict:
    path = Paths.root / "src" / "evidence" / "feature_glossary.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)["features"]


def alias_patterns(features: list[str]) -> list[tuple[re.Pattern, str]]:
    """(compiled regex, canonical feature) sorted so longer aliases win."""
    pairs: list[tuple[str, str]] = []
    gloss = _glossary()
    for name in features:
        variants = {name, name.replace(".", " "), name.replace(".", "_"),
                    name.replace(".", " ").replace("_", " ")}
        for alias in (gloss.get(name) or {}).get("aliases", []) or []:
            variants.add(str(alias))
        for v in variants:
            v = v.strip().lower()
            if len(v) >= 4:
                pairs.append((v, name))
    pairs.sort(key=lambda p: -len(p[0]))
    out = []
    for alias, name in pairs:
        esc = re.escape(alias).replace(r"\ ", r"\s+")
        rx = re.compile(rf"{_LB}{esc}{_LA}", re.I)
        out.append((rx, name))
    return out


def find_mentions(sentence: str, patterns) -> list[tuple[int, int, str]]:
    raw: list[tuple[int, int, str]] = []
    for rx, name in patterns:
        for m in rx.finditer(sentence):
            raw.append((m.start(), m.end(), name))
    # Overlapping matches: the longer alias wins ("dlls per process" over
    # "loaded dlls"; "services sharing their host process" over "number of
    # services"). Identical spans are de-duplicated.
    kept: list[tuple[int, int, str]] = []
    for m in sorted(set(raw), key=lambda t: (-(t[1] - t[0]), t[0])):
        if any(not (m[1] <= s or m[0] >= e) for s, e, _n in kept):
            continue
        kept.append(m)
    kept.sort()
    return kept


def number_spans(sentence: str) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for m in NUMBER_RE.finditer(sentence):
        tail = sentence[m.end():]
        if PERCENTILE_SUFFIX.match(tail):
            continue
        if tail.lstrip().startswith("%"):
            continue
        val = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        if suf == "k":
            val *= 1e3
        elif suf == "m":
            val *= 1e6
        out.append((m.start(), m.end(), val))
    return out


def find_numbers(sentence: str) -> list[float]:
    return [v for _s, _e, v in number_spans(sentence)]


def _cue_spans(sentence: str, cues: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Collect cue spans, dropping shorter cues contained in longer ones
    (e.g. "high" inside "very high")."""
    raw = []
    for pattern, label in cues:
        for m in re.finditer(pattern, sentence, flags=re.I):
            raw.append((m.start(), m.end(), label))
    raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, str]] = []
    for s, e, label in raw:
        contained = False
        for s2, e2, _l2 in kept:
            if s2 <= s and e <= e2 and (e - s) < (e2 - s2):
                contained = True
                break
        if not contained:
            kept.append((s, e, label))
    return sorted(kept)


_GLUE_SPAN = re.compile(r"^[\s\d.,;:()%\-]*$")
_ADJ_SPAN = re.compile(r"^[A-Za-z'\- ]*$")
# a following mention past a clause word ("...high, while X", "...high and
# then X") is a new clause, not the noun the cue modifies
_CLAUSE_WORD = re.compile(r"\b(?:and|but|or|while|whereas|however|then|also|which|that|as)\b|[;:]",
                          re.I)


def _inside_mention(span, mentions) -> bool:
    s, e = span[0], span[1]
    return any(ms <= s and e <= me for ms, me, _n in mentions)


def _assign_cues(sentence: str, spans, mentions, window: int = 44,
                 lookahead: int = 16, fallback_lookahead: int = 12) -> dict[str, str]:
    """Attach each cue to a mention.

    Reading order:
    1. structured "<mention> (<value>, <cue>)" — when only digits/punctuation
       separate a preceding mention from the cue ("... (658.0, typical)"),
       the cue belongs to that mention;
    2. adjective position "the <cue> <noun phrase>" — a following mention
       within a short word-only span wins when no mention is attached by (1)
       and the span is not a new clause ("...high, while X" keeps reading 3);
    3. otherwise the closest preceding mention wins (old behaviour).
    """
    assigned: dict[str, tuple[int, str]] = {}
    for s, e, label in spans:
        mid = (s + e) // 2
        prev_best: tuple[int, str, int] | None = None   # gap, name, mention end
        fwd_best: tuple[int, str, int] | None = None    # gap, name, mention start
        for ms, me, name in mentions:
            if me <= mid:
                gap = mid - me
                if gap <= window and (prev_best is None or gap < prev_best[0]):
                    prev_best = (gap, name, me)
            elif ms <= mid < me:
                prev_best = (0, name, me)
            else:
                gap = ms - mid
                if gap <= lookahead and (fwd_best is None or gap < fwd_best[0]):
                    fwd_best = (gap, name, ms)
        chosen: tuple[int, str] | None = None
        if prev_best is not None and _GLUE_SPAN.match(sentence[prev_best[2]:s]):
            chosen = (prev_best[0], prev_best[1])
        if chosen is None and fwd_best is not None:
            between = sentence[e:fwd_best[2]]
            if _ADJ_SPAN.match(between) and not _CLAUSE_WORD.search(between):
                chosen = (fwd_best[0], fwd_best[1])
        if chosen is None:
            if prev_best is not None:
                chosen = (prev_best[0], prev_best[1])
            elif fwd_best is not None and fwd_best[0] <= fallback_lookahead:
                chosen = (fwd_best[0], fwd_best[1])
        if chosen is not None:
            prev = assigned.get(chosen[1])
            if prev is None or chosen[0] < prev[0]:
                assigned[chosen[1]] = (chosen[0], label)
    return {k: v[1] for k, v in assigned.items()}


# An explicit value keyword ("(value 9,215)", "value: 999") leaves no doubt about
# which feature the number belongs to, so it binds to the closest preceding
# mention even across a long phrase ("X is very_high compared with training
# data (value 9,215)"). Without this rule the plain 20-character backward
# window silently dropped those numbers and the value check never ran.
_VALUE_LEAD = re.compile(r"(?:^|[\s(=\[])(?:value|val)\s*[:=]?\s*$", re.I)


def _is_value_led(sentence: str, pos: int) -> bool:
    return bool(_VALUE_LEAD.search(sentence[:pos]))


def _assign_numbers(sentence: str, numbers: list[tuple[int, int, float]], mentions,
                    window: int = 20, forward_window: int = 16,
                    value_window: int = 64) -> dict[str, float]:
    """Attach numbers to mentions, preferring the forward reading.

    Forward ("<number> <noun phrase>", e.g. "339 section handles", "43 loaded
    DLLs per process") wins when only adjectives/glue separate the number from
    the following mention; otherwise fall back to the closest mention that ends
    before the number ("section handles (339)"). A mention already bound
    forward is not overwritten by the backward pass; a number introduced by the
    keyword "value" may bind across a much longer backward span.
    """
    assigned: dict[str, tuple[int, float]] = {}
    for pos, end, val in numbers:
        best: tuple[int, str] | None = None
        for ms, _me, name in mentions:
            if ms < end:
                continue
            between = sentence[end:ms]
            if len(between) > forward_window:
                continue
            if re.fullmatch(r"[A-Za-z'\- ]*", between) is None:
                continue
            gap = ms - end
            if best is None or gap < best[0]:
                best = (gap, name)
        if best is None:
            continue
        prev = assigned.get(best[1])
        if prev is None or best[0] < prev[0]:
            assigned[best[1]] = (best[0], val)

    back_best: dict[str, tuple[int, float]] = {}
    for pos, _end, val in numbers:
        limit = value_window if _is_value_led(sentence, pos) else window
        for _ms, me, name in mentions:
            if me > pos:
                continue
            gap = pos - me
            if gap > limit:
                continue
            prev = back_best.get(name)
            if prev is None or gap < prev[0]:
                back_best[name] = (gap, val)
    for name, (gap, val) in back_best.items():
        if name not in assigned:
            assigned[name] = (gap, val)
    return {k: v[1] for k, v in assigned.items()}


def extract_claims(prose: str, features: list[str]) -> list[dict]:
    """Return claim dicts with extractor='rules'."""
    patterns = alias_patterns(features)
    claims: list[dict] = []
    for si, sentence in enumerate(split_sentences(prose)):
        mentions = find_mentions(sentence, patterns)
        if not mentions:
            continue
        # A cue inside a mention's own span is part of the feature phrase, not a
        # modifier ("...per-process *average* of modules..." must not read
        # "average" as the level word "typical").
        dir_spans = [t for t in _cue_spans(sentence, DIRECTION_CUES)
                     if not _inside_mention(t, mentions)]
        lvl_spans = [t for t in _cue_spans(sentence, LEVEL_CUES)
                     if not _inside_mention(t, mentions)]
        numbers = number_spans(sentence)

        labels = {lab for _s, _e, lab in dir_spans}
        if len(labels) == 1:
            only = next(iter(labels))
            dir_by_name = {name: only for _s, _e, name in mentions}
        else:
            dir_by_name = _assign_cues(sentence, dir_spans, mentions)
        lvl_by_name = _assign_cues(sentence, lvl_spans, mentions)
        num_by_name = _assign_numbers(sentence, numbers, mentions)

        for _ms, _me, name in mentions:
            v = num_by_name.get(name)
            claims.append({
                "feature": name,
                "direction": dir_by_name.get(name),
                "level": lvl_by_name.get(name),
                "value_stated": v,
                "values_all": [v] if v is not None else [],
                "text": sentence,
                "extractor": "rules",
                "sentence_index": si,
            })
    return claims


VERDICT_PATTERNS = [
    r"\bverdict\s*(?:is|:)?\s*[: ]*\b(ransomware|benign)\b",
    r"\b(ransomware|benign)\s+is\s+the\s+predicted\s+(?:label|verdict|class)\b",
    r"\b(?:the\s+)?(?:predicted\s+)?(?:label|verdict|class)\s+is\s+(ransomware|benign)\b",
    r"\b(?:predicts?|classifie[sd]s?|labels?|calls?|scores?|flags?|says?|reports?|gives?|returns?)"
    r"\b[^.]{0,40}?\b(ransomware|benign)\b",
    r"\b(?:predicts?|classifie[sd]\s+(?:it\s+)?as|classifies?\s+(?:the\s+sample\s+)?as|"
    r"label(?:ed|s)?\s+as|concludes?\s+(?:that\s+)?(?:the\s+sample\s+is\s+)?)"
    r"\s+(?:the\s+sample\s+is\s+)?(ransomware|benign)\b",
    r"\b(?:the|this)\s+(?:sample|binary|dump|memory\s+image)\s+(?:is|appears\s+to\s+be|"
    r"is\s+likely|is\s+most\s+likely)\s+(ransomware|benign)\b",
    r"\blikely\s+(ransomware|benign)\b",
    r"\b(ransomware|benign)[-\s]verdict\b",
]

PROB_PATTERNS = [
    r"\bprobability[^0-9%]{0,25}(\d+(?:\.\d+)?)\s*%",
    r"\bprobability[^0-9%]{0,25}(0?\.\d+|1\.0+)\b",
    r"\b(?:ransomware\s+)?(?:probability|confidence|chance|score)\s*(?:of|is|:|was)?\s*(0?\.\d+)\b",
    r"\b(\d+(?:\.\d+)?)\s*%\s*(?:probability|confidence|chance)",
    r"\b(0\.\d+)\s*(?:probability|confidence|chance)",
]


def infer_verdict_and_probability(prose: str) -> tuple[str | None, float | None]:
    """Extract the stated verdict/probability from free prose (rules only)."""
    low = prose.lower()
    verdict = None
    for pat in VERDICT_PATTERNS:
        m = re.search(pat, low, flags=re.I)
        if m:
            verdict = m.group(1).lower()
            break
    if verdict is None:
        # Sentence-initial verdicts ("Benign, with a ransomware probability of ...",
        # "Ransomware with probability 0.90") that carry no explicit cue word.
        m_head = re.match(r"[\W_]*(ransomware|benign)\b", low)
        if m_head:
            verdict = m_head.group(1)
    if verdict is None:
        n_rans = len(re.findall(r"\bransomware\b", low))
        n_ben = len(re.findall(r"\bbenign\b", low))
        if n_rans and not n_ben:
            verdict = "ransomware"
        elif n_ben and not n_rans:
            verdict = "benign"
    prob = None
    for pat in PROB_PATTERNS:
        m = re.search(pat, low, flags=re.I)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            if 0.0 <= val <= 1.0:
                prob = val
                break
    return verdict, prob


def unparsed_feature_sentences(prose: str, features: list[str], claims: list[dict]) -> list[str]:
    """Sentences that mention a feature but produced no direction/level/number claim."""
    parsed = {c["sentence_index"] for c in claims
              if c.get("direction") or c.get("level") or c.get("value_stated")}
    patterns = alias_patterns(features)
    out = []
    for si, sentence in enumerate(split_sentences(prose)):
        if si in parsed:
            continue
        if find_mentions(sentence, patterns):
            out.append(sentence)
    return out


def main() -> int:  # tiny CLI for debugging
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    args = ap.parse_args()
    feats = json.loads((Paths.processed / "features.json").read_text())
    for c in extract_claims(args.text, feats):
        print(json.dumps(c, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
