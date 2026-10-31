from __future__ import annotations
import re, json
from collections import defaultdict

_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(rf"\b(\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTH}|{_MONTH}\s+\d{{1,2}}(st|nd|rd|th)?,?|{_MONTH}"
                   rf"|\d{{4}}s?(\s*[-–]\s*\d{{2,4}})?|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\b", re.I)


def _ents(ev):
    e = ev.get("entities_involved", [])
    if isinstance(e, str):
        try:
            e = json.loads(e)
        except Exception:
            e = [e]
    return tuple(sorted({re.sub(r"\W+", " ", str(x)).strip().lower() for x in (e or []) if str(x).strip()}))


def event_key(ev):
    s = _DATE.sub(" ", str(ev.get("sentence", "")))
    s = re.sub(r"\W+", " ", s).strip().lower()
    ents = _ents(ev)
    return (ents, s) if ents and s else None


def ts_parts(t):
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", str(t or ""))
    return tuple(int(x) for x in m.groups() if x) if m else None


def conflicting(a, b):
    pa, pb = ts_parts(a), ts_parts(b)
    if not pa or not pb:
        return False
    k = min(len(pa), len(pb))
    return pa[:k] != pb[:k]


def _coarse(t):
    return "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(ts_parts(t) or ())) or str(t)


class ConflictIndex:

    def __init__(self, nodes, chunk_doc):
        self.chunk_doc = chunk_doc
        self.support = defaultdict(lambda: defaultdict(set))
        self.rep = defaultdict(dict)
        for nid, d in nodes.items():
            if not d or not ts_parts(d.get("timestamp")):
                continue
            k = event_key(d)
            if k is None:
                continue
            t = str(d["timestamp"])
            self.support[k][t] |= self.docs_of(d)
            self.rep[k].setdefault(t, {**d, "id": nid})

    def docs_of(self, ev):
        out = set()
        for s in str(ev.get("source_id", "")).split("<SEP>"):
            s = s.strip()
            if s:
                out.add(self.chunk_doc.get(s, s))
        return out

    def stats(self):
        groups = [k for k, v in self.support.items()
                  if any(conflicting(a, b) for a in v for b in v if a < b)]
        cross_doc = [k for k in groups if len(set().union(*self.support[k].values())) > 1]
        return {"event_keys": len(self.support), "conflicting_keys": len(groups),
                "conflicting_keys_across_documents": len(cross_doc)}


_YEAR = re.compile(r"\b(\d{4})\b(?!\s*(BC|BCE|B\.C\.))")
INCONSISTENT_WEIGHT = 0.25


def self_consistent(ev):
    t = ts_parts(ev.get("timestamp"))
    ys = {int(y) for y, _ in _YEAR.findall(str(ev.get("sentence", "")))}
    return not (t and ys and t[0] not in ys)


def fuse(events, index, credibility=None):
    cred = credibility or {}
    out, seen = [], set()
    st = {"conflicts": 0, "replaced": 0, "kept_annotated": 0, "tied": 0}
    local = defaultdict(lambda: defaultdict(set))
    rep = defaultdict(dict)
    for ev in events:
        k = event_key(ev) if ts_parts(ev.get("timestamp")) else None
        if k is not None:
            local[k][str(ev["timestamp"])] |= index.docs_of(ev)
            rep[k].setdefault(str(ev["timestamp"]), ev)
    for ev in events:
        k = event_key(ev) if ts_parts(ev.get("timestamp")) else None
        if k is None:
            out.append(ev); continue
        sup = defaultdict(set)
        for src in (index.support.get(k, {}), local[k]):
            for t, d in src.items():
                sup[t] |= d
        reps = {**index.rep.get(k, {}), **rep[k]}
        weight = lambda t: sum(cred.get(d, 1.0) for d in sup[t]) * (1.0 if self_consistent(reps.get(t, {})) else INCONSISTENT_WEIGHT)
        t = str(ev["timestamp"])
        rivals = [u for u in sup if u != t and conflicting(t, u) and not (sup[t] & sup[u])]
        if not rivals:
            if (k, t) not in seen:
                out.append(ev); seen.add((k, t))
            continue
        st["conflicts"] += 1
        n = lambda u: f"{len(sup[u])} source{'s' if len(sup[u]) != 1 else ''}"
        best = max(rivals, key=weight)
        if weight(best) > weight(t):
            keep, rejected = best, [t] + [u for u in rivals if u != best and weight(u) < weight(best)]
            st["replaced"] += 1
        elif weight(best) == weight(t):
            tied = [t] + [u for u in rivals if weight(u) == weight(t)]
            for u in tied:
                if (k, u) in seen:
                    continue
                e = dict(reps.get(u) or ev); e["timestamp"] = u
                e["sentence"] = (f"{e.get('sentence', '')} [Disputed date: sources disagree on when this event "
                                 f"happened ({'; '.join(f'{_coarse(x)}, {n(x)}' for x in tied)}).]")
                out.append(e); seen.add((k, u))
            st["tied"] += 1
            continue
        else:
            keep, rejected = t, rivals
            st["kept_annotated"] += 1
        if (k, keep) in seen:
            continue
        e = dict(reps.get(keep) or ev); e["timestamp"] = keep
        e["sentence"] = (f"{e.get('sentence', '')} [Conflicting report: "
                         + "; ".join(f"{_coarse(u)} ({n(u)})" for u in rejected)
                         + f"; the date {_coarse(keep)} is better supported ({n(keep)}).]")
        out.append(e); seen.add((k, keep))
    return out, st
