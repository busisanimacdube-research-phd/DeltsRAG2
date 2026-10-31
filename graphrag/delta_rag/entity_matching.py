from __future__ import annotations
import re
import unicodedata
from typing import List

_WS = re.compile(r"\s+")


def normalize_v1(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def match_v1(q: str, e: str) -> bool:
    if not q or not e:
        return False
    if q == e:
        return True
    if len(q) >= 4 and (q in e or e in q):
        return True
    qw, ew = set(q.split()), set(e.split())
    if not qw or not ew:
        return False
    inter = len(qw & ew)
    return bool(inter and inter / len(qw | ew) >= 0.6 and any(len(t) >= 4 for t in (qw & ew)))


def normalize_strict(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return _WS.sub(" ", s).strip()


def valid_index_key(k: str) -> bool:
    return any(len(t) >= 3 for t in k.split())


def filter_query_entities(entities: List[str], question: str) -> List[str]:
    kept = []
    for e in entities or []:
        e = str(e).strip()
        if not e or not valid_index_key(normalize_strict(e)):
            continue
        in_q = re.search(re.escape(e), question, flags=re.IGNORECASE)
        surface = in_q.group(0) if in_q else e
        if any(ch.isupper() for ch in surface):
            kept.append(e)
    return kept or [str(e) for e in (entities or []) if str(e).strip()]


def match_strict(q: str, e: str) -> bool:
    if not q or not e:
        return False
    if q == e:
        return True
    qt, et = q.split(), e.split()
    qs, es = set(qt), set(et)
    if qs <= es and any(len(t) >= 4 for t in qs):
        return True
    if es <= qs and qt[-1] in es and len(qt[-1]) >= 4:
        return True
    if (len(qt[-1]) >= 4 and et[-1] == qt[-1] and len(et) > 1
            and all(any(len(t) >= 2 and (u.endswith(t) or u.startswith(t)) for u in qt[:-1])
                    for t in et[:-1])):
        return True
    inter = qs & es
    return bool(inter and len(inter) / len(qs | es) >= 0.6 and any(len(t) >= 4 for t in inter))
