from typing import Dict, List, Tuple, Set

SpanKey = Tuple[str, int, int]

def _to_keys(spans: List[Dict]) -> Set[SpanKey]:
    return {(sp["label"], sp["start"], sp["end"]) for sp in spans}

def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}

def score_spans(gold: List[Dict], pred: List[Dict], labels=("ORG", "MON", "LEG")) -> Dict:
    out = {"by_label": {}, "micro": {}}

    g_all = _to_keys(gold)
    p_all = _to_keys(pred)

    tp = len(g_all & p_all)
    fp = len(p_all - g_all)
    fn = len(g_all - p_all)
    out["micro"] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    for lab in labels:
        g = {(l, s, e) for (l, s, e) in g_all if l == lab}
        p = {(l, s, e) for (l, s, e) in p_all if l == lab}
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        out["by_label"][lab] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}

    return out