#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, re
from collections import Counter
from datetime import datetime
from hashlib import sha1
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ============ Tokenization (English only) ============

# 单词：字母数字开头，可含一次或多次撇号后缀（can't, don't），也允许纯数字字母组合（gpt4）
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def normalize_text(s: str) -> str:
    return (s or "").replace("\u3000", " ").strip()

def load_stopwords(path: str) -> set:
    sw = set()
    if not path: return sw
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                sw.add(w)
    return sw

def file_sha1(path: str) -> str:
    if not path: return ""
    h = sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1<<16), b''):
            h.update(b)
    return h.hexdigest()

def tokenize(text: str) -> List[str]:
    # 统一小写
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def letters_len(s: str) -> int:
    # 只统计字母数字长度：不含空格/标点
    return sum(ch.isalnum() for ch in s)

def join_tokens(chunk: List[str], join_with: str) -> str:
    return (join_with if join_with is not None else " ").join(chunk)

def topk_pairs(pairs: List[Tuple[str,float]], k:int) -> List[Tuple[str,float]]:
    seen, out = set(), []
    for w,s in pairs:
        if w in seen: continue
        seen.add(w); out.append((w,s))
        if len(out) >= k: break
    return out

def minmax_norm(pairs: List[Tuple[str,float]]) -> Dict[str,float]:
    if not pairs: return {}
    vs = np.array([s for _,s in pairs], dtype=float)
    vmin, vmax = float(vs.min()), float(vs.max())
    if math.isclose(vmin, vmax):
        return {w: 0.0 for w,_ in pairs}
    scaled = (vs - vmin) / (vmax - vmin)
    return {pairs[i][0]: float(scaled[i]) for i in range(len(pairs))}

# ============ IDF (two-pass) with proper DF ============

def compute_idf(in_path: str, n_values: Sequence[int], stopwords: set,
                join_with: str, minlen: int):
    df, N = Counter(), 0
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
            except:
                continue
            body = normalize_text(item.get("body") or "")
            if not body: continue
            N += 1
            toks = tokenize(body)
            L = len(toks)
            doc_terms = set()

            for n in sorted(set(x for x in n_values if x >= 1)):
                if n == 1:
                    for t in toks:
                        if not t: continue
                        if stopwords and t in stopwords: continue
                        if letters_len(t) < minlen: continue
                        doc_terms.add(t)
                else:
                    for i in range(L - n + 1):
                        chunk = toks[i:i+n]
                        if stopwords and any(w in stopwords for w in chunk):
                            continue
                        term = join_tokens(chunk, join_with)
                        if letters_len(term) < minlen: continue
                        doc_terms.add(term)

            df.update(doc_terms)

    # smooth idf
    idf = {term: math.log((N + 1) / (d + 1)) + 1.0 for term, d in df.items()}
    return idf, N

# ============ TF-IDF per document (filtered n-grams) ============

def tfidf_by_n(body: str, idf: Dict[str,float], n_values: Sequence[int],
               stopwords:set, join_with:str, minlen:int):
    toks = tokenize(body); L = len(toks)
    out: Dict[int, List[Tuple[str,float]]] = {}

    for n in sorted(set(x for x in n_values if x >= 1)):
        terms: List[str] = []
        if n == 1:
            for t in toks:
                if not t: continue
                if stopwords and t in stopwords: continue
                if letters_len(t) < minlen: continue
                terms.append(t)
        else:
            for i in range(L - n + 1):
                chunk = toks[i:i+n]
                if stopwords and any(w in stopwords for w in chunk):
                    continue
                term = join_tokens(chunk, join_with)
                if letters_len(term) < minlen: continue
                terms.append(term)
        tf = Counter(terms)
        pairs = [(w, tf[w] * idf.get(w, 0.0)) for w in tf]
        pairs.sort(key=lambda x: x[1], reverse=True)
        out[n] = pairs
    return out

# ============ TextRank (word-level) ============

def textrank_word_scores(body: str, window: int, stopwords: set,
                         iters: int = 30, d: float = 0.85) -> Dict[str, float]:
    """
    简化版 TextRank：词共现图 + power iteration
    """
    words = [w for w in tokenize(body) if not stopwords or w not in stopwords]
    if not words:
        return {}
    idx = {w: i for i, w in enumerate(set(words))}
    N = len(idx)
    if N == 0:
        return {}

    mat = np.zeros((N, N), dtype=float)
    W = max(2, int(window))
    for i, w in enumerate(words):
        a = idx[w]
        for j in range(i + 1, min(i + W, len(words))):
            v = words[j]
            if w == v: continue
            b = idx[v]
            mat[a, b] += 1.0
            mat[b, a] += 1.0

    # 行归一化
    rowsum = mat.sum(axis=1)
    for i in range(N):
        s = rowsum[i]
        if s > 0:
            mat[i] /= s

    pr = np.ones(N, dtype=float) / N
    base = (1.0 - d) / N
    for _ in range(iters):
        pr = base + d * mat.T.dot(pr)

    return {w: float(pr[idx[w]]) for w in idx}

def phrase_scores_from_words(body: str, word_score: Dict[str,float], n_values: Sequence[int],
                             join_with:str, minlen:int, stopwords:set,
                             agg:str="sum", keep_stop:bool=False):
    toks = tokenize(body); L = len(toks)
    out: Dict[int, List[Tuple[str,float]]] = {}

    for n in sorted(set(x for x in n_values if x >= 1)):
        phrases: List[Tuple[str,float]] = []
        if n == 1:
            for t in toks:
                if not t: continue
                if not keep_stop and stopwords and t in stopwords: continue
                if letters_len(t) < minlen: continue
                s = word_score.get(t, 0.0)
                if s > 0:
                    phrases.append((t, s))
        else:
            for i in range(L - n + 1):
                chunk = toks[i:i+n]
                if not keep_stop and stopwords and any(w in stopwords for w in chunk):
                    continue
                term = join_tokens(chunk, join_with)
                if letters_len(term) < minlen:
                    continue
                scs = [word_score.get(w, 0.0) for w in chunk]
                if not any(s > 0 for s in scs):
                    continue
                s = sum(scs) if agg == "sum" else (sum(scs) / len(scs))
                phrases.append((term, s))
        phrases.sort(key=lambda x: x[1], reverse=True)
        out[n] = phrases
    return out

# ============ Main ============

def main():
    print("[BOOT] kw_offline_en.py started", datetime.now().isoformat(), flush=True)

    ap = argparse.ArgumentParser(
        prog="kw_offline_en.py",
        description="English-only TF-IDF (n-gram) + TextRank(phrase) + Fused; body-only"
    )
    ap.add_argument("--in", dest="in_path", required=True, help="input JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="output JSONL")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ngram", type=str, default="1,2")
    ap.add_argument("--minlen", type=int, default=3, help="min alnum length for terms/phrases")
    ap.add_argument("--stopwords", type=str, default=None)
    ap.add_argument("--join-with", type=str, default=" ")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tr-window", type=int, default=5)
    ap.add_argument("--tr-agg", type=str, choices=["sum","mean"], default="sum")
    ap.add_argument("--tr-keep-stop", action="store_true")
    args = ap.parse_args()

    print(f"[ARGS] in={args.in_path} out={args.out_path} topk={args.topk} "
          f"ngram={args.ngram} alpha={args.alpha} tr_agg={args.tr_agg}", flush=True)

    stopwords = load_stopwords(args.stopwords)
    if stopwords:
        print(f"[INFO] stopwords loaded: {len(stopwords)}", flush=True)

    n_values = [int(x.strip()) for x in args.ngram.split(",") if x.strip()]
    if not n_values:
        n_values = [1, 2]

    print("[PASS1] building IDF ...", flush=True)
    idf, N = compute_idf(args.in_path, n_values, stopwords, args.join_with, args.minlen)
    print(f"[PASS1] docs_with_body={N}, vocab={len(idf)}", flush=True)
    if N == 0:
        print("[WARN] no valid body found; exiting.", flush=True)
        return

    meta = {
        "ngram": n_values, "topk": args.topk, "minlen": args.minlen, "alpha": args.alpha,
        "tr_window": args.tr_window,
        "allow_pos": [],  # 英文版占位以保持结构一致
        "stopwords_sha1": file_sha1(args.stopwords) if args.stopwords else "",
        "userdict_sha1": "",  # 占位（与原结构对齐）
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }

    n_read = n_write = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except:
                continue
            n_read += 1
            body = normalize_text(item.get("body") or "")
            if body:
                tfidf_map = tfidf_by_n(body, idf, n_values, stopwords, args.join_with, args.minlen)
                word_score = textrank_word_scores(body, window=args.tr_window, stopwords=stopwords)
                tr_map = phrase_scores_from_words(body, word_score, n_values, args.join_with,
                                                  args.minlen, stopwords,
                                                  agg=args.tr_agg, keep_stop=args.tr_keep_stop)
                fused_map: Dict[int, List[Tuple[str,float]]] = {}
                for n in n_values:
                    tf_pairs = tfidf_map.get(n, [])
                    tr_pairs = tr_map.get(n, [])
                    tf_norm = minmax_norm(tf_pairs)
                    tr_norm = minmax_norm(tr_pairs)
                    keys = {w for w,_ in tf_pairs} | {w for w,_ in tr_pairs}
                    fused = [(w, args.alpha*tf_norm.get(w,0.0) + (1-args.alpha)*tr_norm.get(w,0.0)) for w in keys]
                    fused.sort(key=lambda x:x[1], reverse=True)
                    fused_map[n] = fused

                def pack(m):
                    out = {}
                    for n in n_values:
                        out[str(n)] = [{"w": w, "s": float(s)} for w,s in topk_pairs(m.get(n, []), args.topk)]
                    return out

                item["kw"] = {
                    "tfidf":   pack(tfidf_map),
                    "textrank":pack(tr_map),
                    "fused":   pack(fused_map),
                }
            else:
                item.setdefault("kw", {"tfidf":{}, "textrank":{}, "fused":{}})

            item["_meta"] = meta
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_write += 1

    print(f"[DONE] read={n_read}, wrote={n_write}, out={args.out_path}", flush=True)

if __name__ == "__main__":
    main()
