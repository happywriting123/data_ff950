#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, itertools, unicodedata, re, time
from collections import defaultdict

IN_FILE = "keywords.jsonl"
NODES_OUT = "nodes.json"
EDGES_OUT = "edges.json"
META_OUT  = "meta.json"

ZW = re.compile(r"[\u200B-\u200D\uFEFF]")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = ZW.sub("", s)
    return s.strip().lower()

def main():
    docs_total = 0
    kw_df = defaultdict(int)      # kw -> in how many docs
    co = defaultdict(int)         # (a,b) with a<b -> count
    kw_sources = defaultdict(set) # kw -> set of sources
    kw_samples = defaultdict(list)# kw -> sample titles (cap a few)

    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue

            docs_total += 1
            title = doc.get("title") or ""
            src_raw = doc.get("source") or ""
            # 有的 source 以逗号分隔
            sources = [s.strip() for s in str(src_raw).split(",") if s.strip()]

            ngrams = (doc.get("ngrams_top") or {})
            kws = []
            for k in ("1", "2"):
                for x in ngrams.get(k, []) or []:
                    t = norm(x)
                    if t:
                        kws.append(t)

            uniq = sorted(set(kws))
            if not uniq:
                continue

            # df
            for kw in uniq:
                kw_df[kw] += 1
                for s in sources:
                    if len(kw_sources[kw]) < 20:  # 防止极端膨胀
                        kw_sources[kw].add(s)
                # 收样例标题（去重，限制数量）
                arr = kw_samples[kw]
                if title and (not arr or (arr and title not in arr)):
                    if len(arr) < 3:
                        arr.append(title)

            # 共现对
            for a, b in itertools.combinations(uniq, 2):
                co[(a, b)] += 1

    # 输出 nodes
    nodes = []
    for kw, df in kw_df.items():
        nodes.append({
            "id": kw,
            "df": df,
            "samples": kw_samples.get(kw, []),
            "sources": sorted(kw_sources.get(kw, [])),
        })
    nodes.sort(key=lambda x: (-x["df"], x["id"]))

    # 输出 edges
    edges = []
    for (a, b), cnt in co.items():
        edges.append({"source": a, "target": b, "co": cnt})
    edges.sort(key=lambda x: (-x["co"], x["source"], x["target"]))

    with open(NODES_OUT, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False)

    with open(EDGES_OUT, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False)

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "docs_total": docs_total,
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "normalize": "NFKC + strip + lower + remove_zero_width",
        }, f, ensure_ascii=False)

    print(f"✅ Done. docs={docs_total}, nodes={len(nodes)}, edges={len(edges)}")
    print(f"   -> {NODES_OUT}, {EDGES_OUT}, {META_OUT}")

if __name__ == "__main__":
    main()
