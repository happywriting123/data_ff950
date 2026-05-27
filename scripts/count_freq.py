#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jsonl 语料词频 / n-gram 统计（只读 body 字段）
- 每行 JSONL 必须有 body:str
- 中文分词 + 可选 n-gram 聚合
- 停用词、数字/日期/编号过滤
"""

import argparse, json, re, sys
from pathlib import Path
from collections import Counter
import d

NUM_PAT = re.compile(r"^\d+([.\-:/年号月日])?\d*$")
IDX_PAT = re.compile(r"^[（(]?[一二三四五六七八九十0-9ivxIVX]+[）)]?$")
PUNCT = "，。、“”‘’：；！？（）()《》〈〉—-·…、.,:;!?\"'[]{}()+-=/\\<>|"
TABLE = str.maketrans({c:" " for c in PUNCT})

def tokenize(text, user_dict=None):
    if user_dict:
        d.load_userdict(user_dict)
    text = text.replace("\u00a0"," ").translate(TABLE)
    for tok in d.cut(text, HMM=False):
        tok = tok.strip()
        if tok:
            yield tok

def gen_ngrams(tokens, nvals=(1,)):
    toks = list(tokens)
    for n in nvals:
        if n == 1:
            for t in toks: yield t
        else:
            for i in range(len(toks)-n+1):
                yield " ".join(toks[i:i+n])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--stop", default=None, help="停用词文件，每行一个词（可留空）")
    ap.add_argument("--user-dict", dest="ud", default=None, help="自定义分词词典")
    ap.add_argument("--ngram", nargs="+", type=int, default=[1], help="统计的 n，例如 1 2")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--out", default="freq.tsv")
    args = ap.parse_args()

    stops = set()
    if args.stop and Path(args.stop).exists():
        stops = {l.strip() for l in Path(args.stop).read_text(encoding="utf-8").splitlines() if l.strip()}

    ctr = Counter()
    docfreq = Counter()
    total_docs = 0

    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "body" not in rec or not isinstance(rec["body"], str):
                    continue
                total_docs += 1
                seen_this_doc = set()

                toks = list(tokenize(rec["body"], args.ud))
                toks = [t for t in toks
                        if t not in stops
                        and not NUM_PAT.match(t)
                        and not IDX_PAT.match(t)
                        and len(t) > 1]

                grams = list(gen_ngrams(toks, tuple(args.ngram)))
                ctr.update(grams)
                seen_this_doc.update(set(grams))
                docfreq.update(seen_this_doc)

    with open(args.out, "w", encoding="utf-8", newline="") as w:
        w.write("term\tcount\tdf\ttf_idf_like\n")
        for term, c in ctr.most_common():
            if c < args.min_count: continue
            df = docfreq.get(term, 0)
            import math
            score = c * math.log(1 + (total_docs / (1 + df)))
            w.write(f"{term}\t{c}\t{df}\t{score:.4f}\n")

if __name__ == "__main__":
    sys.exit(main())
