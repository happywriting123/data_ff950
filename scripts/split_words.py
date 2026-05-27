#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 jieba 对 JSONL 逐行的 body 分词，写回 tokens 字段（可选停用词、用户词典与 n-gram）。
- 输入：一个或多个 JSONL（每行至少有 body，建议有 date/category/source）
- 输出：新的 JSONL，增加 tokens 字段（list[str]）
用法示例：
python prep_with_jieba.py data/*.jsonl --stop stopwords.txt --user-dict mydict.txt --ngram 1 2 --minlen 2 --out pretokenized.jsonl
"""
import argparse, json, re, sys
from pathlib import Path
import jieba

NUM_PAT = re.compile(r"^[0-9]+([.:/\-]?[0-9]+)*$")
PUNCT_TABLE = str.maketrans({c: " " for c in "，。、“”‘’：；！？（）()《》〈〉—-·…、.,:;!?\"'[]{}()+-=\\/<>|"})
CJK_CHAR = re.compile(r"[\u3400-\u9FFF]")

def yield_tokens(text: str, user_dict: str | None):
    if user_dict:
        jieba.load_userdict(user_dict)
    # 先做轻度清洗，避免奇怪空白影响分词
    txt = text.replace("\u00a0", " ").translate(PUNCT_TABLE)
    for t in jieba.cut(txt, HMM=False):
        tt = t.strip()
        if tt:
            yield tt

def make_ngrams(tokens, nset):
    toks = list(tokens)
    for n in nset:
        if n == 1:
            for t in toks:
                yield t
        elif n > 1:
            for i in range(len(toks) - n + 1):
                yield " ".join(toks[i:i+n])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="一个或多个 JSONL 文件")
    ap.add_argument("--stop", default=None, help="停用词（每行一个词）")
    ap.add_argument("--user-dict", dest="ud", default=None, help="jieba 用户词典")
    ap.add_argument("--ngram", nargs="+", type=int, default=[1], help="生成的 n-gram，如 1 2 代表 unigram+bigram")
    ap.add_argument("--minlen", type=int, default=2, help="保留的最小 token 长度（中文单字常为噪音，建议>=2）")
    ap.add_argument("--out", default="pretokenized.jsonl", help="输出 JSONL 路径")
    args = ap.parse_args()

    # 读停用词
    stops = set()
    if args.stop and Path(args.stop).exists():
        stops = {l.strip() for l in Path(args.stop).read_text(encoding="utf-8").splitlines() if l.strip()}

    out_path = Path(args.out)
    w = out_path.open("w", encoding="utf-8", newline="")

    total, kept = 0, 0

    def keep(tok: str) -> bool:
        if tok in stops:
            return False
        if len(tok) < args.minlen and CJK_CHAR.search(tok):  # 单个汉字多为虚词/碎片
            return False
        if NUM_PAT.match(tok):
            return False
        return True

    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                body = rec.get("body", "")
                if not isinstance(body, str) or not body.strip():
                    # 仍然写出原记录，但 tokens 为空列表，方便后续统一处理
                    rec["tokens"] = []
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                # jieba 分词 + 过滤
                toks = [t for t in yield_tokens(body, args.ud) if keep(t)]
                # n-gram
                nset = sorted(set(args.ngram))
                grams = list(make_ngrams(toks, nset))

                rec["tokens"] = grams
                kept += 1
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    w.close()
    print(f"Done. Input records: {total}, tokenized: {kept}, out -> {out_path}")

if __name__ == "__main__":
    sys.exit(main())
