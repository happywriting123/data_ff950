#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
关键词共现图（PyVis 美化增强版）
- 仅使用 ngrams_top["1"] 与 ["2"]
- 美化：社区着色、ForceAtlas2 调参、节点大小/边粗映射、标签控拥挤
- 筛选：min-co / min-df / keep / drop / neighbors / giant / topk
- 友好 tooltip：展示 df 与示例标题（自动抽样 1~3 条）

依赖：
  pip install pyvis networkx
"""

import json
import itertools
import argparse
import unicodedata
import re
from collections import defaultdict, deque

import networkx as nx
from pyvis.network import Network

# ---------- 规范化 ----------
_ZW = re.compile(r"[\u200B-\u200D\uFEFF]")

def norm(s: str) -> str:
    """NFKC + 去零宽 + strip + lower"""
    s = unicodedata.normalize("NFKC", str(s))
    s = _ZW.sub("", s)
    return s.strip().lower()

# ---------- 读取 ----------
def read_keywords(jsonl_path):
    """
    返回：
      docs_keywords: List[List[str]]  # 每篇文档的关键词（1/2-gram）
      kw_samples: Dict[str, List[str]]  # 关键词 -> 示例标题
    """
    docs_keywords = []
    kw_samples = defaultdict(list)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except Exception as e:
                print(f"[WARN] 第{ln}行 JSON 解析失败: {e}")
                continue

            title = (item.get("title") or "").strip()
            ngrams = item.get("ngrams_top") or item.get("ngrams") or {}
            kws = []
            for k in ("2", "3"):
                for x in ngrams.get(k, []) or []:
                    t = norm(x)
                    if t:
                        kws.append(t)
                        # 收 1~3 个示例标题
                        if title:
                            arr = kw_samples[t]
                            if title not in arr and len(arr) < 3:
                                arr.append(title)
            uniq = list(set(kws))
            if uniq:
                docs_keywords.append(uniq)

    return docs_keywords, kw_samples

# ---------- 统计 ----------
def build_counts(docs_keywords):
    """
    返回：
      co_counts: Dict[(a,b), int]  # a<b 的边共现次数
      kw_doc_count: Dict[str, int]  # 节点文档频次 df
    """
    co_counts = defaultdict(int)
    kw_doc_count = defaultdict(int)

    for kws in docs_keywords:
        uniq = set(kws)
        for kw in uniq:
            kw_doc_count[kw] += 1
        for a, b in itertools.combinations(sorted(uniq), 2):
            co_counts[(a, b)] += 1

    return co_counts, kw_doc_count

# ---------- keep + 邻居 ----------
def nodes_from_keep(edges, keep_list, hops=1):
    """从 keep 子串匹配的节点出发，取 k 跳邻居"""
    if not keep_list:
        return None
    g = nx.Graph()
    for (a, b), w in edges.items():
        g.add_edge(a, b, weight=w)
    seeds = set()
    for n in g.nodes():
        if any(k in n for k in keep_list):
            seeds.add(n)
    if not seeds:
        print("[INFO] --keep 未命中节点，忽略。")
        return None
    if hops <= 0:
        return seeds
    keep_nodes = set(seeds)
    q = deque([(s, 0) for s in seeds])
    seen = set(seeds)
    while q:
        v, d = q.popleft()
        if d == hops:
            continue
        for nb in g.neighbors(v):
            if nb not in seen:
                seen.add(nb)
                keep_nodes.add(nb)
                q.append((nb, d + 1))
    return keep_nodes

# ---------- 社区着色 ----------
def greedy_communities(nodes, edges):
    """
    用 NetworkX greedy modularity 社区划分（无需额外库）
    返回：dict: node -> community_id
    """
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for (a, b), w in edges.items():
        g.add_edge(a, b, weight=w)
    if g.number_of_edges() == 0 or g.number_of_nodes() == 0:
        return {n: 0 for n in g.nodes()}
    comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight='weight'))
    node2c = {}
    for cid, cset in enumerate(comms):
        for n in cset:
            node2c[n] = cid
    # 某些孤点（被过滤后）不在任何社区，归为 0
    for n in g.nodes():
        node2c.setdefault(n, 0)
    return node2c

def pastel_hex(i, s=55, l=68):
    """
    生成柔和色 HSL -> HEX；i 为社区编号
    s,l 取 0-100 之间的整数
    """
    h = (i * 137) % 360  # 取一个“漂亮”的步长打散颜色
    return hsl_to_hex(h, s, l)

def hsl_to_hex(h, s, l):
    s /= 100.0
    l /= 100.0
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60.0) % 2 - 1))
    m = l - c/2
    r, g, b = 0, 0, 0
    if   0 <= h < 60:   r, g, b = c, x, 0
    elif 60 <= h < 120: r, g, b = x, c, 0
    elif 120<= h <180:  r, g, b = 0, c, x
    elif 180<= h <240:  r, g, b = 0, x, c
    elif 240<= h <300:  r, g, b = x, 0, c
    else:               r, g, b = c, 0, x
    to255 = lambda v: int(round((v + m) * 255))
    return "#{:02x}{:02x}{:02x}".format(to255(r), to255(g), to255(b))

# ---------- 可视化 ----------
def visualize(co_counts, kw_doc_count, kw_samples, args):
    # 1) 初筛边：min_co
    edges = {e: w for e, w in co_counts.items() if w >= args.min_co}

    # 2) 初筛点：min_df
    nodes = {kw for kw, df in kw_doc_count.items() if df >= args.min_df}

    # 3) drop 黑名单
    if args.drop:
        drops = [norm(x) for x in args.drop]
        nodes = {n for n in nodes if not any(d in n for d in drops)}

    # 4) 仅保留仍出现在有效边两端的节点（自然去孤点）
    connected_nodes = set()
    for (a, b), _ in edges.items():
        if a in nodes and b in nodes:
            connected_nodes.add(a)
            connected_nodes.add(b)
    # 裁边
    edges = {(a, b): w for (a, b), w in edges.items() if a in connected_nodes and b in connected_nodes}

    # 5) keep + 邻居聚焦（基于当前 edges）
    if args.keep:
        keeps = [norm(x) for x in args.keep]
        kept = nodes_from_keep(edges, keeps, args.neighbors)
        if kept is not None:
            connected_nodes &= kept
            edges = {(a, b): w for (a, b), w in edges.items() if a in connected_nodes and b in connected_nodes}

    # 再次去孤点
    deg = defaultdict(int)
    for (a, b), w in edges.items():
        deg[a] += 1
        deg[b] += 1
    connected_nodes = {n for n in connected_nodes if deg.get(n, 0) > 0}

    if not edges or not connected_nodes:
        raise SystemExit("[INFO] 过滤后无边或无节点，请放宽 min-co / min-df，或调整 keep/drop。")

    # 6) 每节点 TopK 边（可选；按共现值降序，仅保留强连接，减少毛刺）
    if args.topk and args.topk > 0:
        by_node = defaultdict(list)
        for (a, b), w in edges.items():
            by_node[a].append((b, w))
            by_node[b].append((a, w))
        allowed = set()
        for n, lst in by_node.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            for nb, w in lst[: args.topk]:
                a, b = (n, nb) if n < nb else (nb, n)
                allowed.add((a, b))
        edges = {e: edges[e] for e in edges if e in allowed}

        # 重新去孤点
        deg.clear()
        for (a, b), w in edges.items():
            deg[a] += 1
            deg[b] += 1
        connected_nodes = {n for n in connected_nodes if deg.get(n, 0) > 0}

    # 7) 仅保留最大连通子图（可选）
    if args.giant:
        g = nx.Graph()
        g.add_nodes_from(connected_nodes)
        for (a, b), w in edges.items():
            g.add_edge(a, b, weight=w)
        comps = sorted(nx.connected_components(g), key=len, reverse=True)
        main = comps[0]
        connected_nodes = set(main)
        edges = {(a, b): w for (a, b), w in edges.items() if a in connected_nodes and b in connected_nodes}

    # 8) 社区划分，赋色
    node2c = greedy_communities(connected_nodes, edges)

    # 9) 标签控拥挤：只给“前 N”显示 label（按 df 或度）
    # 评分：alpha*df + (1-alpha)*degree
    alpha = 0.7
    scores = {}
    for n in connected_nodes:
        df = kw_doc_count.get(n, 1)
        d  = deg.get(n, 0)
        scores[n] = alpha*df + (1-alpha)*d
    label_allow = set(sorted(connected_nodes, key=lambda x: scores[x], reverse=True)[: args.label_top])

    # 10) PyVis 绘制
    net = Network(height=args.height, width="100%", bgcolor="#ffffff", font_color="#1f2937",
                  cdn_resources="in_line", directed=False)

    # ForceAtlas2 调参（更紧凑、更稳）
    net.force_atlas_2based(
        gravity=-30,              # 负值可更开一些；-20~-40 看图感
        spring_length=110,        # 边长
        spring_strength=0.08,     # 弹性
        damping=0.85,             # 阻尼
        central_gravity=0.015     # 中心吸引
    )
    if args.buttons:
        net.show_buttons(filter_=['physics'])

    # 节点
    # 大小：size = base + scale * df^0.85
    base = 8.0
    scale = 2.2
    for n in connected_nodes:
        df = kw_doc_count.get(n, 1)
        size = base + scale * (df ** 0.85)
        cid  = node2c.get(n, 0)
        color = pastel_hex(cid)
        label = n if n in label_allow else ""  # 控标签数量
        samples = kw_samples.get(n, [])
        tip = f"df: {df}"
        if samples:
            tip += "<br>" + " / ".join(samples)
        net.add_node(
            n_id=n,
            label=label,
            title=tip,
            size=size,
            color=color,
            borderWidth=1,
            borderWidthSelected=2
        )

    # 边（value 控宽度）
    for (a, b), w in edges.items():
        net.add_edge(a, b, value=w, title=f"共现: {w}")

    net.write_html(args.output, open_browser=False, notebook=False)
    print(f"✅ 出图完成：{args.output}")
    print(f"节点: {len(connected_nodes)}，边: {len(edges)}")

# ---------- 主程序 ----------
def parse_args():
    p = argparse.ArgumentParser(description="PyVis 关键字共现图（美化增强版）")
    p.add_argument("--input",  default="keywords.jsonl", help="输入 JSONL（每行一个文档）")
    p.add_argument("--output", default="keyword_cooccurrence.html", help="输出 HTML 文件名")
    p.add_argument("--min-co", type=int, default=2, help="最小共现次数阈值（边过滤）")
    p.add_argument("--min-df", type=int, default=2, help="最小文档频次阈值（点过滤）")
    p.add_argument("--keep", type=str, default="", help="只保留名称包含这些子串的节点（逗号分隔），并可扩邻居")
    p.add_argument("--neighbors", type=int, default=1, help="keep 的邻居扩展跳数（0=不扩展）")
    p.add_argument("--drop", type=str, default="", help="去掉名称包含这些子串的节点（逗号分隔）")
    p.add_argument("--giant", action="store_true", help="只保留最大连通子图")
    p.add_argument("--topk", type=int, default=0, help="每节点仅保留共现最强的前 K 条边，0 表示不限制")
    p.add_argument("--label-top", type=int, default=120, help="只给前 N 个重要节点显示标签，减轻拥挤")
    p.add_argument("--height", type=str, default="880px", help="画布高度（如 720px / 90vh）")
    p.add_argument("--buttons", action="store_true", help="显示 physics 面板按钮")
    args = p.parse_args()

    args.keep = [s.strip() for s in args.keep.split(",") if s.strip()]
    args.drop = [s.strip() for s in args.drop.split(",") if s.strip()]
    return args

if __name__ == "__main__":
    args = parse_args()
    docs_keywords, kw_samples = read_keywords(args.input)
    if not docs_keywords:
        raise SystemExit("[ERROR] 未解析到任何 1/2-gram 关键字，请检查输入文件与字段路径。")
    co_counts, kw_doc_count = build_counts(docs_keywords)
    visualize(co_counts, kw_doc_count, kw_samples, args)
