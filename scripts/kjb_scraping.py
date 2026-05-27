#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOST 科技部门工作（/kjbgz/）爬虫
- 路线1：栏目翻页 index.html, index_1.html, ... 发现详情URL
- 路线2：按 /kjbgz/YYYYMM/tYYYYMMDD_<id>.html 模板试探补漏
- 关键词筛选 + 一条一写 JSONL
"""

import json, re, time, hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Iterator, Optional
import requests
from bs4 import BeautifulSoup

ROOT = "https://www.most.gov.cn"
COLUMN_PATH = "/kjbgz/"
OUT = Path("most_kjbgz.jsonl")
SEEN_FILE = Path("most_kjbgz.seen.txt")  # 断点续跑：已见URL

KEYWORDS = ("人工智能", "大模型", "生成式", "AI", "自动驾驶")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/0.2; +https://example.org/bot)"
}

def load_seen():
    if SEEN_FILE.exists():
        return set(x.strip() for x in SEEN_FILE.read_text(encoding="utf-8").splitlines() if x.strip())
    return set()

SEEN = load_seen()

def persist_seen(url: str):
    with SEEN_FILE.open("a", encoding="utf-8") as f:
        f.write(url + "\n")

def write_jsonl(rec: dict):
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def fetch(url, timeout=20):
    for i in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(1.2 * (i + 1))
    return None

def match_title(title: str) -> bool:
    t = title.lower()
    return any(k.lower() in t for k in KEYWORDS)

DATE_PAT = re.compile(r'(20\d{2})[-年/\.](\d{1,2})[-月/\.](\d{1,2})')

def extract_date_text(html: str) -> str:
    # 常见格式：发布时间：2019-06-17 / 2019年06月17日 / 2019/06/17
    m = DATE_PAT.search(html)
    if not m:
        return ""
    y, mth, d = m.groups()
    try:
        dt = datetime(int(y), int(mth), int(d))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return ""

def parse_detail(url: str) -> Optional[dict]:
    if url in SEEN:
        return None
    r = fetch(url)
    if not r or r.status_code != 200:
        return None
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title else ""
    if not match_title(title):
        SEEN.add(url); persist_seen(url)
        return None
    pubdate = extract_date_text(html)
    rec = {
        "url": url,
        "title": title,
        "date": pubdate,
        "source": "MOST/kjbgz",
        "fetched_at": int(time.time())
    }
    SEEN.add(url); persist_seen(url)
    write_jsonl(rec)  # 抓到一条就写一条
    print("[OK]", pubdate, title)
    return rec

def iter_list_pages() -> Iterator[str]:
    # 自动往后翻到“连续若干页无新链接”就停
    empty_streak = 0
    i = 0
    while True:
        if i == 0:
            url = ROOT + COLUMN_PATH + "index.html"
        else:
            url = ROOT + COLUMN_PATH + f"index_{i}.html"
        r = fetch(url)
        if not r or r.status_code != 200:
            empty_streak += 1
            if empty_streak >= 3:
                break
            i += 1
            continue
        soup = BeautifulSoup(r.text, "lxml")
        links = 0
        for a in soup.select('a[href]'):
            href = a.get("href", "")
            if not href: 
                continue
            if href.startswith("http"):
                u = href
            else:
                u = requests.compat.urljoin(url, href)
            # 只要 kjbgz 栏目下的详情页
            if "/kjbgz/" in u and re.search(r'/t20\d{6}\d{2}_\d+\.html$', u):
                links += 1
                yield u
        if links == 0:
            empty_streak += 1
        else:
            empty_streak = 0
        i += 1
        time.sleep(0.8)  # 礼貌限速

def iter_date_id_guess(start: date, end: date) -> Iterator[str]:
    """
    试探：/kjbgz/YYYYMM/tYYYYMMDD_<id>.html
    - 对每天，先尝试把 <id> 限定在一个相对窄的区间
    - 连续 N 次 miss 就跳出当天
    说明：ID 的绝对范围不清楚，设一个宽但有限的步进（比如 100000~300000）
    """
    # 若你抓到过若干真实ID，可基于统计动态缩窄范围
    ID_MIN, ID_MAX, STEP = 100000, 300000, 1  # 可调；小心负载
    day = start
    while day <= end:
        ymd = day.strftime("%Y%m%d")
        ym = day.strftime("%Y%m")
        miss = 0
        # 为控制请求量，这里采用“稀疏抽样 + 邻域扩张”的策略：
        # 先粗抽样（步长大），命中则在该ID附近小步扫描
        coarse_step = 997  # 质数步长降低周期性碰撞
        hit_points = []
        # 粗扫
        for i in range(ID_MIN, ID_MAX, coarse_step):
            url = f"{ROOT}{COLUMN_PATH}{ym}/t{ymd}_{i}.html"
            r = fetch(url)
            if not r:
                continue
            if r.status_code == 200:
                hit_points.append(i)
                yield url
                miss = 0
            else:
                miss += 1
                if miss >= 50:  # 当天早停
                    break
            time.sleep(0.2)
        # 命中处细扫
        for center in hit_points:
            for j in range(center-50, center+51):
                if j < ID_MIN or j > ID_MAX:
                    continue
                url = f"{ROOT}{COLUMN_PATH}{ym}/t{ymd}_{j}.html"
                yield url
                time.sleep(0.05)
        day += timedelta(days=1)

def main():
    # 路线1：栏目翻页
    for u in iter_list_pages():
        parse_detail(u)

    # 路线2：日期+ID 试探（补漏；起止日期自己改小一点更稳）
    start = date(2010, 1, 1)
    end = date.today()
    for u in iter_date_id_guess(start, end):
        if u in SEEN:
            continue
        parse_detail(u)

if __name__ == "__main__":
    main()
