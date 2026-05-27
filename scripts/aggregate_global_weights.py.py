import json
import math
import csv
import os
from collections import defaultdict

# ================= 配置区域 / Configuration =================
INPUT_FILE = 'local_keywords_extracted.jsonl'       # 你的输入文件名
OUTPUT_FILE = 'english_global_keywords.csv' # 输出文件名
MIN_DOC_FREQ = 2                        # 过滤阈值 (DF Threshold)
                                        # 建议: 英文的长 n-gram (如 5-9 gram) 重复率通常较低。
                                        # 如果你发现结果太少，可以将此值设为 1。
MAX_NGRAM = 5                           # 最大提取到几元词组 (根据你的数据是 9)
# ===========================================================

def main():
    # 1. 初始化存储 / Initialize storage
    global_weights = defaultdict(float) # { "keyword": total_score }
    doc_frequency = defaultdict(int)    # { "keyword": doc_count }
    ngram_types = {}                    # { "keyword": "9-gram" }
    
    total_docs = 0
    skipped_lines = 0

    print(f"[-] Reading file: {INPUT_FILE} ...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: File {INPUT_FILE} not found.")
        return

    # 2. 逐行读取 / Process line by line
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    total_docs += 1
                    
                    # 定位数据路径: kw -> fused
                    fused_data = data.get('kw', {}).get('fused', {})
                    
                    if not fused_data:
                        continue

                    # 遍历 1 到 9 (MAX_NGRAM)
                    # Loop through "1", "2", ... "9"
                    for n in range(1, MAX_NGRAM + 1):
                        n_key = str(n)
                        keywords_list = fused_data.get(n_key, [])
                        
                        for item in keywords_list:
                            word = item.get('w')
                            score = item.get('s', 0)
                            
                            if word:
                                # 英文清洗: 建议统一转小写进行统计 (虽然你的示例数据已经是小写)
                                # word_clean = word.lower().strip() 
                                word_clean = word # 暂时保持原样，视你的数据情况而定
                                
                                global_weights[word_clean] += score
                                doc_frequency[word_clean] += 1
                                
                                # 记录类型 (优先保留较短的定义，或第一次出现的定义)
                                if word_clean not in ngram_types:
                                    ngram_types[word_clean] = f"{n_key}-gram"
                                
                except json.JSONDecodeError:
                    skipped_lines += 1
                    # print(f"[!] JSON Error at line {line_num+1}")
                    continue
                    
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        return

    print(f"[-] Processed {total_docs} documents. (Skipped {skipped_lines} malformed lines)")
    print("[-] Calculating global scores...")

    # 3. 计算得分 / Calculate Scores
    final_results = []
    
    for word, total_weight in global_weights.items():
        df = doc_frequency[word]
        
        # 过滤噪音 / Filter noise
        if df < MIN_DOC_FREQ:
            continue
            
        # 核心算分公式 / Scoring Formula
        # Score = Total_Weight * log(DF + 1)
        score = total_weight * math.log(df + 1)
        
        final_results.append({
            'keyword': word,
            'type': ngram_types.get(word, 'unknown'),
            'global_score': round(score, 4),
            'doc_freq': df,
            'total_weight': round(total_weight, 4)
        })

    # 4. 排序 / Sort
    final_results.sort(key=lambda x: x['global_score'], reverse=True)

    # 5. 终端预览 / Terminal Preview
    print("\n" + "="*85)
    print(f"{'Rank':<5} {'Keyword (Truncated)':<40} {'Type':<10} {'Score':<10} {'DF':<5}")
    print("="*85)
    for i, res in enumerate(final_results[:20]):
        # 截断显示过长的英文短语
        display_word = res['keyword']
        if len(display_word) > 37: 
            display_word = display_word[:34] + "..."
        
        print(f"{i+1:<5} {display_word:<40} {res['type']:<10} {res['global_score']:<10} {res['doc_freq']:<5}")
    print("="*85)

    # 6. 保存 CSV / Save to CSV
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            headers = ['keyword', 'type', 'global_score', 'doc_freq', 'total_weight']
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(final_results)
            
        print(f"\n[√] Done! Results saved to: {os.path.abspath(OUTPUT_FILE)}")
        print(f"[√] Total keywords extracted: {len(final_results)} (DF >= {MIN_DOC_FREQ})")
        
    except IOError as e:
        print(f"\n[!] File Write Error: {e}")

if __name__ == '__main__':
    main()