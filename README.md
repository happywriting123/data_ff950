# Governance Semantics in AI Policy: Data and Code Repository

This repository contains the replication datasets, data collection pipelines, preprocessing scripts, and network analysis code for the manuscript of Submission ID: `ff950b9e-8eb0-4ce4-88af-af046dcde6c8`.

---

## 1. Repository Structure

```text
├── README.md               # This documentation file
├── data/
│   ├── raw/                # Original scraped raw JSON/text files from portals
│   └── cleaned/            # Structurally cleaned texts free of administrative noise
├── keywords/               # Processed datasets appended with LLM-extracted keyword tokens
└── scripts/
    ├── data_collection/    # Web scrapers and API integration scripts
    ├── preprocessing/      # LLM-based text cleaning and structural parsing scripts
    └── network_analysis/   # Co-occurrence matrix generation and topological metric calculations

```

---

## 2. Data Collection & Search Strategies

### 2.1 Chinese Policy Dataset (China)

* **Sources**: Official web portals of major central government agencies, including:
* The National Development and Reform Commission (`www.ndrc.gov.cn`)
* The State Council of the PRC (`www.gov.cn`)
* The Ministry of Industry and Information Technology (`www.miit.gov.cn`)
* The Ministry of Science and Technology (`www.most.gov.cn`)
* The Cyberspace Administration of China (`www.cac.gov.cn`)


* **Temporal Coverage**: 2016 – 2024
* **Search Strategy**: Focused web crawlers targeted columns such as *News/Press Releases (新闻动态)*, *Policy Documents (政策文件)*, *Policy Interpretations (政策解读)*, *Authoritative Releases (权威发布)*, and *Laws and Regulations (法律法规)*.
* **Query Terms**: `人工智能` (Artificial Intelligence), `智能` (Intelligence/Smart), and `大语言模型` (Large Language Models).
* **Inclusion/Exclusion Criteria**:
* *Inclusion*: Strategic plans, opinions, guiding catalogs, measures, and official interpretations related to AI governance.
* *Exclusion*: Routine administrative notices (e.g., meeting notifications, personal appointments) and purely technical/industrial standards specifications lacking governance narratives.



### 2.2 U.S. Policy Dataset (United States)

* **Source**: The Federal Register API (`www.federalregister.gov`)
* **Temporal Coverage**: 2017 – 2025
* **Search Strategy**: Systematic programmatic queries fetching documents where targeted keywords appeared in the document titles.
* **Query Terms**: `Artificial Intelligence`, `Smart`, and `Large Language Models`.
* **Inclusion/Exclusion Criteria**:
* *Inclusion*: Executive orders, presidential proclamations, agency rules, proposed rules, and notices regarding AI ethics, safety, risk management, and regulatory frameworks.
* *Exclusion (Filtered via LLM Pipeline)*:
1. Purely routine or mundane administrative announcements (e.g., standard agency workshop scheduling, routine health department updates on generic testing methodologies).
2. Documents that mentioned "Artificial Intelligence" merely as a peripheral example or modifier within a broader unrelated context, rather than focusing on AI policy or governance as a core subject.





---

## 3. Data Preprocessing & Keyword Extraction Prompts

Data cleaning and keyword extraction were automated utilizing Large Language Model (LLM) APIs to ensure reproducibility and minimize human subjective bias.

### 3.1 Chinese Text Cleaning Prompt

Executed to structurally clean JSON block arrays by stripping away web layout boilerplate and administrative noise.

```json
System Prompt:
你是结构化文本清洗助手。输入是 JSON 对象的 blocks 数组，元素可能包含：type、text、level、num、order、path。
请在保持正文结构顺序的前提下，删除所有噪音：
1) 网站导航、栏目名、页眉页脚、分享/打印/关闭等按钮文案、版权与免责声明、来源/责任编辑、二维码提示、与正文无关的链接提示。
2) 公文抬头（如“国务院关于…的意见”）、发文字号（如“国发〔2025〕11号”）、收文对象（如“各省、自治区、直辖市…”）。
3) 孤立或无意义的符号行，例如只有“##”、“--”、“***”、“====”或仅由标点/空格组成的块。
4) 重复标题、空白块。
正文应保留：标题（heading/head）、正文段落（para/paragraph）等。
输出严格 JSON：{"blocks":[...]}，不要解释。

User Prompt Template:
请按上述要求清洗并‘语义分层/编号/排序’以下 blocks：
{blocks_json}

输出（严格 JSON）：{"blocks":[{"type":"...","text":"...","level":(1..6可缺省),"num":"可缺省","order":1,"path":"可缺省"}, ...]}

```

### 3.2 U.S. Text Cleaning Prompt

Executed as a rigorous text preprocessing engine to strip Federal Register administrative boilerplate while maintaining the absolute integrity of the narrative text.

```text
System Prompt:
You are a strict text preprocessing engine for NLP tasks. Your goal is to clean US Federal Register documents by removing administrative noise while preserving the original narrative text exactly as is.

### CLEANING RULES
1. REMOVE Boilerplate Header: Delete the entire introductory block starting with "Document Headings" or "Document headings vary by document type".
2. REMOVE Metadata Tags: Delete occurrences of:
   - "printed page [number]" or "(printed page ...)"
   - "Billing Code [code]"
   - "FR Doc. [number]"
   - "Start Printed Page" / "End Printed Page"
3. REMOVE Navigational Noise: Delete text like "Back to top", "View original".
4. REMOVE Redundant Headers: Remove the header section (AGENCY, ACTION, SUMMARY, DATES, ADDRESSES). Start output from 'SUPPLEMENTARY INFORMATION' or the main body.
5. Format Fix: Join lines broken by page numbers.

### NEGATIVE CONSTRAINTS (CRITICAL)
- NO SUMMARIZATION: Output the FULL original content.
- NO PARAPHRASING: Do not change a single word.
- NO MARKDOWN: Just return the raw plain text.
- NO CONVERSATION: Do not say "Here is the cleaned text".

```

### 3.3 Keyword Extraction Prompt

Executed to identify and extract core semantic tokens consistently across both corpora.

```text
System Prompt:
You are an expert policy analyst and keyword extractor.
    
Task: Extract exactly 20 distinct keywords or key phrases from the text below.
    
Requirements:
1. Granularity: Mix of 1-gram, 2-gram, 3-gram, 4-gram, and 5-gram.
2. Content: Focus on specific entities (Agencies, Committees), specific policies (Acts, Orders), technical terms, and core topics. 
3. Quality: Avoid generic verbs or stop words. Prefer nouns and noun phrases.
4. Output Format: Return a JSON object with a single key "keywords" containing a list of strings.
    
Example Output structure:
{
    "keywords": ["Artificial Intelligence", "NIST", "grid capacity", "machine learning safety", "Executive Order 14110"]
}

```

---

## 4. Network Metrics & Topology Formulae

The semantic keyword co-occurrence networks were constructed, computed, and visualized via Python using `NetworkX` for graph-theoretical modeling and `PyVis` for interactive force-directed layouts.

### 4.1 Edge Weighting

The network is modeled as a document-level, undirected, weighted graph. Let $V$ represent the standardized, deduplicated set of keywords. If keyword $i$ and keyword $j$ co-occur within the same policy document $d$, the undirected edge weight $W_{ij}$ increments by 1.

$$W_{ij} = \sum_{d=1}^{N} I(i \in d \land j \in d)$$

Where:

* $I(\cdot)$ is the indicator function (equals 1 if the condition is true, 0 otherwise).
* $N$ is the total number of policy documents in the corpus.

### 4.2 Network Pruning & Connectivity

To isolate structurally salient semantic patterns and eliminate peripheral noise, a multi-stage filtering pipeline was implemented:

* **Document Frequency (DF) Thresholding**: Nodes with a document frequency ($DF$) lower than a designated threshold ($min\_df$) are pruned.
* **Co-occurrence Intensity Filtering**: Edges with a co-occurrence weight $W_{ij} < min\_co$ are removed.
* **Local Sparsification (Top-K Filtering)**: To prevent dense hairball effects, each node is restricted to retain only its top $K$ strongest edges. Following this step, the **Largest Connected Component (LCC)** of the graph is extracted for downstream analysis.

### 4.3 Community Detection

Unsupervised semantic clustering was performed using the Clauset-Newman-Moore **Greedy Modularity Maximization** algorithm. Incorporating edge weights $W_{ij}$, the algorithm optimizes the network's modularity score through an agglomerative hierarchical process, grouping highly dense co-occurring keywords into distinct policy themes (visualized via node color aggregation).

### 4.4 Visualization Scaling Formulae

To mitigate the dominance of highly frequent generic terms (e.g., "Artificial Intelligence") and resolve visual clutter in dense subgraphs, the following mathematical mappings were enforced:

* **Non-linear Node Sizing**: Node radius is scaled exponentially according to its Document Frequency ($DF$) to stabilize visual proportions:
$$\text{Size} = 8.0 + 2.2 \times DF^{0.85}$$


* **Anti-clutter Label Scoring**: To guarantee text legibility, labels are selectively rendered based on an importance metric combining absolute frequency and degree centrality:
$$\text{Score} = 0.7 \times DF + 0.3 \times \text{Degree}$$


*Only nodes ranking in the top percentiles of this score display visible text labels by default.*
* **Physical Layout Simulation**: Node positions are optimized using the ForceAtlas2 force-directed algorithm under the following physics configuration: `Gravity = -30`, `Spring Length = 110`, and `Damping = 0.85`.

```

```
