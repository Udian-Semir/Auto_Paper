#!/usr/bin/env python3


"""
Auto Paper Agent
每日从 arXiv 抓取感兴趣的论文，使用 DeepSeek 生成中文摘要，
并自动创建 GitHub Issue 推送。
"""

import os
import sys
import json
import time
import logging
import datetime
import requests
import xml.etree.ElementTree as ET
from typing import Optional


from pathlib import Path

import yaml
from openai import OpenAI  # DeepSeek 兼容 OpenAI SDK

# 本地运行时自动加载 .env 文件（GitHub Actions 环境无需，缺失也不影响）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
# arXiv RSS 订阅源：由 CDN 分发，不受 API 限流影响，天然是"当天新论文"
ARXIV_RSS = "https://rss.arxiv.org/rss"
# arXiv Atom API：作为 RSS 的补充/备用
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

# Semantic Scholar：国内可直连，作为 arXiv 失败时的备用数据源
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,authors,year,publicationDate,externalIds,openAccessPdf,fieldsOfStudy,isOpenAccess"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Papers with Code：通过 arXiv ID 反查论文的开源代码仓库
PWC_PAPER_API = "https://paperswithcode.com/api/v1/papers/"



HTTP_HEADERS = {
    "User-Agent": "Auto-Paper-Agent/1.0 (https://github.com/Udian-Semir/Auto_Paper; research digest bot)"
}
# 遇到限流时的最大重试次数
MAX_RETRIES = 5




# ── 配置加载 ──────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 论文抓取（arXiv RSS，主力数据源，无限流）────────────────────────────────
def fetch_rss_papers(
    categories: list[str],
    keywords: list[str],
    max_results_per_query: int,
) -> Optional[list[dict]]:
    """
    通过 arXiv RSS 订阅源获取当天新论文，本地按关键词过滤。
    RSS 由 CDN 缓存分发，不受 API 限流，GitHub Actions 必定可访问。
    返回 None 表示所有 category RSS 均请求失败。
    """
    # 合并所有分类为一条 RSS 请求（arXiv 支持 cat1+cat2 语法）
    cats_str = "+".join(categories) if categories else "cs.AI+cs.LG+cs.CV+cs.CL"
    rss_url = f"{ARXIV_RSS}/{cats_str}"
    log.info(f"[RSS] 获取 arXiv RSS: {rss_url}")

    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(rss_url, headers=HTTP_HEADERS, timeout=30)
            if resp.status_code == 200:
                break
            wait = min(20, 5 * attempt)
            log.warning(f"[RSS] HTTP {resp.status_code}，{wait}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
        except requests.RequestException as e:
            wait = min(20, 5 * attempt)
            log.warning(f"[RSS] 请求异常: {e}，{wait}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    else:
        log.warning("[RSS] arXiv RSS 请求失败，将尝试备用数据源。")
        return None

    if resp is None or resp.status_code != 200:
        return None

    # 解析 RSS XML（arXiv RSS 2.0 + Dublin Core namespace）
    # 注册命名空间避免 ET 输出 ns0: 前缀
    DC_NS = "http://purl.org/dc/elements/1.1/"
    ARXIV_TERMS_NS = "http://arxiv.org/schemas/atom"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        log.warning(f"[RSS] XML 解析失败: {e}")
        return None

    channel = root.find("channel")
    if channel is None:
        log.warning("[RSS] RSS 格式异常，未找到 channel 节点。")
        return None

    items = channel.findall("item")
    log.info(f"[RSS] 获取到 {len(items)} 篇今日论文，开始关键词过滤...")

    # 关键词匹配（不区分大小写，title 或 description/abstract 中命中即可）
    keywords_lower = [kw.lower() for kw in keywords]

    # 按关键词分组收集，保持与旧数据结构兼容
    query_papers: dict[str, list[dict]] = {kw: [] for kw in keywords}

    for item in items:
        title = (item.findtext("title") or "").strip()
        # arXiv RSS title 格式: "Paper Title (arXiv:2407.xxxxx [cs.AI])"
        # 去掉末尾的 arXiv ID 标注
        import re as _re
        title_clean = _re.sub(r"\s*\(arXiv:\S+\)\s*$", "", title).strip()

        abstract = (item.findtext("description") or "").strip()
        # description 可能包含 HTML，简单去标签
        abstract_clean = _re.sub(r"<[^>]+>", "", abstract).strip().replace("\n", " ")

        link = (item.findtext("link") or "").strip()
        paper_id = link.split("/abs/")[-1] if "/abs/" in link else ""
        if not paper_id:
            continue

        # 作者（Dublin Core）
        creator = item.findtext(f"{{{DC_NS}}}creator") or ""
        authors = [a.strip() for a in creator.split(",")][:5]

        # 分类 tags
        tags = [c.text for c in item.findall("category") if c.text]

        today_str = datetime.date.today().strftime("%Y-%m-%d")

        paper = {
            "id": paper_id,
            "title": title_clean,
            "abstract": abstract_clean,
            "authors": authors,
            "published": today_str,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
            "tags": tags,
            "query": "",  # 后面按关键词匹配后填写
        }

        text_to_search = (title_clean + " " + abstract_clean).lower()

        # 将论文归入匹配的关键词分组
        matched = False
        for kw in keywords:
            if kw.lower() in text_to_search:
                if len(query_papers[kw]) < max_results_per_query:
                    p = dict(paper)
                    p["query"] = kw
                    query_papers[kw].append(p)
                    matched = True
                    break  # 每篇论文只归入第一个匹配的关键词

    all_matched = []
    for kw in keywords:
        cnt = len(query_papers[kw])
        log.info(f"[RSS] 关键词 '{kw}' 匹配到 {cnt} 篇")
        all_matched.extend(query_papers[kw])

    return all_matched


# ── 论文抓取（arXiv Atom API，备用数据源 1）─────────────────────────────────
def build_arxiv_query(query: str, categories: list[str]) -> str:

    q = f'all:"{query}"'
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        q = f"({q}) AND ({cat_filter})"
    return q


def fetch_arxiv_papers(
    query: str,
    categories: list[str],
    max_results: int,
    days_back: int,
) -> Optional[list[dict]]:
    """从 arXiv API 获取最近论文。返回 None 表示请求失败（触发备用源）。"""
    params = {
        "search_query": build_arxiv_query(query, categories),
        "start": 0,
        "max_results": max_results * 3,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                ARXIV_API, params=params, headers=HTTP_HEADERS, timeout=30
            )
            if resp.status_code == 429:
                wait = min(30, 3 * attempt)
                log.warning(f"arXiv 429 限流，{wait}s 后重试 ({attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            wait = min(30, 3 * attempt)
            log.warning(f"arXiv 请求异常: {e}，{wait}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    else:
        log.warning(f"关键词 '{query}' 从 arXiv 获取失败。")
        return None

    if resp is None or resp.status_code != 200:
        return None

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=days_back
    )
    papers = []
    root = ET.fromstring(resp.text)
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        published_text = entry.findtext(f"{{{ARXIV_NS}}}published", "")
        try:
            published = datetime.datetime.fromisoformat(
                published_text.replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if published < cutoff:
            continue

        id_raw = entry.findtext(f"{{{ARXIV_NS}}}id", "")
        paper_id = id_raw.split("/abs/")[-1] if "/abs/" in id_raw else id_raw
        title = entry.findtext(f"{{{ARXIV_NS}}}title", "").strip().replace("\n", " ")
        abstract = entry.findtext(f"{{{ARXIV_NS}}}summary", "").strip().replace("\n", " ")
        authors = [
            a.findtext(f"{{{ARXIV_NS}}}name", "")
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
        ][:5]
        tags = [t.get("term", "") for t in entry.findall(f"{{{ARXIV_NS}}}category")]

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published.strftime("%Y-%m-%d"),
                "url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
                "tags": tags,
                "query": query,
            }
        )
        if len(papers) >= max_results:
            break

    log.info(f"[arXiv] 关键词 '{query}' 获取到 {len(papers)} 篇论文")
    return papers


# ── 论文抓取（Semantic Scholar，备用数据源） ─────────────────────────────────
def _parse_date(paper: dict) -> Optional[datetime.date]:

    """从 S2 返回中解析发布日期"""
    date_str = paper.get("publicationDate")
    if date_str:
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            pass
    # 退化：只有年份时按当年 1 月 1 日算
    year = paper.get("year")
    if year:
        return datetime.date(int(year), 1, 1)
    return None


def fetch_papers(
    query: str,
    categories: list[str],
    max_results: int,
    days_back: int,
) -> list[dict]:
    """从 Semantic Scholar API 搜索最近的论文（国内可直连）"""
    params = {
        "query": query,
        "limit": min(max_results * 5, 100),  # 多取一些，后面按时间过滤
        "fields": S2_FIELDS,
        "sort": "publicationDate:desc",
    }

    # 可选：Semantic Scholar API Key（免费申请，大幅提升额度，解决匿名 429）
    # 申请地址: https://www.semanticscholar.org/product/api#api-key
    headers = dict(HTTP_HEADERS)
    s2_key = os.environ.get("S2_API_KEY", "")
    if s2_key:
        headers["x-api-key"] = s2_key

    # 指数退避重试（匿名调用共享全球限流池，容易 429）
    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                S2_SEARCH_API, params=params, headers=headers, timeout=30
            )

            if resp.status_code == 429:
                wait = min(60, 5 * attempt)
                log.warning(
                    f"Semantic Scholar 返回 429 限流，{wait}s 后重试 ({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            wait = min(60, 5 * attempt)
            log.warning(f"S2 请求异常: {e}，{wait}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    else:
        log.error(f"关键词 '{query}' 多次重试后仍失败，跳过。")
        return []

    if resp is None or resp.status_code != 200:
        log.error(f"关键词 '{query}' 请求失败，跳过。")
        return []

    data = resp.json().get("data", []) or []
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)

    papers = []
    for item in data:
        pub_date = _parse_date(item)
        if pub_date is None or pub_date < cutoff:
            continue

        abstract = (item.get("abstract") or "").strip().replace("\n", " ")
        if not abstract:
            continue  # 没摘要的没法概述，跳过

        # arXiv ID（如果有），用于生成 arXiv 链接
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        s2_id = item.get("paperId", "")
        paper_id = arxiv_id or s2_id

        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        else:
            url = f"https://www.semanticscholar.org/paper/{s2_id}"
            oa = item.get("openAccessPdf") or {}
            pdf_url = oa.get("url", url)

        authors = [a.get("name", "") for a in (item.get("authors") or [])][:5]
        tags = item.get("fieldsOfStudy") or []

        papers.append(
            {
                "id": paper_id,
                "title": (item.get("title") or "").strip().replace("\n", " "),
                "abstract": abstract,
                "authors": authors,
                "published": pub_date.strftime("%Y-%m-%d"),
                "url": url,
                "pdf_url": pdf_url,
                "tags": tags,
                "query": query,
            }
        )

        if len(papers) >= max_results:
            break

    log.info(f"关键词 '{query}' 获取到 {len(papers)} 篇论文")
    return papers



def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """按论文 ID 去重"""
    seen = set()
    unique = []
    for p in papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    return unique


# ── 开源代码仓库检索 ──────────────────────────────────────────────────────────
def _extract_repos_from_abstract(abstract: str) -> list[str]:
    """从摘要正文里正则提取 GitHub / GitLab 仓库链接（作者常在摘要放出仓库）。"""
    import re as _re

    if not abstract:
        return []
    # 匹配 github.com/owner/repo 或 gitlab.com/owner/repo
    pattern = _re.compile(
        r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        _re.IGNORECASE,
    )
    repos = []
    for m in pattern.findall(abstract):
        # 清理结尾的标点
        url = m.rstrip(".,;:)")
        # 去掉常见非仓库路径（如 github.com/blog）
        if url not in repos:
            repos.append(url)
    return repos


def fetch_code_repos(paper: dict) -> list[str]:
    """
    获取论文的开源代码仓库链接。
    优先用 Papers with Code（按 arXiv ID 反查官方实现），
    失败或无结果则用摘要正则兜底。
    返回去重后的仓库 URL 列表（最多 3 个）。
    """
    repos: list[str] = []

    # 只有 arXiv 论文才能用 PwC 按 arXiv ID 查询
    paper_id = paper.get("id", "")
    is_arxiv = bool(paper_id) and paper.get("url", "").startswith("https://arxiv.org")

    if is_arxiv:
        # arXiv ID 去掉版本号后缀（如 2407.01234v2 -> 2407.01234）
        import re as _re
        arxiv_id = _re.sub(r"v\d+$", "", paper_id)
        url = f"{PWC_PAPER_API}arxiv/{arxiv_id}/repositories/"
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", []) if isinstance(data, dict) else []
                # 官方实现优先，其次按 star 数排序
                results.sort(
                    key=lambda r: (not r.get("is_official", False), -(r.get("stars") or 0))
                )
                for r in results:
                    repo_url = r.get("url", "")
                    if repo_url and repo_url not in repos:
                        repos.append(repo_url)
                    if len(repos) >= 3:
                        break
            elif resp.status_code != 404:
                log.info(f"[PwC] {arxiv_id} 返回 HTTP {resp.status_code}")
        except requests.RequestException as e:
            log.info(f"[PwC] {arxiv_id} 查询异常: {e}")

    # 兜底：从摘要里提取仓库链接
    if not repos:
        repos = _extract_repos_from_abstract(paper.get("abstract", ""))[:3]

    return repos



# ── DeepSeek 概述 ──────────────────────────────────────────────────────────────
def summarize_paper(
    client: OpenAI,
    paper: dict,
    model: str,
    max_tokens: int,
    language: str,
    application_context: str = "",
) -> str:
    """使用 DeepSeek 生成论文的简要概述，可选附带 RoboMaster 应用关联分析。"""
    lang_instruction = "用中文" if language == "zh" else "in English"

    app_section = ""
    if application_context:
        app_section = f"""

---

**第二部分：应用关联分析**

背景：
{application_context.strip()}

请分析：
5. **与自瞄的关联**：该论文的技术能否迁移到装甲板检测、目标位姿估计或运动预测？有哪些可改进点？
6. **与雷达站/哨兵决策的关联**：该论文的技术能否改善全场态势感知、威胁评估或自主决策调度？有哪些可改进点？
7. **落地难点**：在 RoboMaster 嵌入式/实时约束下，该技术落地的主要挑战是什么？

如果该论文与上述场景关联性很低，请直接说明"与 RoboMaster 关联性较低"并简要说明原因。"""

    prompt = f"""请{lang_instruction}对以下学术论文进行分析。

**第一部分：论文概述**（控制在 250 字以内）
1. **核心问题**：这篇论文要解决什么问题？
2. **主要方法**：提出了什么方法或技术？
3. **关键结论**：主要发现或贡献是什么？
4. **一句话总结**：用一句话概括这篇论文的意义。

论文标题：{paper['title']}
论文摘要：{paper['abstract']}
{app_section}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        choice = response.choices[0]
        content = choice.message.content.strip()
        # 若因 token 上限被截断，明确提示用户调大 max_summary_tokens
        if getattr(choice, "finish_reason", None) == "length":
            log.warning(
                f"概述被 token 上限截断 ({paper['id']})，"
                f"建议在 config.yaml 调大 max_summary_tokens（当前 {max_tokens}）。"
            )
            content += (
                "\n\n> ⚠️ *（本段因达到 token 上限被截断，"
                "可在 config.yaml 调大 `max_summary_tokens` 获取完整分析）*"
            )
        return content
    except Exception as e:

        log.warning(f"DeepSeek 概述失败 ({paper['id']}): {e}")
        # 降级：直接使用原始摘要
        return f"**摘要（原文）：** {paper['abstract'][:500]}..."



# ── GitHub Issues 发布 ────────────────────────────────────────────────────────
def ensure_labels(repo: str, token: str, labels: list[str]):
    """确保 GitHub Issue 标签存在，不存在则创建"""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    label_colors = {"daily-papers": "0075ca", "auto-generated": "e4e669"}

    for label in labels:
        url = f"https://api.github.com/repos/{repo}/labels/{label}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 404:
            requests.post(
                f"https://api.github.com/repos/{repo}/labels",
                headers=headers,
                json={"name": label, "color": label_colors.get(label, "ededed")},
            )


def build_issue_body(papers: list[dict], summaries: dict[str, str]) -> str:
    """构建 GitHub Issue 的正文内容"""
    today = datetime.date.today().strftime("%Y年%m月%d日")
    lines = [
        f"# 📄 每日论文推送 - {today}",
        "",
        f"> 今日共推送 **{len(papers)}** 篇论文，由 [Auto Paper Agent](https://github.com) 自动生成。",
        "",
        "---",
        "",
    ]

    # 按搜索关键词分组
    groups: dict[str, list[dict]] = {}
    for p in papers:
        groups.setdefault(p["query"], []).append(p)

    for query, group in groups.items():
        lines.append(f"## 🔍 关键词：`{query}`")
        lines.append("")
        for i, paper in enumerate(group, 1):
            summary = summaries.get(paper["id"], "概述生成失败")
            authors_str = ", ".join(paper["authors"])
            if len(paper["authors"]) == 5:
                authors_str += " 等"
            tags_str = " · ".join(f"`{t}`" for t in paper["tags"][:4])

            lines += [
                f"### {i}. {paper['title']}",
                "",
                f"- **作者**：{authors_str}",
                f"- **发布时间**：{paper['published']}",
                f"- **分类**：{tags_str}",
                f"- **链接**：[arXiv]({paper['url']}) | [PDF]({paper['pdf_url']})",
            ]

            # 开源代码仓库（若检索到），方便一键 star
            repos = paper.get("code_repos", [])
            if repos:
                repo_links = " | ".join(f"[{r.rstrip('/').split('/')[-1]}]({r})" for r in repos)
                lines.append(f"- **💻 开源代码**：{repo_links}")

            lines += [
                "",
                "**📝 AI 概述：**",
                "",
                summary,
                "",
                "---",
                "",
            ]

    lines += [
        "",
        "<sub>🤖 由 [Auto Paper Agent](https://github.com/Udian-Semir/Auto_Paper) 自动生成 · "
        f"数据来源：[arXiv](https://arxiv.org) · 概述模型：DeepSeek</sub>",
    ]

    return "\n".join(lines)



def create_github_issue(
    repo: str,
    token: str,
    title: str,
    body: str,
    labels: list[str],
) -> Optional[str]:
    """在 GitHub 仓库创建 Issue，返回 Issue URL"""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"title": title, "body": body, "labels": labels}

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if resp.status_code == 201:
        issue_url = resp.json().get("html_url", "")
        log.info(f"GitHub Issue 创建成功: {issue_url}")
        return issue_url
    else:
        log.error(f"GitHub Issue 创建失败: {resp.status_code} - {resp.text}")
        return None


# ── 本地输出（VSCode 模式） ───────────────────────────────────────────────────
def save_local_report(papers: list[dict], summaries: dict[str, str]):
    """保存为本地 Markdown 文件"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{today}.md"

    body = build_issue_body(papers, summaries)
    output_path.write_text(body, encoding="utf-8")
    log.info(f"本地报告已保存: {output_path}")
    return str(output_path)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 加载配置
    config = load_config("config.yaml")
    arxiv_cfg = config["arxiv"]
    ds_cfg = config["deepseek"]
    gh_cfg = config["github"]

    # 读取环境变量中的 API Keys
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    github_repo = os.environ.get("GITHUB_REPO", "")  # 格式: owner/repo

    # 运行模式：local（本地）或 github（GitHub Actions）
    run_mode = os.environ.get("RUN_MODE", "local")

    log.info(f"运行模式: {run_mode}")

    # ── Step 1: 抓取论文（三级数据源，依次降级）──
    # 优先级: arXiv RSS（无限流）> arXiv Atom API > Semantic Scholar
    categories = arxiv_cfg.get("categories", [])
    max_results = arxiv_cfg.get("max_results_per_query", 5)
    days_back = arxiv_cfg.get("days_back", 1)
    keywords = arxiv_cfg["queries"]

    log.info("开始抓取论文（数据源优先级：arXiv RSS > arXiv API > Semantic Scholar）...")

    # 1️⃣ 首选：arXiv RSS（CDN 分发，无限流，GitHub Actions / 国内均可）
    all_papers = fetch_rss_papers(categories, keywords, max_results)

    if all_papers is None:
        # 2️⃣ 备用 1：arXiv Atom API（GitHub Actions 美国服务器可用）
        log.warning("RSS 不可用，尝试 arXiv Atom API...")
        all_papers = []
        for query in keywords:
            result = fetch_arxiv_papers(query, categories, max_results, days_back)
            if result is None:
                all_papers = None  # 标记完全失败
                break
            all_papers.extend(result)
            time.sleep(2)

    if all_papers is None:
        # 3️⃣ 备用 2：Semantic Scholar（国内 IP 可用）
        log.warning("arXiv API 也不可用，切换到 Semantic Scholar...")
        all_papers = []
        for query in keywords:
            papers = fetch_papers(query, categories, max_results, days_back)
            all_papers.extend(papers)
            time.sleep(3)




    all_papers = deduplicate_papers(all_papers)
    log.info(f"共获取 {len(all_papers)} 篇唯一论文")

    if not all_papers:
        log.warning("今日没有找到符合条件的论文，退出。")
        sys.exit(0)

    # ── Step 1.5: 检索每篇论文的开源代码仓库（方便一键 star）──
    log.info("开始检索论文开源代码仓库（Papers with Code）...")
    for i, paper in enumerate(all_papers):
        repos = fetch_code_repos(paper)
        paper["code_repos"] = repos
        if repos:
            log.info(f"  [{i+1}/{len(all_papers)}] 找到 {len(repos)} 个仓库: {paper['title'][:40]}")
        time.sleep(0.5)  # 轻微限速，友好访问 PwC


    # ── Step 2: 使用 DeepSeek 生成概述 ──
    summaries: dict[str, str] = {}

    # 是否启用 RoboMaster 应用关联分析
    app_ctx = ""
    if ds_cfg.get("enable_application_analysis", False):
        app_ctx = ds_cfg.get("application_context", "")

    if deepseek_api_key:
        log.info("开始使用 DeepSeek 生成论文概述...")
        client = OpenAI(api_key=deepseek_api_key, base_url=DEEPSEEK_BASE_URL)

        for i, paper in enumerate(all_papers):
            log.info(f"概述进度: {i+1}/{len(all_papers)} - {paper['title'][:50]}...")
            summaries[paper["id"]] = summarize_paper(
                client=client,
                paper=paper,
                model=ds_cfg.get("model", "deepseek-chat"),
                max_tokens=ds_cfg.get("max_summary_tokens", 500),
                language=ds_cfg.get("summary_language", "zh"),
                application_context=app_ctx,
            )
            time.sleep(1)  # 避免 API 限流

    else:
        log.warning("未设置 DEEPSEEK_API_KEY，将使用原始摘要（截断）")
        for paper in all_papers:
            summaries[paper["id"]] = f"**摘要：** {paper['abstract'][:400]}..."

    # ── Step 3: 推送/保存结果 ──
    today = datetime.date.today().strftime("%Y-%m-%d")
    issue_title = f"{gh_cfg.get('issue_title_prefix', '📄 每日论文推送')} [{today}]"

    if run_mode == "github" and github_token and github_repo:
        # GitHub Actions 模式：创建 Issue
        ensure_labels(github_repo, github_token, gh_cfg.get("issue_labels", []))
        body = build_issue_body(all_papers, summaries)
        issue_url = create_github_issue(
            repo=github_repo,
            token=github_token,
            title=issue_title,
            body=body,
            labels=gh_cfg.get("issue_labels", []),
        )
        if issue_url:
            print(f"✅ Issue 创建成功: {issue_url}")
        else:
            sys.exit(1)
    else:
        # 本地模式：保存为 Markdown 文件
        report_path = save_local_report(all_papers, summaries)
        print(f"✅ 报告已保存到: {report_path}")

    # 同时保存一份 JSON 数据备份
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    json_path = data_dir / f"{today}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"date": today, "count": len(all_papers), "papers": all_papers},
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info(f"数据备份已保存: {json_path}")


if __name__ == "__main__":
    main()
