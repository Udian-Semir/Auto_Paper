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
from typing import Optional

from pathlib import Path

import yaml
from openai import OpenAI  # DeepSeek 兼容 OpenAI SDK

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
# 改用 Semantic Scholar API：国内可直连，完整收录 arXiv 论文，免费无需 Key
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,authors,year,publicationDate,externalIds,openAccessPdf,fieldsOfStudy,isOpenAccess"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

HTTP_HEADERS = {
    "User-Agent": "Auto-Paper-Agent/1.0 (https://github.com/Udian-Semir/Auto_Paper; research digest bot)"
}
# 遇到限流时的最大重试次数
MAX_RETRIES = 5




# ── 配置加载 ──────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 论文抓取（Semantic Scholar） ──────────────────────────────────────────────
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

    # 指数退避重试（S2 免费额度偶尔限流 429）
    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                S2_SEARCH_API, params=params, headers=HTTP_HEADERS, timeout=30
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


# ── DeepSeek 概述 ──────────────────────────────────────────────────────────────
def summarize_paper(
    client: OpenAI,
    paper: dict,
    model: str,
    max_tokens: int,
    language: str,
) -> str:
    """使用 DeepSeek 生成论文的简要概述"""
    lang_instruction = "用中文" if language == "zh" else "in English"

    prompt = f"""请{lang_instruction}对以下学术论文进行简要概述，包括：
1. **核心问题**：这篇论文要解决什么问题？
2. **主要方法**：提出了什么方法或技术？
3. **关键结论**：主要发现或贡献是什么？
4. **一句话总结**：用一句话概括这篇论文的意义。

论文标题：{paper['title']}
论文摘要：{paper['abstract']}

请控制在 300 字以内，语言简洁易懂。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
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

    # ── Step 1: 从 Semantic Scholar 抓取论文 ──
    log.info("开始从 Semantic Scholar 抓取论文...")
    all_papers = []
    for query in arxiv_cfg["queries"]:
        papers = fetch_papers(
            query=query,
            categories=arxiv_cfg.get("categories", []),
            max_results=arxiv_cfg.get("max_results_per_query", 5),
            days_back=arxiv_cfg.get("days_back", 1),
        )
        all_papers.extend(papers)
        time.sleep(3)  # 礼貌性延迟，避免 API 限流


    all_papers = deduplicate_papers(all_papers)
    log.info(f"共获取 {len(all_papers)} 篇唯一论文")

    if not all_papers:
        log.warning("今日没有找到符合条件的论文，退出。")
        sys.exit(0)

    # ── Step 2: 使用 DeepSeek 生成概述 ──
    summaries: dict[str, str] = {}

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
