# 📄 Auto Paper Agent

每天自动从 arXiv 抓取你感兴趣的论文，用 DeepSeek 生成中文概述，并推送到 GitHub Issues。

![GitHub Actions](https://github.com/Udian-Semir/Auto_Paper/actions/workflows/daily_paper.yml/badge.svg)

---

## ✨ 功能

- 🔍 **按关键词 + 分类** 每日从 arXiv 抓取最新论文
- 🤖 **DeepSeek AI 概述** 自动生成中文摘要（核心问题 / 方法 / 结论）
- 📬 **GitHub Issues 推送** 每天自动在仓库创建一条 Issue，方便浏览和订阅
- 💾 **本地模式** 也可在 VSCode 本地运行，生成 Markdown 报告
- ⚙️ **完全可配置** 修改 `config.yaml` 即可定制关键词和领域

## 效果预览

每天会自动创建类似这样的 Issue：

> **📄 每日论文推送 [2026-07-21]**
>
> 今日共推送 **12** 篇论文
>
> ### 🔍 关键词：`large language models`
> #### 1. Paper Title Here
> - **作者**：Author A, Author B 等
> - **链接**：[arXiv](#) | [PDF](#)
>
> **📝 AI 概述：**
> 1. **核心问题**：...
> 2. **主要方法**：...
> 3. **关键结论**：...
> 4. **一句话总结**：...

---

## 🚀 快速开始

### 方式一：GitHub Actions（推荐，全自动）

**第一步：Fork 或 Clone 本仓库**

**第二步：添加 Secret**

进入仓库 → **Settings → Secrets and variables → Actions → New repository secret**，添加：

| Secret 名称 | 值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | `sk-xxxxxxxx` | [DeepSeek 控制台](https://platform.deepseek.com/api_keys) 获取 |

> `GITHUB_TOKEN` 已由 GitHub Actions 自动提供，无需手动添加。

**第三步：确认 Actions 权限**

进入仓库 → **Settings → Actions → General → Workflow permissions**，选择 **Read and write permissions**。

**第四步：自定义关键词**

编辑 `config.yaml`，修改你感兴趣的研究方向：

```yaml
arxiv:
  queries:
    - "large language models"
    - "your research topic"
  categories:
    - "cs.AI"
    - "cs.LG"
```

**第五步：手动触发测试**

进入仓库 → **Actions → 📄 每日论文推送 → Run workflow**，验证是否正常运行。

之后每天北京时间 **09:00** 自动运行，新 Issue 会出现在仓库的 Issues 列表中。

---

### 方式二：本地运行（VSCode）

```bash
# 1. 克隆仓库
git clone https://github.com/Udian-Semir/Auto_Paper.git
cd Auto_Paper

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量（可选，没有则使用原始摘要）
export DEEPSEEK_API_KEY="sk-your-key-here"

# 4. 运行（本地模式默认保存 Markdown 到 reports/ 目录）
python paper_agent.py
```

报告会保存在 `reports/YYYY-MM-DD.md`，用 VSCode Markdown 预览即可查看。

---

## ⚙️ 配置说明

编辑 `config.yaml` 来定制推送内容：

```yaml
arxiv:
  queries:              # 搜索关键词列表
    - "diffusion models"
    - "reinforcement learning"
  categories:           # arXiv 分类过滤（空列表 = 不过滤）
    - "cs.AI"
    - "cs.LG"
  max_results_per_query: 5   # 每个关键词最多几篇
  days_back: 1               # 只看最近几天的论文

deepseek:
  model: "deepseek-chat"     # 模型版本
  summary_language: "zh"     # zh=中文, en=英文
  max_summary_tokens: 500

github:
  issue_title_prefix: "📄 每日论文推送"
  issue_labels:
    - "daily-papers"
    - "auto-generated"
```

### 常用 arXiv 分类

| 分类 | 领域 |
|---|---|
| `cs.AI` | 人工智能 |
| `cs.LG` | 机器学习 |
| `cs.CV` | 计算机视觉 |
| `cs.CL` | 计算语言学 / NLP |
| `cs.RO` | 机器人 |
| `stat.ML` | 统计机器学习 |
| `cs.IR` | 信息检索 |
| `cs.NE` | 神经网络 |

---

## 📁 项目结构

```
Auto_Paper/
├── paper_agent.py              # 核心脚本
├── config.yaml                 # 配置文件（修改此处定制）
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── daily_paper.yml     # GitHub Actions 工作流
├── reports/                    # 本地运行生成的 Markdown 报告
│   └── 2026-07-21.md
└── data/                       # 论文数据 JSON 备份
    └── 2026-07-21.json
```

---

## 🔧 进阶用法

**手动触发并指定天数范围：**

在 GitHub Actions 手动触发时，可以输入 `days_back=7` 来一次性抓取过去 7 天的论文。

**订阅通知：**

在 GitHub 仓库页面，点击右上角 **Watch → Custom → Issues**，每次新 Issue 创建时你会收到邮件通知。

---

## 📜 License

MIT
