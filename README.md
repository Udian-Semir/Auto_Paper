# 📄 Auto Paper Agent

> 每天晚上 9 点自动从 arXiv 抓取最新论文，用 DeepSeek AI 生成中文概述，并推送到 GitHub Issues。

[![每日论文推送](https://github.com/Udian-Semir/Auto_Paper/actions/workflows/daily_paper.yml/badge.svg)](https://github.com/Udian-Semir/Auto_Paper/actions/workflows/daily_paper.yml)

---

## ✨ 功能特性

- 🕘 **每晚 9 点自动推送**：GitHub Actions 定时触发，无需手动操作
- 📡 **三级数据源兜底**：arXiv RSS → arXiv API → Semantic Scholar，确保每天都能抓到论文
- 🤖 **AI 中文概述**：调用 DeepSeek 对每篇论文生成核心问题 / 方法 / 结论的中文摘要
- 📌 **GitHub Issue 推送**：每天新建一个 Issue，方便订阅通知和历史查阅
- 🗂️ **本地运行支持**：也可以在 VSCode 本地运行，生成 Markdown 报告

---

## 🔍 当前订阅领域

| 关键词 | 领域说明 |
|---|---|
| `large language model decision making` | 大模型决策 |
| `LLM agent decision` | 大模型智能体决策 |
| `decision making` | 传统决策规划 |
| `SLAM` | 同步定位与建图 |
| `visual pose estimation` | 视觉位姿估计 |
| `camera pose estimation` | 相机位姿估计 |
| `optimization algorithm` | 优化算法 |
| `vision language action` | 视觉-语言-动作模型（VLA） |
| `reinforcement learning` | 强化学习（RL） |
| `world model` | 世界模型 |

覆盖 arXiv 分类：`cs.AI` · `cs.LG` · `cs.CV` · `cs.CL` · `cs.RO` · `math.OC`

## 🎯 RoboMaster 应用关联分析

除了标准的论文概述，本 Agent 还会自动分析每篇论文与 **RoboMaster 机器人竞赛**的关联性：

- **自瞄（auto-aim）场景**：装甲板检测、目标位姿估计、运动预测、云台弹道控制
- **雷达站/哨兵决策场景**：全场态势感知、多目标跟踪、威胁评估、自主决策调度
- **落地难点**：嵌入式/实时约束下的技术挑战

可在 `config.yaml` 中设置 `enable_application_analysis: true/false` 来开启/关闭此功能，也可以修改 `application_context` 来定制你自己的应用场景。


---

## 🚀 快速开始

### 方式一：GitHub Actions（推荐，自动运行）

#### 1. Fork 本仓库

点击右上角 **Fork** → 在你自己的账号下创建一个副本。

#### 2. 申请 DeepSeek API Key

前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并获取 API Key。

#### 3. 配置 GitHub Secrets

在你 Fork 的仓库中：**Settings → Secrets and variables → Actions → New repository secret**

| Secret 名称 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 你的 DeepSeek API Key |

#### 4. 开启写权限

**Settings → Actions → General → Workflow permissions → 选 "Read and write permissions" → Save**

这一步是允许 Actions 自动创建 Issue。

#### 5. 手动触发测试

**Actions → 📄 每日论文推送 → Run workflow**

之后每天晚上 9 点（北京时间）会自动运行并在 Issues 里创建当天的论文推送。

---

### 方式二：本地运行（VSCode）

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 配置环境变量

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

或在项目根目录创建 `.env` 文件（脚本会自动读取）：

```
DEEPSEEK_API_KEY=your_api_key_here
```

#### 运行

```bash
python paper_agent.py
```

报告会保存在 `reports/YYYY-MM-DD.md`，数据备份在 `data/YYYY-MM-DD.json`。

---

## ⚙️ 配置说明

编辑 `config.yaml` 来自定义感兴趣的领域：

```yaml
arxiv:
  queries:
    - "large language models"     # 添加或修改关键词
    - "SLAM"
    - "reinforcement learning"
    # ...

  categories:                     # arXiv 分类过滤
    - "cs.AI"
    - "cs.RO"
    # ...

  max_results_per_query: 4        # 每个关键词最多几篇

deepseek:
  summary_language: "zh"                 # zh=中文, en=英文
  enable_application_analysis: true      # 是否附加 RoboMaster 应用关联分析
  application_context: |                 # 自定义你的应用场景描述
    我的应用场景是 RoboMaster ...
```


---

## 📁 项目结构

```
Auto_Paper/
├── paper_agent.py              # 核心脚本
├── config.yaml                 # 关键词与配置
├── requirements.txt            # Python 依赖
├── .github/
│   └── workflows/
│       └── daily_paper.yml     # GitHub Actions 定时任务
├── reports/                    # 本地运行的 Markdown 报告（自动生成）
└── data/                       # JSON 数据备份（自动生成）
```

---

## 🔄 数据源说明

脚本按优先级依次尝试以下数据源：

1. **arXiv RSS**（首选）— CDN 分发，无限流，国内/GitHub Actions 均可访问
2. **arXiv Atom API**（备用 1）— GitHub Actions 美国服务器可用
3. **Semantic Scholar**（备用 2）— 国内可直连，作为最终兜底

---

## 📬 Issue 示例

每日推送效果如下（在仓库 Issues 页可查看）：

```
📄 每日论文推送 [2026-07-21]

## 🔍 关键词：`SLAM`

### 1. Gaussian Splatting SLAM with Motion Compensation

- 作者：Zhang Wei, Li Fang 等
- 发布时间：2026-07-21
- 分类：`cs.CV` · `cs.RO`
- 链接：[arXiv](...) | [PDF](...)

**📝 AI 概述：**

**第一部分：论文概述**
**核心问题**：...
**主要方法**：...
**关键结论**：...
**一句话总结**：...

**第二部分：应用关联分析**
**与自瞄的关联**：...
**与雷达站/哨兵决策的关联**：...
**落地难点**：...
```


---

## 🛠️ 可选：Semantic Scholar API Key

匿名调用 Semantic Scholar 有速率限制。如需更稳定的访问，可免费申请 API Key：
[https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

申请后在 GitHub Secrets 中添加 `S2_API_KEY`。

---

## 📄 License

MIT License
