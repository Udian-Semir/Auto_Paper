import unittest
from unittest.mock import patch

import requests

import paper_agent


def make_paper(paper_id: str, query: str = "topic") -> dict:
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "abstract",
        "authors": ["Author"],
        "published": "2026-08-08",
        "url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
        "tags": ["cs.AI"],
        "query": query,
    }


class SplitIssueBatchesTests(unittest.TestCase):
    def test_splits_without_losing_or_reordering_papers(self):
        papers = [make_paper(str(i)) for i in range(3)]
        summaries = {paper["id"]: "x" * 500 for paper in papers}

        batches = paper_agent.split_issue_batches(papers, summaries, max_chars=1_300)

        flattened = [paper for batch, _ in batches for paper in batch]
        self.assertEqual(flattened, papers)
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(len(body) <= 1_300 for _, body in batches))

    def test_truncates_a_single_oversized_paper(self):
        paper = make_paper("oversized")
        batches = paper_agent.split_issue_batches(
            [paper], {paper["id"]: "x" * 5_000}, max_chars=1_000
        )

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0], [paper])
        self.assertLessEqual(len(batches[0][1]), 1_000)
        self.assertIn("已按 GitHub Issue 正文限制截断", batches[0][1])


class SelectDailyPapersTests(unittest.TestCase):
    def test_limits_count_and_prefers_title_matches_across_topics(self):
        topics = [("RL", ["reinforcement learning"]), ("SLAM", ["SLAM"])]
        papers = [
            {
                **make_paper("rl-abstract", "RL"),
                "title": "A General Method",
                "abstract": "A reinforcement learning method.",
            },
            {
                **make_paper("rl-title", "RL"),
                "title": "Reinforcement Learning for Robots",
            },
            {
                **make_paper("slam-title", "SLAM"),
                "title": "Fast SLAM",
            },
        ]

        selected = paper_agent.select_daily_papers(papers, topics, max_papers=2)

        self.assertEqual({paper["id"] for paper in selected}, {"rl-title", "slam-title"})

    def test_non_positive_limit_keeps_all_papers(self):
        papers = [make_paper("1"), make_paper("2")]
        self.assertEqual(paper_agent.select_daily_papers(papers, [], 0), papers)


class CreateGithubIssueTests(unittest.TestCase):
    @patch("paper_agent.requests.post", side_effect=requests.Timeout("timed out"))
    def test_request_error_returns_none(self, _post):
        result = paper_agent.create_github_issue("owner/repo", "token", "title", "body", [])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
