"""Tortuosity subgroup summaries are paired, diagnostic-only and exhaustive."""

from __future__ import annotations

import csv
import json

import pytest

from layanav.evaluation.benchmark import benchmark, tortuosity_bin


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "direct"),
        (1.249, "direct"),
        (1.25, "moderate_detour"),
        (1.999, "moderate_detour"),
        (2.0, "large_detour"),
        (5.0, "large_detour"),
        (None, None),
    ],
)
def test_tortuosity_bin_threshold(value, expected):
    assert tortuosity_bin(value) == expected


def test_benchmark_subgroups_are_exhaustive_and_paired(tmp_path):
    report = benchmark(
        methods=("astar", "apf"), families=("sparse", "dense"),
        topologies=("maze", "corridor"), episodes=2,
        eval_seed=42000, size=12, radius=2, movement_mode=8,
        sensor="raycast", max_steps=2, oracle_diagnostic=True,
        save_failures=False, bootstrap_replicates=100, output_dir=tmp_path,
    )
    assert report["protocol"]["structured_map_generator_version"] == 2
    assert report["protocol"]["tortuosity_bins_are_heuristic"]

    with (tmp_path / "tortuosity_groups.csv").open(encoding="utf-8") as f:
        strata = list(csv.DictReader(f))
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as f:
        episodes = list(csv.DictReader(f))
    assert len(strata) == 2 * 2 * 2 * 3
    for row in episodes:
        assert row["tortuosity_bin"] in (
            "direct", "moderate_detour", "large_detour"
        )
        assert float(row["mean_path_clearance"]) >= float(row["minimum_clearance"])
        same = [
            other for other in episodes
            if other["seed"] == row["seed"] and other["topology"] == row["topology"]
            and other["family"] == row["family"]
        ]
        assert len(same) == 2
        assert len({item["map_sha256"] for item in same}) == 1
        assert len({item["tortuosity_bin"] for item in same}) == 1

    for key, summary in report["summary"].items():
        groups = summary["tortuosity_groups"]
        assert sum(g["episodes"] for g in groups.values()) == 2, key
        for band, group in groups.items():
            if group["episodes"]:
                assert group["success"]["n"] == group["episodes"]
                assert group["spl"]["n"] == group["episodes"]
            else:
                assert group["success"] is group["spl"] is None
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]


def test_no_oracle_no_stratification_or_truth_access(tmp_path):
    report = benchmark(
        methods=("astar",), families=("sparse",),
        episodes=1, eval_seed=123, size=9, radius=2,
        sensor="circle", max_steps=25, oracle_diagnostic=False,
        save_failures=False, bootstrap_replicates=100, output_dir=tmp_path,
    )
    assert report["summary"]["sparse/astar"]["tortuosity_groups"] == {}
    assert not (tmp_path / "tortuosity_groups.csv").exists()
