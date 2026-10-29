"""Smoke tests: package imports + repo invariants. Fast by design (<1s, no GPU/data)."""

import importlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SUBPACKAGES = [
    "duet.data",
    "duet.baselines",
    "duet.baselines.pal",
    "duet.baselines.ppt",
    "duet.critic",
    "duet.steering",
    "duet.bidir",
    "duet.training",
    "duet.eval",
]


def test_imports():
    import duet

    assert duet.__version__
    for name in SUBPACKAGES:
        importlib.import_module(name)


def test_project_plan_exists():
    assert (ROOT / "project_plan.md").is_file()


def test_configs_exist():
    for cfg in ["sft", "dpo_entailment", "dpo_critic", "steer", "bidir"]:
        assert (ROOT / "configs" / f"{cfg}.yaml").is_file(), f"missing configs/{cfg}.yaml"
