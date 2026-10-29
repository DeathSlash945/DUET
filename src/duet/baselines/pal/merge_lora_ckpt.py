"""Merge a PAL LoRA checkpoint (train_lora.py output) into a plain policy.pt
that decode.py can load.

Upstream limitation: FSDPTrainer.save's is_lora branch writes an EMPTY
policy.pt (payload `[]`) and saves only the adapter via save_pretrained.
decode.py then does torch.load(evaluate_model_path)['state'] and
model.load_state_dict(...) — which cannot work for LoRA runs. This helper
merges base + adapter and writes `LATEST/policy_merged.pt` in decode.py's
expected format.

Run with the MAIN venv (modern peft), not .venv-pal:

    python -m duet.baselines.pal.merge_lora_ckpt --run-dir outputs/pal_runs/<dpo_run>
"""

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_run(run_dir: Path) -> Path:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    base_name = cfg["model"]["name_or_path"]

    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    adapter = run_dir / "LATEST"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no LoRA adapter in {adapter} — nothing to merge")

    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()

    out = adapter / "policy_merged.pt"
    torch.save({"state": merged.state_dict(), "step_idx": 0, "metrics": {}}, out)
    print(f"merged {base_name} + adapter -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="DPO run dir (contains config.yaml + LATEST/)")
    a = ap.parse_args()
    merge_run(Path(a.run_dir))


if __name__ == "__main__":
    main()
