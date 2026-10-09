"""Build critic-ranked DPO pairs in PAL's expected format.

For each training example: sample N responses from the stage-1 policy using PAL's
persona-free 'generate_for_dpo' prompt, score each with the critic (probability that
the true profile is identified), and keep the LOWEST scoring one as the rejected
response. PAL's loader keeps the gold response as chosen, so nothing else changes.

Outputs in --out_dir:
  pairs.json                      list of {"prompt", "generated_response", "critic_score"}
  train_origin_dpo_train.txt      the matching data subset (same JSON format as input)

Usage:
  python critic_pairs.py --data data/train_origin.txt --policy <run_dir>/LATEST/policy.pt \
      --critic critic_ckpt --out_dir dpo_data --max_examples 300 --n_samples 4
"""
import json, random, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultipleChoice
from tqdm import tqdm
from critic_train import make_candidates, toks, load_data

GEN_DPO = '''You are a helpful and thorough conversational agent. You have a dialogue history. 
You should provide the next turn of the dialogue based on the dialogue history. 
The dialogue history is: <dialogue context>.
'''


def dpo_prompt(history):
    lines = [("person1: " if e % 2 == 0 else "person2: ") + h for e, h in enumerate(history)]
    which = "person1: " if len(history) == 0 else "person2: "
    return GEN_DPO.replace("<dialogue context>", "'''" + "\n".join(lines) + "'''") + "\n" + which


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--policy", required=True, help="policy.pt from SFT run")
    ap.add_argument("--critic", required=True)
    ap.add_argument("--out_dir", default="dpo_data")
    ap.add_argument("--max_examples", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    raw = json.load(open(a.data, encoding="utf-8"))
    _, profs = load_data(a.data)
    tok_sets = [toks(" ".join(p)) for p in profs]

    # policy
    ptok = AutoTokenizer.from_pretrained("gpt2")
    ptok.pad_token = ptok.eos_token
    ptok.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained("gpt2")
    ckpt = torch.load(a.policy, map_location="cpu")
    state = ckpt["state"] if "state" in ckpt else ckpt
    missing, unexpected = policy.load_state_dict(state, strict=False)
    print(f"policy loaded, missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    policy.to(dev).eval()

    # critic
    ctok = AutoTokenizer.from_pretrained(a.critic)
    critic = AutoModelForMultipleChoice.from_pretrained(a.critic).to(dev).eval()

    def critic_p_true(resp, cands, true_idx):
        enc = ctok([resp] * len(cands), cands, truncation=True, max_length=128,
                   padding="max_length", return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = critic(**{k: v.unsqueeze(0) for k, v in enc.items()}).logits[0]
        return torch.softmax(logits, -1)[true_idx].item()

    idx = list(range(len(raw)))
    rng.shuffle(idx)
    seen, pairs, subset = set(), [], []
    for i in tqdm(idx):
        if len(pairs) >= a.max_examples:
            break
        persona_list, history, response, label = raw[i]
        prompt = dpo_prompt(history)
        if prompt in seen:
            continue
        seen.add(prompt)
        enc = ptok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        if enc["input_ids"].shape[1] > 700:
            continue
        with torch.no_grad():
            out = policy.generate(**enc, do_sample=True, top_p=0.9, max_new_tokens=40,
                                  num_return_sequences=a.n_samples, pad_token_id=ptok.eos_token_id)
        samples = []
        for o in out:
            text = ptok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            text = text.split("\n")[0].strip()
            if text:
                samples.append(text)
        if not samples:
            continue
        cands, true_idx = make_candidates(persona_list, profs, tok_sets, rng)
        scored = [(critic_p_true(s, cands, true_idx), s) for s in samples]
        scored.sort()
        worst_score, worst = scored[0]
        pairs.append({"prompt": prompt, "generated_response": worst, "critic_score": worst_score})
        subset.append(raw[i])

    import os
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(pairs, open(os.path.join(a.out_dir, "pairs.json"), "w", encoding="utf-8"), indent=1)
    json.dump(subset, open(os.path.join(a.out_dir, "train_origin_dpo_train.txt"), "w", encoding="utf-8"))
    print(f"wrote {len(pairs)} pairs to {a.out_dir}")


if __name__ == "__main__":
    main()
