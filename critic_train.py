"""DUET persona critic.
Given a response and 5 candidates (true profile, 3 lexically similar distractors,
and a fixed 'no matching profile' slot), predict which one produced the response.
Copies of profile sentences and generic replies are trained to map to 'no match'.

Usage:
  python critic_train.py --data data/train_origin.txt --out critic_ckpt --n_train 8000
"""
import json, random, re, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm

NO_MATCH = "no matching profile"
N_CAND = 5  # 4 profiles + NO_MATCH
GENERIC = [
    "That is interesting, tell me more.",
    "I see. What else is going on with you?",
    "That sounds great! How is your day going?",
    "Wow, that is cool. I like talking with you.",
    "Oh really? That is nice to hear.",
    "I am not sure what to say, but I am glad we are chatting.",
    "Haha yes. Anyway, what do you like to do?",
    "Nice! I hope you are having a good day.",
]


def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def load_data(path):
    data = json.load(open(path, encoding="utf-8"))
    items, profs, seen = [], [], set()
    for persona_list, history, response, label in data:
        key = " ".join(persona_list)
        if key not in seen:
            seen.add(key)
            profs.append(persona_list)
        items.append((persona_list, response))
    return items, profs


def make_candidates(true_list, profs, tok_sets, rng, k=3, pool=300):
    """Return (candidate_texts, index_of_true). Distractors are the k pool profiles
    with highest word overlap with the true profile (hard negatives)."""
    true_text = " ".join(true_list)
    tt = toks(true_text)
    sample = rng.sample(range(len(profs)), min(pool, len(profs)))
    scored = []
    for i in sample:
        if " ".join(profs[i]) == true_text:
            continue
        inter = len(tt & tok_sets[i])
        union = len(tt | tok_sets[i]) or 1
        scored.append((inter / union, i))
    scored.sort(reverse=True)
    dis = [" ".join(profs[i]) for _, i in scored[:k]]
    cands = [true_text] + dis
    order = list(range(len(cands)))
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    true_idx = order.index(0)
    return cands + [NO_MATCH], true_idx


def build_example(persona_list, response, profs, tok_sets, rng):
    r = rng.random()
    kind = "normal"
    if r < 0.15:
        response, kind = rng.choice(persona_list), "copy"
    elif r < 0.25:
        response, kind = rng.choice(GENERIC), "generic"
    cands, true_idx = make_candidates(persona_list, profs, tok_sets, rng)
    label = N_CAND - 1 if kind in ("copy", "generic") else true_idx
    return response, cands, label, kind


class CriticData(Dataset):
    def __init__(self, items, profs, tok_sets, tokenizer, n, seed, max_len=128):
        rng = random.Random(seed)
        self.ex = []
        for persona_list, response in rng.sample(items, min(n, len(items))):
            self.ex.append(build_example(persona_list, response, profs, tok_sets, rng))
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        resp, cands, label, kind = self.ex[i]
        enc = self.tok([resp] * len(cands), cands, truncation=True,
                       max_length=self.max_len, padding="max_length", return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], label, kind


def collate(b):
    ids = torch.stack([x[0] for x in b])
    mask = torch.stack([x[1] for x in b])
    lab = torch.tensor([x[2] for x in b])
    return ids, mask, lab, [x[3] for x in b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="critic_ckpt")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    items, profs = load_data(a.data)
    # split by profile so validation profiles are unseen
    rng = random.Random(a.seed)
    keys = sorted({" ".join(p) for p in profs})
    rng.shuffle(keys)
    val_keys = set(keys[: max(1, len(keys) // 10)])
    tr_items = [x for x in items if " ".join(x[0]) not in val_keys]
    va_items = [x for x in items if " ".join(x[0]) in val_keys]
    tr_profs = [p for p in profs if " ".join(p) not in val_keys]
    va_profs = [p for p in profs if " ".join(p) in val_keys]
    tr_sets = [toks(" ".join(p)) for p in tr_profs]
    va_sets = [toks(" ".join(p)) for p in va_profs]
    print(f"train items {len(tr_items)}, val items {len(va_items)}, "
          f"train profiles {len(tr_profs)}, val profiles {len(va_profs)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMultipleChoice.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    tr = CriticData(tr_items, tr_profs, tr_sets, tok, a.n_train, a.seed)
    va = CriticData(va_items, va_profs, va_sets, tok, a.n_val, a.seed + 1)
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=a.bs, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    for ep in range(a.epochs):
        model.train()
        for ids, mask, lab, _ in tqdm(tl, desc=f"epoch {ep}"):
            ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        hit, tot = {}, {}
        with torch.no_grad():
            for ids, mask, lab, kinds in vl:
                pred = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.argmax(-1).cpu()
                for p, l, k in zip(pred, lab, kinds):
                    tot[k] = tot.get(k, 0) + 1
                    hit[k] = hit.get(k, 0) + int(p == l)
        print("Held-out profile accuracy (chance on normal is 25%):")
        for k in tot:
            print(f"  {k}: {hit[k] / tot[k]:.3f} ({tot[k]} examples)")

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved critic to", a.out)


if __name__ == "__main__":
    main()
