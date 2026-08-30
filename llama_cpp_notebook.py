"""llama_cpp_notebook.py — NoteBook Prior on a REAL LLM (llama.cpp).

Port of the architect's notebook idea to DeepSeek-Coder:
- baseline: model alone
- morin: fixed β=0.3 (nb0)
- notebook: β from running reference book of "was the prior right here" (nb1)
- hybrid: notebook + match_len (nb2)

The notebook accumulates online during eval (recency signal, like Cache LM).
CPU only (n_gpu_layers=0).

Usage: python llama_cpp_notebook.py [--n-test 200] [--warmup 50]
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morin_filter import build_ngram_prior, CorpusPrior

CTX_TOKENS = 32
MAX_ORDER = 4
BETA = 0.3
WARMUP = 50


class Notebook:
    def __init__(self, max_entries=100_000):
        self.notes = defaultdict(lambda: [0, 0])
        self.max_entries = max_entries

    def trust(self, ctx):
        r, w = self.notes.get(ctx[-8:], (0, 0))
        if r + w == 0:
            return 0.5
        return (r + 1) / (r + w + 2)

    def update(self, ctx, prior_right):
        if len(self.notes) < self.max_entries:
            k = ctx[-8:]
            if prior_right:
                self.notes[k][0] += 1
            else:
                self.notes[k][1] += 1

    def size(self):
        return len(self.notes)


def tokenize(llm, text):
    return llm.tokenize(text.encode("utf-8"), add_bos=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"))
    ap.add_argument("--corpus", default=r"C:\Users\Geroin\chaotic-llm\phase01\corpus5m_train.txt")
    ap.add_argument("--n-lines", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    from llama_cpp import Llama
    print(f"Loading {args.model}...", flush=True)
    llm = Llama(model_path=args.model, n_ctx=1024, n_threads=args.threads,
                verbose=False, logits_all=True, n_gpu_layers=0)
    V = llm.n_vocab()
    print(f"vocab={V:,}", flush=True)

    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 3]
    lines = lines[:args.n_lines]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(lines))
    n_a = int(len(lines) * 0.6)
    n_b = int(len(lines) * 0.2)
    train_lines = [lines[i] for i in perm[:n_a]]
    val_lines = [lines[i] for i in perm[n_a:n_a + n_b]]
    test_lines = [lines[i] for i in perm[n_a + n_b:]]

    # prior from Train_A
    print("Building prior...", flush=True)
    train_tokens = [tokenize(llm, t) for t in train_lines]
    prior = CorpusPrior(build_ngram_prior(train_tokens, order=MAX_ORDER))
    print(f"  contexts={len(prior.prior):,}", flush=True)

    # warm up notebook on val (recency: it learns verdicts on val)
    nb = Notebook()
    print(f"Warming notebook on val ({len(val_lines)} docs)...", flush=True)
    for t in val_lines[:args.n_train]:
        ids = tokenize(llm, t)
        for i in range(CTX_TOKENS, len(ids) - 1):
            ctx = tuple(ids[i - CTX_TOKENS:i])
            target = ids[i]
            tab, tot = None, 0
            for L in range(min(MAX_ORDER, len(ctx)), 0, -1):
                tbl = prior.prior.get(ctx[-L:])
                if tbl:
                    tab, tot = tbl, sum(tbl.values())
                    break
            prior_right = bool(tab and tab.get(target, 0) > 0)
            nb.update(ctx, prior_right)

    print(f"Evaluating on test ({len(test_lines)} docs)...", flush=True)
    nll = {"nb0": [], "nb1": [], "nb2": []}
    acc = {"nb0": 0, "nb1": 0, "nb2": 0}
    cnt = 0
    t0 = time.time()
    for t in test_lines:
        ids = tokenize(llm, t)
        for i in range(CTX_TOKENS, len(ids) - 1):
            ctx = tuple(ids[i - CTX_TOKENS:i])
            target = ids[i]
            llm.reset()
            llm.eval(ctx)
            logits = np.array(llm.scores[len(ctx) - 1], dtype=np.float32, copy=True)
            logp = logits.astype(np.float64)
            logp -= np.logaddexp.reduce(logp)

            tab, tot = None, 0
            for L in range(min(MAX_ORDER, len(ctx)), 0, -1):
                tbl = prior.prior.get(ctx[-L:])
                if tbl:
                    tab, tot = tbl, sum(tbl.values())
                    break
            prior_right = bool(tab and tab.get(target, 0) > 0)

            b0 = BETA
            b1 = min(0.95, max(0.05, nb.trust(ctx)))
            # b2 hybrid: notebook + match_len
            if tab:
                mLen = 0
                for L in range(min(MAX_ORDER, len(ctx)), 0, -1):
                    if prior.prior.get(ctx[-L:]):
                        mLen = L
                        break
                b2 = min(0.95, max(0.05, 0.5 * nb.trust(ctx) + 0.5 * (mLen / MAX_ORDER)))
            else:
                b2 = b0

            for key, b in (("nb0", b0), ("nb1", b1), ("nb2", b2)):
                if tab and tot > 0:
                    prior_logp = np.full(V, -np.inf, dtype=np.float64)
                    for tok, c in tab.items():
                        prior_logp[tok] = np.log(c / tot)
                    with np.errstate(divide="ignore"):
                        mix = np.logaddexp(np.log1p(-b) + logp, np.log(b) + prior_logp)
                else:
                    mix = logp
                nll[key].append(-mix[target])
                if int(np.argmax(mix)) == target:
                    acc[key] += 1

            nb.update(ctx, prior_right)   # keep learning online
            cnt += 1
            if cnt >= args.n_test:
                break
        if cnt >= args.n_test:
            break

    n = len(nll["nb0"])
    print(f"\n=== NoteBook Prior on LLM (n={n}, {time.time()-t0:.0f}s) ===")
    for key in ("nb0", "nb1", "nb2"):
        ppl = float(np.exp(np.mean(nll[key])))
        print(f"  {key}: PPL={ppl:.3f}  acc={acc[key]/n:.3f}")
    print(f"  notebook entries={nb.size():,}")

    res = {"model": os.path.basename(args.model), "n": n, "notebook_entries": nb.size(),
           "ppl": {k: float(np.exp(np.mean(v))) for k, v in nll.items()},
           "acc": {k: acc[k] / n for k in acc}}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "llama_notebook_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
