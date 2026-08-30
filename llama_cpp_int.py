"""llama_cpp_int.py — INTEGRATION (morin + static NoteBook + TrustScorer)
on a REAL LLM (DeepSeek-Coder via llama.cpp). CPU only.

Closes the reviewer's decisive test: does INT hold on a real model?
Variants on test:
  nb0  = morin fixed β=0.3 (baseline)
  nb1  = NoteBook static trust only
  T3c  = head only (notebook feature neutralized)
  INT  = head + notebook feature (the integration)

Usage: python llama_cpp_int.py [--n-test 200] [--warmup 50]
"""
import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morin_filter import build_ngram_prior, CorpusPrior

CTX_TOKENS = 32
MAX_ORDER = 4
BETA = 0.3


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


class IntegratedHead:
    N_FEAT = 7

    def __init__(self):
        self.w = np.zeros(self.N_FEAT + 1)
        self._mean = None
        self._std = None

    def fit(self, X, pm, pp, lr=0.03, steps=3000):
        X = np.array(X, dtype=np.float64)
        pm = np.array(pm)
        pp = np.array(pp)
        self._mean = X.mean(0)
        self._std = X.std(0)
        self._std = np.where(self._std < 1e-6, 1.0, self._std)
        Xs = (X - self._mean) / self._std
        w = np.zeros(Xs.shape[1] + 1)
        best = None
        for _ in range(steps):
            z = Xs @ w[:-1] + w[-1]
            b = 1.0 / (1.0 + np.exp(-z))
            b = np.clip(b, 0.05, 0.95)
            mix = (1 - b) * pm + b * pp
            d = -(pp - pm) / np.clip(mix, 1e-300, None)
            g_beta = d * b * (1 - b)
            grad = np.concatenate([[g_beta.sum()], Xs.T @ g_beta])
            w -= lr * grad / len(pm)
            nll = np.mean(-np.log(np.clip(mix, 1e-300, 1)))
            if best is None or nll < best[0]:
                best = (nll, w.copy())
        self.w = best[1]
        return best[0]

    def beta(self, feats, neutralize_nb=False):
        f = np.array(feats, dtype=np.float64)
        if neutralize_nb:
            f[-1] = 0.5
        x = (f - self._mean) / self._std
        z = x @ self.w[:-1] + self.w[-1]
        b = 1.0 / (1.0 + math.exp(-z))
        return max(0.05, min(b, 0.95))


def tokenize(llm, text):
    return llm.tokenize(text.encode("utf-8"), add_bos=False)


def backoff(prior, ctx):
    for L in range(min(MAX_ORDER, len(ctx)), 0, -1):
        tbl = prior.prior.get(ctx[-L:])
        if tbl:
            return tbl, sum(tbl.values())
    return None, 0


def feats_of(prior, nb, ctx):
    tab, tot = backoff(prior, ctx)
    if tab is None:
        return None, None, 0
    mLen = 0
    for L in range(min(MAX_ORDER, len(ctx)), 0, -1):
        if prior.prior.get(ctx[-L:]):
            mLen = L
            break
    p_top = max(tab.values()) / tot
    n_cand = len(tab)
    ps = np.array(list(tab.values()), dtype=np.float64) / tot
    entropy = float(-(ps * np.log(ps)).sum())
    nb_t = nb.trust(ctx)
    return np.array([mLen, math.log1p(tot), p_top, n_cand, len(ctx),
                     entropy, nb_t], dtype=np.float64), tab, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"))
    ap.add_argument("--corpus", default=r"C:\Users\Geroin\chaotic-llm\phase01\corpus5m_train.txt")
    ap.add_argument("--n-lines", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    from llama_cpp import Llama
    print(f"Loading {args.model}...", flush=True)
    llm = Llama(model_path=args.model, n_ctx=1024, n_threads=args.threads,
                verbose=False, logits_all=True, n_gpu_layers=0)
    V = llm.n_vocab()

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

    print("Building prior from Train_A...", flush=True)
    train_tokens = [tokenize(llm, t) for t in train_lines]
    prior = CorpusPrior(build_ngram_prior(train_tokens, order=MAX_ORDER))
    print(f"  contexts={len(prior.prior):,}", flush=True)

    # static notebook from Train_A
    print("Building static notebook from Train_A...", flush=True)
    nb = Notebook()
    for t in train_tokens:
        for i in range(CTX_TOKENS, len(t) - 1):
            ctx = tuple(t[i - CTX_TOKENS:i])
            target = t[i]
            tab, tot = backoff(prior, ctx)
            nb.update(ctx, bool(tab and tab.get(target, 0) > 0))
    print(f"  entries={nb.size():,}", flush=True)

    # collect val samples + fit head
    print("Collecting val samples...", flush=True)
    X, pm, pp = [], [], []
    for t in val_lines:
        ids = tokenize(llm, t)
        for i in range(CTX_TOKENS, len(ids) - 1):
            ctx = tuple(ids[i - CTX_TOKENS:i])
            target = ids[i]
            feats, tab, tot = feats_of(prior, nb, ctx)
            if feats is None:
                continue
            llm.reset()
            llm.eval(ctx)
            logits = np.array(llm.scores[len(ctx) - 1], dtype=np.float32, copy=True)
            logp = logits.astype(np.float64)
            logp -= np.logaddexp.reduce(logp)
            X.append(feats)
            pm.append(float(np.exp(float(logp[target]))))
            pp.append(tab.get(target, 0) / tot)
            if len(X) >= args.n_train:
                break
        if len(X) >= args.n_train:
            break
    head = IntegratedHead()
    nll = head.fit(X, pm, pp, lr=0.03, steps=3000)
    print(f"  head fit NLL={nll:.4f}  w={np.round(head.w, 3)}", flush=True)

    # evaluate on test
    print(f"Evaluating on test...", flush=True)
    nlls = {"nb0": [], "nb1": [], "T3c": [], "INT": []}
    acc = {"nb0": 0, "nb1": 0, "T3c": 0, "INT": 0}
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
            feats, tab, tot = feats_of(prior, nb, ctx)
            b0 = BETA
            b1 = min(0.95, max(0.05, nb.trust(ctx)))
            if feats is not None:
                b3 = head.beta(feats, neutralize_nb=True)
                b4 = head.beta(feats)
            else:
                b3 = b4 = b0
            for key, b in (("nb0", b0), ("nb1", b1), ("T3c", b3), ("INT", b4)):
                if tab and tot > 0:
                    prior_logp = np.full(V, -np.inf, dtype=np.float64)
                    for tok, c in tab.items():
                        prior_logp[tok] = np.log(c / tot)
                    with np.errstate(divide="ignore"):
                        mix = np.logaddexp(np.log1p(-b) + logp, np.log(b) + prior_logp)
                else:
                    mix = logp
                nlls[key].append(-mix[target])
                if int(np.argmax(mix)) == target:
                    acc[key] += 1
            cnt += 1
            if cnt >= args.n_test:
                break
        if cnt >= args.n_test:
            break

    n = len(nlls["nb0"])
    print(f"\n=== INTEGRATION on LLM (n={n}, {time.time()-t0:.0f}s) ===")
    for key in ("nb0", "nb1", "T3c", "INT"):
        ppl = float(np.exp(np.mean(nlls[key])))
        print(f"  {key}: PPL={ppl:.3f}  acc={acc[key]/n:.3f}")

    res = {"model": os.path.basename(args.model), "n": n,
           "ppl": {k: float(np.exp(np.mean(v))) for k, v in nlls.items()},
           "acc": {k: acc[k] / n for k in acc},
           "head_w": head.w.tolist(), "notebook_entries": nb.size()}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "llama_int_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
