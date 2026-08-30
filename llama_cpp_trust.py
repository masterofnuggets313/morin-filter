"""llama_cpp_trust.py — port TrustScorer (Neural Trust) onto a real LLM via llama.cpp.

Follows the reviewer's 4-step protocol on the REAL model:
  Step 1: llama.cpp baseline PPL on code corpus
  Step 2: + morin (fixed β=0.3) PPL
  Step 3: + Neural Trust (TrustScorer) PPL
  Step 4: comparison

Loads DeepSeek-Coder-V2-Lite-Instruct GGUF, builds n-gram prior on a train split
of the corpus (tokenized with the LLM's own tokenizer), fits TrustScorer on a val
split, evaluates on a test split. All on CPU (llama.cpp).

Usage: python llama_cpp_trust.py [--model path] [--corpus path] [--n-test 300]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morin_filter import build_ngram_prior, CorpusPrior, TrustScorer

CTX_TOKENS = 32
MAX_ORDER = 4          # look BACK up to 4 (LLM already knows ~3; prior adds 4)
BETA_FALLBACK = 0.3


def tokenize(llm, text: str) -> list[int]:
    return llm.tokenize(text.encode("utf-8"), add_bos=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"))
    ap.add_argument("--corpus", default=r"C:\Users\Geroin\chaotic-llm\phase01\corpus5m_train.txt")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-lines", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading model {args.model}...", flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=1024, n_threads=args.threads,
                verbose=False, logits_all=True, n_gpu_layers=0)   # CPU only
    V = llm.n_vocab()
    print(f"model loaded, vocab={V:,}", flush=True)

    # ---- corpus split into documents ----
    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 3]
    lines = lines[:args.n_lines]
    n_docs = len(lines)
    # 3-way split by DOCUMENT (not position) — strongest against leakage
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_docs)
    n_a = int(n_docs * 0.6)
    n_b = int(n_docs * 0.2)
    train_lines = [lines[i] for i in perm[:n_a]]       # Train_A: prior
    val_lines = [lines[i] for i in perm[n_a:n_a + n_b]]  # Train_B: trust head
    test_lines = [lines[i] for i in perm[n_a + n_b:]]    # Test: eval
    print(f"docs: train_A={len(train_lines)} val_B={len(val_lines)} test={len(test_lines)}", flush=True)

    # ---- build prior from Train_A (LLM tokenizer) ----
    print("Building prior from Train_A...", flush=True)
    train_tokens = [tokenize(llm, t) for t in train_lines]
    prior_dict = build_ngram_prior(train_tokens, order=MAX_ORDER)
    prior = CorpusPrior(prior_dict)
    print(f"  prior contexts={len(prior_dict):,}", flush=True)

    # ---- collect eval samples (val + test) ----
    def collect_samples(docs, max_samps):
        ctxs, tgts, logits = [], [], []
        for t in docs:
            ids = tokenize(llm, t)
            for i in range(CTX_TOKENS, len(ids) - 1):
                ctx = ids[i - CTX_TOKENS:i]
                ctxs.append(ctx)
                tgts.append(ids[i])
                if len(ctxs) >= max_samps:
                    break
            if len(ctxs) >= max_samps:
                break
        return ctxs, tgts

    print("Collecting val samples (for trust head)...", flush=True)
    v_ctx, v_tgt = collect_samples(val_lines, args.n_train)
    print(f"  val samples={len(v_ctx)}", flush=True)

    # run model once on all contexts, cache logits
    print("Running model on val (1 forward pass per sample)...", flush=True)
    v_logp = []
    for ctx in v_ctx:
        llm.reset()
        llm.eval(ctx)
        logits = np.array(llm.scores[len(ctx) - 1], dtype=np.float32, copy=True)
        logp = logits.astype(np.float64)
        logp -= np.logaddexp.reduce(logp)
        v_logp.append(logp)

    # ---- fit TrustScorer on val ----
    print("Fitting TrustScorer on val...", flush=True)
    ts = TrustScorer(prior, beta_fallback=BETA_FALLBACK, max_back=MAX_ORDER)
    nll = ts.fit_batch(v_ctx, v_tgt, v_logp, lr=0.03, steps=2000)
    print(f"  val NLL={nll:.4f}  w={np.round(ts.w,3)}", flush=True)

    # fixed-beta scorer with the SAME 8-back prior (separate, untrained)
    morin_scorer = TrustScorer(prior, beta_fallback=BETA_FALLBACK, max_back=MAX_ORDER)

    # ---- evaluate on test ----
    print("Evaluating on test...", flush=True)
    t_ctx, t_tgt = collect_samples(test_lines, args.n_test)
    print(f"  test samples={len(t_ctx)}", flush=True)

    nll_base = []
    nll_morin = []
    nll_trust = []
    acc_base = acc_morin = acc_trust = 0
    t0 = time.time()
    for idx, ctx in enumerate(t_ctx):
        target = t_tgt[idx]
        llm.reset()
        llm.eval(ctx)
        logits = np.array(llm.scores[len(ctx) - 1], dtype=np.float32, copy=True)
        logp = logits.astype(np.float64)
        logp -= np.logaddexp.reduce(logp)
        # baseline
        nll_base.append(-logp[target])
        if int(logp.argmax()) == target:
            acc_base += 1
        # fixed beta (8-back prior, untrained scorer = pure β=0.3)
        mix_m = morin_scorer.mixture_logp(logp, ctx)
        nll_morin.append(-mix_m[target])
        if int(mix_m.argmax()) == target:
            acc_morin += 1
        # trained trust
        mix_t = ts.mixture_logp(logp, ctx)
        nll_trust.append(-mix_t[target])
        if int(mix_t.argmax()) == target:
            acc_trust += 1

    n = len(t_ctx)
    ppl_b = float(np.exp(np.mean(nll_base)))
    ppl_m = float(np.exp(np.mean(nll_morin)))
    ppl_t = float(np.exp(np.mean(nll_trust)))
    print(f"\n=== LLM ({os.path.basename(args.model)}) ===")
    print(f"  n={n}  ({time.time()-t0:.0f}s)")
    print(f"  baseline:       PPL {ppl_b:8.2f}   top-1 {acc_base/n:6.1%}")
    print(f"  +morin β=0.3:   PPL {ppl_m:8.2f}   top-1 {acc_morin/n:6.1%}  ({ppl_b/ppl_m:.2f}x)")
    print(f"  +Neural Trust:  PPL {ppl_t:8.2f}   top-1 {acc_trust/n:6.1%}  ({ppl_b/ppl_t:.2f}x)")

    res = {"model": os.path.basename(args.model), "n": n,
           "baseline_ppl": ppl_b, "morin_ppl": ppl_m, "trust_ppl": ppl_t,
           "baseline_acc": acc_base / n, "morin_acc": acc_morin / n,
           "trust_acc": acc_trust / n}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "llama_trust_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
