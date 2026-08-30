"""validate_trust.py — verify TrustScorer reproduces the −33% PPL boost.

Loads corpus5m from chaotic-llm, builds the prior, fits TrustScorer on a val
split with a smoothed-trigram placeholder model, evaluates on test split.
Compares: MixtureScorer (fixed β=0.3) vs TrustScorer (learned β).
"""
import os
import sys
import math
import json
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morin_filter import build_ngram_prior, CorpusPrior, MixtureScorer, TrustScorer

CHAOS = r"C:\Users\Geroin\chaotic-llm\phase01"
MAX_ORDER = 8


def tokenize_corpus():
    sys.path.insert(0, CHAOS)
    from exp_memory_selector.experiment import build_tokenizer
    tok = build_tokenizer()
    with open(os.path.join(CHAOS, "corpus5m_train.txt"), encoding="utf-8", errors="ignore") as f:
        tr = f.read()
    with open(os.path.join(CHAOS, "corpus5m_test.txt"), encoding="utf-8", errors="ignore") as f:
        te = f.read()
    return tok, tok.encode(tr).ids, tok.encode(te).ids


def build_smoothed(train_ids, V, order=3):
    tables = {L: defaultdict(Counter) for L in range(1, order + 1)}
    for i in range(1, len(train_ids)):
        tok = train_ids[i]
        for L in range(1, order + 1):
            if i - L < 0:
                break
            tables[L][tuple(train_ids[i - L:i])][tok] += 1
    return tables


def smoothed_logp(tables, ctx, target, V, lam):
    p = 0.0
    for L in range(1, len(tables) + 1):
        if len(ctx) < L:
            continue
        tab = tables[L].get(tuple(ctx[-L:]))
        if tab:
            st = sum(tab.values())
            p += lam[L - 1] * (tab.get(target, 0) + 1) / (st + V)
        else:
            p += lam[L - 1] * (1 / V)
    return np.log(p) if p > 0 else -1e300


def main():
    print("Loading corpus...", flush=True)
    tok, train_ids, test_ids = tokenize_corpus()
    V = tok.get_vocab_size()
    # NOTE: tokenizer trained on small corpus5m (vocab 512); reuse prior build on all train
    print(f"V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    print("Building prior (order %d)..." % MAX_ORDER, flush=True)
    prior_dict = build_ngram_prior([train_ids], order=MAX_ORDER)
    prior = CorpusPrior(prior_dict)
    print(f"  contexts={len(prior_dict):,}", flush=True)

    sm = build_smoothed(train_ids, V, order=3)
    lam = (0.2, 0.3, 0.5)

    rng = np.random.default_rng(42)
    n = len(test_ids) - MAX_ORDER - 1
    all_idx = rng.choice(n, size=min(20000, n), replace=False)
    half = len(all_idx) // 2
    val_idx, te_idx = all_idx[:half], all_idx[half:]

    def make_logp(ctx, V, sm, lam):
        lp = np.full(V, np.log(1 / V), dtype=np.float64)
        for L in range(1, len(sm) + 1):
            if len(ctx) < L:
                continue
            tab = sm[L].get(tuple(ctx[-L:]))
            if tab:
                st = sum(tab.values())
                for t, c in tab.items():
                    lp[t] = np.logaddexp(lp[t], np.log(lam[L - 1] * (c + 1) / (st + V)))
        # normalize
        lp -= np.logaddexp.reduce(lp)
        return lp

    def prep(idx, max_ctx):
        ctxs, tgts, lps = [], [], []
        for i in idx:
            ctx = tuple(test_ids[max(0, i - max_ctx):i])
            ctxs.append(ctx)
            tgts.append(test_ids[i])
            lps.append(make_logp(ctx, V, sm, lam))
        return ctxs, tgts, lps

    max_ctx = MAX_ORDER
    v_ctx, v_tgt, v_lp = prep(val_idx, max_ctx)
    t_ctx, t_tgt, t_lp = prep(te_idx, max_ctx)

    # --- TrustScorer fit on val ---
    print("Fitting TrustScorer on val...", flush=True)
    ts = TrustScorer(prior, beta_fallback=0.3)
    nll = ts.fit_batch(v_ctx, v_tgt, v_lp, lr=0.03, steps=3000)
    print(f"  val mixture NLL={nll:.4f}  w={np.round(ts.w, 3)}", flush=True)

    # --- evaluate on test ---
    def eval_scorer(scorer, name):
        nll = 0.0
        acc = 0
        for ctx, tgt, lp in zip(t_ctx, t_tgt, t_lp):
            mix = scorer.mixture_logp(lp, ctx)
            nll += -mix[tgt]
            if int(np.argmax(mix)) == tgt:
                acc += 1
        ppl = float(np.exp(nll / len(t_ctx)))
        print(f"  {name}: PPL={ppl:.4f}  acc={acc/len(t_ctx):.4f}")
        return {"name": name, "ppl": ppl, "acc": acc / len(t_ctx)}

    results = []
    # model alone
    nll_m = sum(-lp[t] for lp, t in zip(t_lp, t_tgt))
    results.append({"name": "model_alone", "ppl": float(np.exp(nll_m / len(t_lp)))})
    # T0 = same 8-back prior, fixed beta (untrained TrustScorer)
    t0 = TrustScorer(prior, beta_fallback=0.3)
    results.append(eval_scorer(t0, "T0_fixed03_8back"))
    results.append(eval_scorer(ts, "T3c_TrustScorer"))

    print(json.dumps(results, indent=2))
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "validate_trust_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
