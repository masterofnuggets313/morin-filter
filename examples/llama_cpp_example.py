"""Example: use morin-filter with any llama.cpp model.

Run: python examples/llama_cpp_example.py --model path/to/model.gguf \
       --corpus path/to/corpus.txt [--beta 0.3] [--n-test 300]

Builds the n-gram prior from the corpus (first 80% of texts),
evaluates next-token prediction on the rest, and prints
PPL / top-1 accuracy of the model vs the mixture.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from morin_filter import build_ngram_prior, CorpusPrior, MixtureScorer  # noqa: E402

CTX_TOKENS = 8
ORDER = 2


def tokenize(llm, text: str) -> list[int]:
    return llm.tokenize(text.encode("utf-8"), add_bos=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .gguf")
    ap.add_argument("--corpus", required=True, help="plain-text corpus file")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=4000)
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=64, n_threads=8,
                verbose=False, logits_all=True)
    print(f"model loaded, vocab {llm.n_vocab():,}")

    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        texts = [l.strip() for l in f if len(l.strip()) > 3]
    texts_train = texts[4::5][:args.n_train]
    texts_test = [t for i, t in enumerate(texts) if i % 5 != 4]

    train_tokens = [tokenize(llm, t) for t in texts_train]
    prior = CorpusPrior(build_ngram_prior(train_tokens, ORDER))
    scorer = MixtureScorer(prior, beta=args.beta)
    print(f"prior built: {len(prior.prior):,} contexts")

    te_samps = []
    for t in texts_test:
        ids = tokenize(llm, t)
        for i in range(len(ids) - CTX_TOKENS):
            te_samps.append((ids[i:i + CTX_TOKENS], ids[i + CTX_TOKENS]))
        if len(te_samps) >= args.n_test:
            break

    nll_base = []
    nll_mix = []
    acc_base = acc_mix = 0
    t0 = time.time()
    for ctx, target in te_samps:
        llm.reset()
        llm.eval(ctx)
        # llama.cpp keeps logits in a (n_ctx, V) buffer; only the first
        # len(ctx) rows are filled — read row len(ctx)-1, and copy!
        logits = np.array(llm.scores[len(ctx) - 1], dtype=np.float32,
                          copy=True)
        logp = logits.astype(np.float64)
        logp -= np.logaddexp.reduce(logp)
        nll_base.append(-logp[target])
        if int(logp.argmax()) == target:
            acc_base += 1
        mix = scorer.mixture_logp(logp, ctx)
        nll_mix.append(-mix[target])
        if int(mix.argmax()) == target:
            acc_mix += 1

    n = len(te_samps)
    ppl_b = float(np.exp(np.mean(nll_base)))
    ppl_m = float(np.exp(np.mean(nll_mix)))
    print(f"\n  n={n}  beta={args.beta}  ({time.time()-t0:.0f}s)")
    print(f"  model:      PPL {ppl_b:8.2f}   top-1 {acc_base/n:6.1%}")
    print(f"  +morin:     PPL {ppl_m:8.2f}   top-1 {acc_mix/n:6.1%}  "
          f"({ppl_b/ppl_m:.2f}x PPL)")


if __name__ == "__main__":
    main()
