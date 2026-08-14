"""morin-filter: cheap corpus prior for neural language models.

A corpus-aware prior that improves a neural LM's next-token predictions
by blending its output with n-gram frequencies from a local corpus.

Why it works: many locally plausible continuations never actually occur
in a given context ("phantom" transitions). A model wastes probability
mass on them. Restricting candidates to observed continuations — or,
better, blending in corpus frequencies as a prior — improves both
quality (accuracy, perplexity) and effective compute (mass concentrates
on fewer candidates).

Measured results (DeepSeek-Coder-V2-Lite 16B MoE, 102,400-token BPE,
CPU inference via llama.cpp, honest 4:1 text split, 300 queries):

    domain          PPL base -> mix   top-1 acc        prior size
    JS/TS code      14.7   ->  9.0    50.3% -> 53.0%   ~9K keys
    Python code      7.9   ->  4.0    55.7% -> 60.7%   ~10K keys

Compare: kNN-LM (Khandelwal et al., ICLR 2020) needs a billion-scale
embedding index; this prior is a plain n-gram table. Same idea, orders
of magnitude cheaper.

NOTE: the mechanism is not new — it is Jelinek-Mercer smoothing (1980)
extended to a neural model, and a simplified variant of kNN-LM. This
library is an engineering artifact, not a novelty claim. See
docs/LITERATURE.md for the full map of prior art.

License: MIT.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["build_ngram_prior", "MixtureScorer", "CorpusPrior"]


def build_ngram_prior(
    token_streams: Iterable[Sequence[int]],
    order: int = 2,
) -> dict:
    """Build an n-gram frequency prior from tokenized texts.

    Parameters
    ----------
    token_streams : iterable of token-id sequences (the corpus).
    order : max context length for the n-gram table.

    Returns
    -------
    prior : dict[(context_tuple), dict[token_id, count]]
        Frequencies of every observed continuation for every observed
        context of length 1..order. Unobserved pairs get zero mass —
        the neural model carries the tail (that is what makes the
        mixture safe; hard masking of unobserved tokens hurts PPL).
    """
    prior: dict = defaultdict(lambda: defaultdict(int))
    for ids in token_streams:
        for i in range(len(ids)):
            for back in range(1, min(order, i) + 1):
                prior[tuple(ids[i - back:i])][ids[i]] += 1
    return {k: dict(v) for k, v in prior.items()}


class CorpusPrior:
    """Lookup of corpus log-probabilities for a context.

    Only observed tokens get mass. For unobserved contexts returns
    None so callers can fall back to the plain model.
    """

    def __init__(self, prior: dict):
        self.prior = prior
        self._cache: dict = {}

    def logp(self, context: Sequence[int], vocab_size: int) -> Optional[np.ndarray]:
        """Return log P_corpus over the full vocab for `context`.

        Unobserved tokens get -inf (zero mass). Returns None when the
        context has no observations.
        """
        key = tuple(context)
        if key in self._cache:
            return self._cache[key]
        counts = np.zeros(vocab_size, dtype=np.float64)
        total = 0
        for back in range(min(3, len(key)), 0, -1):
            ctx = key[-back:]
            table = self.prior.get(ctx)
            if table:
                for tok, c in table.items():
                    counts[tok] += c
                    total += c
        if total == 0:
            self._cache[key] = None
            return None
        with np.errstate(divide="ignore"):
            out = np.log(counts / total)
        self._cache[key] = out
        return out


class MixtureScorer:
    """Mixture of model and corpus distributions.

    log P_mix = logadd[log(1-beta) + log P_model, log beta + log P_corpus]

    The mixture never zeroes out tokens, so recall is 100% by
    construction and rare-token PPL is preserved — unlike hard masking.
    """

    def __init__(self, prior: CorpusPrior, beta: float = 0.3):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.prior = prior
        self.beta = beta

    def mixture_logp(
        self,
        model_logp: np.ndarray,  # (V,) normalized log-probs of the model
        context: Sequence[int],
    ) -> np.ndarray:
        """Return log P_mix over the vocab for one position."""
        c_logp = self.prior.logp(context, len(model_logp))
        if c_logp is None:
            return model_logp
        log_beta = np.log(self.beta)
        log_alpha = np.log1p(-self.beta)
        with np.errstate(divide="ignore"):
            return np.logaddexp(log_alpha + model_logp, log_beta + c_logp)

    def topk_candidates(
        self,
        model_logp: np.ndarray,
        context: Sequence[int],
        k: int = 16,
    ) -> list[int]:
        """Top-k token ids under the mixture (for constrained decoding
        or cheap candidate pruning)."""
        mix = self.mixture_logp(model_logp, context)
        return list(np.argsort(-mix)[:k])

    def predict(
        self,
        model_logp: np.ndarray,
        context: Sequence[int],
    ) -> int:
        """Argmax under the mixture."""
        mix = self.mixture_logp(model_logp, context)
        return int(np.argmax(mix))
