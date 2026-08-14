# morin-filter

**Cheap corpus prior for neural language models.** Blends a model's
next-token distribution with n-gram frequencies from a local corpus.
The mixture improves both perplexity and top-1 accuracy while
concentrating probability mass on far fewer candidates.

> 🇷🇺 **[Русская версия](README.ru.md)**

Named after the *Morin surface* intuition that started the research:
most locally plausible transitions in a token graph are "phantom" —
they never actually occur in context. A corpus table knows which
transitions are real, and a probabilistic blend (never a hard mask)
uses that knowledge without breaking the rare-token tail.

## Results (honest 4:1 split, 300 queries, DeepSeek-Coder-V2-Lite 16B MoE on CPU)

| Domain | PPL model | PPL +morin | top-1 model | top-1 +morin |
|---|---|---|---|---|
| JS/TS code | 14.7 | **9.0** (1.64×) | 50.3% | **53.0%** |
| Python code | 7.9 | **4.0** (1.98×) | 55.7% | **60.7%** |

Prior size: a few thousand n-gram keys. Compare kNN-LM (ICLR 2020):
same idea via a billion-scale embedding index. Here — a plain table.

## Install

```bash
pip install numpy
git clone https://github.com/koshechkintimur-a11y/morin-filter.git
```

Requires Python 3.8+. For the llama.cpp example: `pip install llama-cpp-python`.

## Usage

```python
from morin_filter import build_ngram_prior, CorpusPrior, MixtureScorer

# 1. Build the prior from YOUR corpus (tokenized):
prior = CorpusPrior(build_ngram_prior(train_token_ids, order=2))

# 2. At each step, get normalized model log-probs (V,) and context ids:
scorer = MixtureScorer(prior, beta=0.3)
mix_logp = scorer.mixture_logp(model_logp, context_ids)

# 3. Sample / argmax / top-k under mix_logp instead of model_logp.
```

Or run the full evaluation example:

```bash
python examples/llama_cpp_example.py \
  --model path/to/model.gguf \
  --corpus path/to/your/codebase.txt \
  --beta 0.3
```

## What it is and what it is NOT

**Not a novelty claim.** The mechanism is Jelinek-Mercer smoothing
(1980) applied to a neural model, and a simplified variant of kNN-LM
(Khandelwal et al., ICLR 2020) without the embedding index. The
recency-flavoured sibling is Cache LM (Grave et al., ICLR 2017).
See `docs/LITERATURE.md` for the prior-art map.

**What it is:** a deliberately cheap engineering artifact — a
corpus prior you can attach to any llama.cpp / transformers model
in minutes, on CPU, with kilobytes-to-megabytes of memory.

## Research history

### Fracode Phase 0–1: the compression era

This project started as **Fracode**, an attempt to build a fractal
compression codec. Phase 0 produced a benchmarked container codec —
and an honest loss: `zstd` + dictionary beat it on real data. Phase 1
explored semantic codebooks, terminal-style addressing, Russian chat
corpora — a long series of negative results, each falsified against
adversarial checks before being recorded. That phase taught the
project's core habit: **when a result looks too good, it is too good —
verify until it breaks.**

### Fracode Phase 2: the geometry race

The question became: can an address-space geometry (Lévy curves,
Hilbert, tesseract, Möbius, Poincaré relay) make routing cheaper?
Experiments A–Q measured every proposed geometry on real data. All of
them failed to beat a plain Euclidean k-means router; the routing idea
converged to a known pattern (MoE). The one intuition that survived —
the *Morin surface as a model of token paths* — led to the discovery
that ~90% of local token cycles are **phantom** (plausible locally,
impossible globally), and that pruning them helps both speed and
accuracy. That became the corpus prior in this package.

### Where we ended up (honest accounting)

The winning mechanism turned out to be known: Jelinek–Mercer smoothing
(1980), Cache LM (2017), kNN-LM (2020). What survived the literature
check is an engineering angle: **the same gain as kNN-LM at a fraction
of the memory** — a plain n-gram table instead of a billion-scale
embedding index.

The full journey — 22 experiments, the falsifications we caught (the
best one: a data leak that inflated our recall estimate by 5 p.p.),
the geometries that failed, and the one effect that survived — is in
`docs/FULL_JOURNEY.md`. It is a journal of hypothesis elimination, and
we consider it the most honest part of this repository.

## License

MIT.
