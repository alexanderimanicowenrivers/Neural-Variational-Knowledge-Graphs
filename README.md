## Neural Variational Knowledge Graph Embeddings

### Introduction

Small and quick experiment on learning Gaussian embeddings for entitie and predicates in a Knowledge Graph.
The generative model can be described as follows:

```
h_s ~ N(0, 1)
h_p ~ N(0, 1)
h_o ~ N(0, 1)
y ~ Ber(y | σ(score(h_s, h_p, h_o)))
```

The rationale is that, by doing so, you can:
- Having uncertainty estimates that a triple is true,
- Associate a Gaussian distribution to each entity and predicate emebdding.

By associating a "variance" to each entity embedding, it can be possible to perform some sort of "active learning" - for instance, by sampling triples that help reduce the variance of entity embeddings (as in [1] for Gaussian Processes).

[1] Active Learning with Gaussian Processes for Object Categorization

### Model

The log-likelihood of a `<s, p, o>` triple being true is the following:

```
log p(s, p, o, y=1)
    = log ∫ p(s, p, o, h_s, h_p, h_o, y=1) d h_s, h_p, h_o
    = log ∫ p(s) p(p) p(o) p(h_s | s) p(h_p | p) p(h_o | o) p(y=1 | h_s, h_p, h_o) d h_s, h_p, h_o

```