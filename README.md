## Neural Variational Knowledge Graph Embeddings

### Introduction

Small and quick experiment inspired by [1] on learning Gaussian embeddings for entities and predicates in a Knowledge Graph.
The generative model can be described as follows:

```
h_s ~ N(μ_s, Σ_s)
h_p ~ N(μ_p, Σ_p)
h_o ~ N(μ_o, Σ_o)
y ~ Ber(y | σ(score(h_s, h_p, h_o)))
```

The rationale is that, by doing so, you can:
- Having uncertainty estimates that a triple is true,
- Associate a Gaussian distribution to each entity and predicate emebddings.

By associating a "variance" to each entity embedding, it can be possible to perform some sort of "active learning" - for instance, by sampling triples that help reduce the variance of entity embeddings (similarly to [2]).

[1] [Neural Variational Inference for Text Processing](https://arxiv.org/abs/1511.06038) - ICML 2016

[2] [Active Learning with Gaussian Processes for Object Categorization](http://people.csail.mit.edu/rurtasun/publications/iccv_kapoor_et_al.pdf) - ICCV 2007

### Model

The log-likelihood of a `<s, p, o>` triple being true is the following:

```
log p(s, p, o, y=1)
    = log ∫ p(s, p, o, h_s, h_p, h_o, y=1) d h_s, h_p, h_o
    = log ∫ p(s) p(p) p(o) p(h_s | s) p(h_p | p) p(h_o | o) p(y=1 | h_s, h_p, h_o) d h_s, h_p, h_o
```

However, such an integral is messy to solve.

We can define a variational lower bound to `log p(s, p, o, y=1)`:

```
log p(s, p, o, y=1)
    = log ∫ p(s, p, o, h_s, h_p, h_o, y=1) d h_s, h_p, h_o
    = log ∫ p(s, p, o, h_s, h_p, h_o, y=1) (q(h_s, h_p, h_o | s, p, o) / q(h_s, h_p, h_o | s, p, o)) d h_s, h_p, h_o
    >= E_q(h_s, h_p, h_o | s, p, o) [ log p(s, p, o, h_s, h_p, h_o, y=1) - log q(h_s, h_p, h_o | s, p, o) ]
    = E_q(h_s, h_p, h_o | s, p, o) [ p(y=1 | h_s, h_p, h_o) ] - KL(q(h_s | s) || p(h_s | s)p(s)) - KL(q(h_p | p) || p(h_p | p)p(p)) - KL(q(h_o | o) || p(h_o | o)p(o))
```