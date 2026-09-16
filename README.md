# Bayesian Mann-Whitney (Rank Sum) Test

A JAX implementation of the Bayesian rank sum test of van Doorn et al. (2020), computing a Bayes factor (BF10) for whether two independent samples differ, without assuming normality. The core Gibbs sampler runs under `jax.lax.scan`, and the whole test can be vectorized across many independent sample-group comparisons at once via `jax.vmap` + `jax.jit`.

## Reference

van Doorn, J., Ly, A., Marsman, M., & Wagenmakers, E.-J. (2020). Bayesian rank-based hypothesis testing for the rank sum test, the signed rank test, and Spearman's ρ. *Journal of Applied Statistics*, 47(16), 2984–3006. https://doi.org/10.1080/02664763.2019.1709053
(PMID: 35707708, PMCID: PMC9041780)

## How it works

Bayesian inference for rank-based tests is complicated by the fact that ranks have no explicit likelihood function. The paper resolves this with a data-augmentation approach: the observed ranks are treated as an impoverished view of latent, continuous scores `Z^x ~ N(-δ/2, 1)` and `Z^y ~ N(+δ/2, 1)`, constrained to respect the observed rank order. The effect size `δ` gets a `Cauchy(0, γ)` prior under H1 (H0 fixes `δ = 0`), and posterior samples of `δ` are drawn via Gibbs sampling. The Bayes factor `BF10` is then obtained from the Savage–Dickey density ratio: the prior density of `δ` at 0, divided by the posterior density of `δ` at 0 (estimated via KDE on the post-burn-in draws).

This implementation follows the paper's Gibbs sampler directly, with two additions on top of the base algorithm:

- **A decorrelation step** (Liu & Sabatti, 2000; Morey, Rouder & Speckman, 2008): after each per-element update of the latent scores `Z`, a single shared shift — drawn from its exact conditional distribution — is added to all of `Z` at once. This can never change the relative order of the `Z`'s (so it's always a valid move), and it substantially reduces the autocorrelation that plagues the element-wise data-augmentation updates on their own.
- **Numerically stable initialization** of the truncation boundaries for `Z`: the first draw of `Z` is bounded using *normal scores* (rank → standard-normal quantile) rather than the raw data values. Raw data can sit on an arbitrary or heavy-tailed scale (e.g. outliers), which can push the initial truncated-normal sampling into a region where the standard normal CDF saturates to exactly 0 or 1 in floating point, producing `NaN`s. Normal scores are rank-equivalent (so the ordinal constraint is unaffected) but always lie on a well-behaved unit-normal scale.

## Usage

```python
BF10 = Bayes_rank_sum_test(X, Y)
```

For running many comparisons in parallel (e.g. simulation studies), use the batched, JIT-compiled version:

```python
BF10s = Bayes_rank_sum_test_Batch(X, Y)  # X, Y: shape (n_batch, n_X), (n_batch, n_Y)
```

## Validation: Type I error rate vs. sample size

As a sanity check, we compared the Bayesian rank sum test's false-positive rate against the classical Mann-Whitney U test's, under a **practically null** scenario: two normal distributions with a genuine mean difference of only 0.1 against a shared standard deviation of 3 (standardized effect size ≈ 0.033 — negligible in practice). Rejection thresholds were `p < 0.05` for the frequentist test and `BF10 > 3` (conventional "moderate evidence" cutoff) for the Bayesian test. For each sample size, 50 independent replicate datasets were drawn and the false-positive rate was computed as the fraction of replicates crossing the rejection threshold.

```python
sample_sizes = np.concatenate([[10], [50], [100], [500], np.arange(1000, 8500, 500)])
mu_diff = 0.1
sigma = 3.
FB10_error_I_prob_s, p_val_error_I_prob_s = [], []
for s in sample_sizes:
  X, Y = np.random.normal(0., sigma, [50, s]), np.random.normal(mu_diff,sigma, [50, s])
  p_vals = scipy.stats.mannwhitneyu(X, Y,axis=-1).pvalue
  BF10s = Bayes_rank_sum_test_Batch(X,Y).ravel()
  FB10_error_I_prob = jnp.nonzero(BF10s>3.)[0].shape[0]/BF10s.shape[0]
  p_val_error_I_prob = jnp.nonzero(p_vals<0.05)[0].shape[0]/p_vals.shape[0]
  FB10_error_I_prob_s.append(FB10_error_I_prob)
  p_val_error_I_prob_s.append(p_val_error_I_prob)
```

Plotting code:

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- scientific style settings ---
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.linewidth": 1.1,
    "axes.edgecolor": "0.15",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "figure.dpi": 130,
})

fig, ax = plt.subplots(figsize=(6.5, 4.8))

ax.plot(
    sample_sizes, p_val_error_I_prob_s,
    marker="o", markersize=6, linewidth=1.8,
    color="#D55E00", label="Mann-Whitney U ($p<0.05$)",
)
ax.plot(
    sample_sizes, FB10_error_I_prob_s,
    marker="s", markersize=6, linewidth=1.8,
    color="#0072B2", label="Bayesian rank sum ($BF_{10}>3$)",
)

ax.set_xscale("log")
ax.set_xlabel("Sample size per group ($n$)")
ax.set_ylabel("False positive rate - Type I error")
ax.set_title(
    "False positive rate vs. sample size\n"
    r"($\mu_1=0,\ \mu_2=0.1,\ \sigma=3$; true effect size $\approx 0.033$)",
    fontsize=12,
)

ax.set_ylim(-0.02, max(max(p_val_error_I_prob_s), max(FB10_error_I_prob_s)) * 1.15)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.6, color="0.75")
ax.legend(loc="upper left", fontsize=10)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
plt.show()
```
<img width="1903" height="1390" alt="BF10_U_test (1)" src="https://github.com/user-attachments/assets/e9ff5cae-c947-48e4-9603-3c22e3a59ae8" />

*Figure: False-positive rate as a function of per-group sample size, for the classical Mann-Whitney U test (p < 0.05) vs. the Bayesian rank sum test (BF10 > 3), under a negligible true effect size (≈0.033).*

## Requirements

- `jax`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
