import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrn
import scipy
import pandas as pd
from jax.typing import ArrayLike

def Bayes_rank_sum_test(X: ArrayLike, Y: ArrayLike) -> jax.Array:
  """
  Bayesian rank sum (Mann-Whitney) test via a latent-normal Gibbs sampler.

  Computes the Bayes factor BF10 comparing:
    H1: the two groups differ (effect size delta ~ Cauchy(0, gamma))
    H0: the two groups are identical (delta = 0)

  Reference
  ---------
  van Doorn, J., Ly, A., Marsman, M., & Wagenmakers, E.-J. (2020).
  Bayesian rank-based hypothesis testing for the rank sum test, the
  signed rank test, and Spearman's rho. Journal of Applied Statistics,
  47(16), 2984-3006. https://doi.org/10.1080/02664763.2019.1709053
  PMID: 35707708, PMCID: PMC9041780

  Implementation notes
  ---------------------
  Posterior sampling follows the Gibbs sampler described in the
  original article (data augmentation on latent scores Z, subject to
  the ordinal constraints implied by the observed ranks), with two
  additions on top of the paper's base algorithm:

    - A decorrelation step (Liu & Sabatti, 2000; Morey, Rouder &
      Speckman, 2008): after each per-element update of Z, a single
      shared shift is drawn from its exact conditional distribution
      and added to all latent scores at once. This move can never
      change the relative order of the Z's (so it's always valid),
      and it substantially reduces the autocorrelation that the
      element-wise data-augmentation updates alone are prone to.
    - Numerically stable initialization of the truncation boundaries
      (a, b) for Z: the very first draw of Z is bounded using normal
      scores (rank -> standard-normal quantile) rather than the raw
      data values. Raw data can be on an arbitrary or heavy-tailed
      scale (e.g. outliers), which can push the initial truncated-
      normal sampling into a region where the standard normal CDF
      saturates to exactly 0 or 1 in floating point, producing NaNs.
      Normal scores are rank-equivalent (so the ordinal constraint is
      unaffected) but always lie on a well-behaved unit-normal scale.

  The Bayes factor itself is obtained via the Savage-Dickey density
  ratio (evaluated in log-space for numerical stability): BF10 is the
  ratio of the prior density of delta at 0 to the posterior density of
  delta at 0, estimated from the post-burn-in MCMC samples via KDE.

  Parameters
  ----------
  X, Y : ArrayLike
      The two independent samples to compare (1-D, shape (n_X,) and
      (n_Y,) respectively; n_X and n_Y need not be equal). Accepts
      numpy arrays, JAX arrays, or plain Python sequences.

  Returns
  -------
  BF10 : jax.Array (scalar)
      Bayes factor in favor of H1 (a difference between groups) over
      H0 (no difference). BF10 > 1 favors H1; BF10 < 1 favors H0.
  """

  X, Y = jnp.sort(X), jnp.sort(Y)
  n_X, n_Y = X.shape[0], Y.shape[0]
  n = n_X + n_Y
  decor_sigma = jnp.sqrt(1./n)
  XY = jnp.concatenate([X,Y])
  ranked_XY = jax.scipy.stats.rankdata(XY)
  normal_scores = jax.scipy.stats.norm.ppf((ranked_XY - 0.5) / n)
  a_mat = ranked_XY<ranked_XY[..., None]
  b_mat = ranked_XY>ranked_XY[..., None]

  # init a and b
  a = jnp.max(jnp.where(a_mat, normal_scores, -jnp.inf), axis=-1)
  b = jnp.min(jnp.where(b_mat, normal_scores, jnp.inf), axis=-1)

  #sample init

  key = jrn.key(57334)

  #1. g
  gamma = 5.
  scale = (gamma**2)/2.
  key, subkey = jrn.split(key)
  g = 1./(jrn.gamma(subkey, 0.5)/scale)

  #2. delta
  key, subkey = jrn.split(key)
  delta = jnp.sqrt(g)*jrn.normal(subkey)

  #3. Z
  mean_X, mean_Y = -0.5*delta, 0.5*delta
  key, subkey = jrn.split(key)
  Z_X = mean_X + jrn.truncated_normal(subkey, lower=a[:n_X]-mean_X, upper=b[:n_X]-mean_X)
  key, subkey = jrn.split(key)
  Z_Y = mean_Y + jrn.truncated_normal(subkey, lower=a[n_X:]-mean_Y, upper=b[n_X:]-mean_Y)

  carry = [key, g, delta,  Z_X, Z_Y]

  def one_Gibbs_step(carry, iter):
    key, g, delta,  Z_X, Z_Y = carry

    # 0. a and b
    Z = jnp.concatenate([Z_X, Z_Y])
    a = jnp.max(jnp.where(a_mat, Z, -jnp.inf), axis=-1)
    b = jnp.min(jnp.where(b_mat, Z, jnp.inf), axis=-1)

    #1. Z
    mean_X, mean_Y = -0.5*delta, 0.5*delta
    key, subkey = jrn.split(key)
    Z_X = mean_X + jrn.truncated_normal(subkey, lower=a[:n_X]-mean_X, upper=b[:n_X]-mean_X)
    key, subkey = jrn.split(key)
    Z_Y = mean_Y + jrn.truncated_normal(subkey, lower=a[n_X:]-mean_Y, upper=b[n_X:]-mean_Y)

    #decorrelation step

    S = Z_X.sum() + Z_Y.sum() + delta*(n_X-n_Y)/2.
    key, subkey = jrn.split(key)
    c = -S/n + decor_sigma*jrn.normal(subkey)
    Z_X, Z_Y = Z_X+c, Z_Y+c

    #2. delta
    key, subkey = jrn.split(key)
    mu = 2*g*(n_Y*Z_Y.mean() - n_X*Z_X.mean())/(g*n+4.)
    sigma = 4*g/(g*n+4.)
    delta = jnp.sqrt(sigma)*jrn.normal(subkey) + mu

    #3. g
    key, subkey = jrn.split(key)
    scale = (gamma**2 + delta**2)/2.
    g = 1./(jrn.gamma(subkey, 1.)/scale)

    carry = [key, g, delta,  Z_X, Z_Y]
    return carry, dict(g=g, delta=delta,  Z_X=Z_X, Z_Y=Z_Y)

  final_carry, idata = jax.lax.scan(one_Gibbs_step, init=carry, xs=jnp.arange(2000))
  delta_post = idata['delta'][1000:]
  kde = jax.scipy.stats.gaussian_kde(delta_post)
  logBF10 = jax.scipy.stats.cauchy.logpdf(0., loc=0., scale=gamma) - kde.logpdf(0.)
  BF10 = jnp.exp(logBF10)
  return BF10


### Batched (parallel) computation
#
# `Bayes_rank_sum_test_Batch` runs `Bayes_rank_sum_test` across many
# `(X, Y)` sample-group pairs simultaneously, using `jax.vmap` to
# vectorize the Gibbs sampler over an extra leading batch dimension
# and `jax.jit` to compile the whole batch into a single fused,
# hardware-accelerated computation.
#
# Use this when you need to run the test many times in parallel with
# the same sample sizes per group (e.g. simulation studies, power /
# type-I-error analyses across many synthetic datasets, or testing
# many features/variables at once) rather than in a Python loop.
#
# Input shape: `X` and `Y` should each be arrays of shape
# `(n_batch, n_X)` and `(n_batch, n_Y)` respectively (equal sample
# sizes within each array, but `n_X` need not equal `n_Y`).
# Output shape: `(n_batch,)`, one `BF10` per row.
Bayes_rank_sum_test_Batch = jax.jit(jax.vmap(Bayes_rank_sum_test))
