# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Hierarchical Bayesian Inference of Normal Means #
# 
# Suppose we have $J$ observations $\bar{y}_{\cdot j} \mid \theta_j \sim \text{N}(\theta_j, \sigma_j^2)$, conditionally independent of each other given its parameters.  The variance parameter $\sigma_j^2$ is assumed to be known, and the mean parameter $\theta_j$'s are independently sampled from prior $\text{N}(\mu, \tau^2)$.  Finally, we assign uniform hyperprior distribution for $\mu$ given $\tau$:
# $$
# p(\mu,\tau) = p(\mu \mid \tau) p(\tau) \propto p(\tau).
# $$

# <codecell>

# prepare inline pylab
% pylab inline

# <markdowncell>

# ## BDA 5.5. Parallel Experiments in Eight Schools ##
# 
# Please refer to the corresponding chapter of the book for description of the data.  I typed the data myself:

# <codecell>

# load hand-typed data
from pandas import *
import numpy as np
school_data = DataFrame(data={'means':[28,8,-3,7,-1,1,18,12],'serrs':[15,10,16,11,9,11,10,18]},
                        index=['A','B','C','D','E','F','G','H'])

# <codecell>

# the data looks like this
school_data

# <markdowncell>

# Since mean parameters are conjugate to each other, only tricky part is the standard deviation parameter $\tau$ for group means.  We marginalize out all other variables to make inference on $\tau$.

# <codecell>

# define marginal log posterior function of tau
def log_posterior_tau_helper(tau, means, serrs, log_prior=lambda tau: 0):
    '''
    computes log marginal hyperparameter density of tau
    '''
    ret = log_prior(tau)
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    # precision weighted average
    mu_hat = np.sum((1.0/marginal_variances) * means) / \
            np.sum(1.0/marginal_variances)
    ret += 0.5 * log(total_variance)
    ret -= 0.5 * np.sum(np.log(marginal_variances))
    ret -= 0.5 * np.sum(((means - mu_hat) ** 2)/marginal_variances)
    return ret
# vectorize the function for convenience
log_posterior_tau = \
    np.vectorize(log_posterior_tau_helper, otypes=[np.float], excluded=[1,2,3])

# <markdowncell>

# Now let us reproduce Figure 5.5, which draws marginal posterior density $p(\tau \mid y)$.

# <codecell>

import pylab as pl
# grid points to evaluate evaluate density function
tau_min = 0; tau_max = 30; tau_grid_num = 1000
tau_knots = np.linspace(tau_min, tau_max, tau_grid_num)
lop_posterior_tau_densities = \
    log_posterior_tau(tau_knots, school_data.means, 
                      school_data.serrs, log_prior=lambda tau: 0)
# when calculating densities, it is numerically more stable to 
# first compute in log space, subtract the maximum, and then exponentiate
posterior_tau_densities = \
        np.exp(lop_posterior_tau_densities - np.max(lop_posterior_tau_densities))
#np.exp(log_posterior_tau_densities - log_posterior_tau_densities.max())
pl.plot(tau_knots, posterior_tau_densities, color='k', linestyle='-', linewidth=1)
pl.axes().set_xlabel(r'$\tau$')
# the y-axis of unnormalized posterior means nothing, so rather hide the scale
pl.axes().get_yaxis().set_visible(False)

# <markdowncell>

# This plot indicates that values of $\tau$ near zero are most plausible, but large values such as 10 or 15 are still quite likely.
# 
# Now, we study the change of $\mathbb{E}[\theta_j \mid \tau]$'s and $\text{sd}[\theta_j \mid \tau]$'s as a function of $\tau$.  This can be done by marginalizing out $\mu$ from the conditional distribution $\theta_j \mid \mu,\tau,y$.

# <codecell>

def mean_posterior_given_tau(tau, means, serrs):
    assert(len(means) == len(serrs))
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    # precision weighted average
    mu_hat = np.sum((1.0/marginal_variances) * means) / \
            np.sum(1.0/marginal_variances)
    if tau == 0:
        return np.repeat(mu_hat, len(means))
    return ((1.0/(serrs ** 2)) * means + 1.0/(tau ** 2) * mu_hat) / (1.0/(serrs ** 2) + 1.0/(tau ** 2))

def sd_posterior_given_tau(tau, serrs):
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    individual_variances = (((serrs ** 2)/(serrs ** 2 + tau ** 2)) ** 2) * total_variance
    if tau == 0:
        return np.sqrt(individual_variances)
    return np.sqrt(individual_variances + 1.0 / (1.0/(serrs ** 2) + 1.0/(tau ** 2)))

mean_posterior_given_taus = \
    np.vectorize(mean_posterior_given_tau, otypes=[np.ndarray], excluded=[1,2])
tau_conditional_means = \
        pandas.DataFrame.from_records(mean_posterior_given_taus(tau_knots, school_data.means, school_data.serrs),
                                      index = tau_knots, columns=school_data.index)
    
sd_posterior_given_taus = \
    np.vectorize(sd_posterior_given_tau, otypes=[np.ndarray], excluded=[1])
tau_conditional_sds = \
        pandas.DataFrame.from_records(sd_posterior_given_taus(tau_knots, school_data.serrs),
                                      index = tau_knots, columns=school_data.index)

# <codecell>

# I was not able to put inline labels in Python. 
# this page contains an example code for it: http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
tau_conditional_means.plot(style=['-','--','-.',':','-','--','-.',':'])
pl.legend(bbox_to_anchor=(1.05, 1), loc=2)
pl.ylim(-5,30)
pl.axes().set_xlabel(r'$\tau$')
pl.show()

# <markdowncell>

# When $\tau = 0$, all group means are treated to be the same, while as $\tau \rightarrow \infty$ they converge to individual MLEs.

# <codecell>

tau_conditional_sds.plot(style=['-','--','-.',':','-','--','-.',':'])
pl.legend(bbox_to_anchor=(1.05, 1), loc=2)
pl.ylim(0,20)
pl.axes().set_xlabel(r'$\tau$')
pl.show()

# <markdowncell>

# Naturally, larger $\tau$ means higher uncertainty in estimates.  
# 
# Now, let us sample parameters from the joint posterior distribution.

# <codecell>

# seed the RNG
np.random.seed(13531)
sample_num = 200
# first sample tau's
tau_probs = posterior_tau_densities / posterior_tau_densities.sum()
samples_tau = np.random.choice(tau_knots, sample_num, p=tau_probs)
# now sample mu conditioned on tau
def sample_means_from_tau(tau, means, serrs):
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    # precision weighted average
    mu_hat = np.sum((1.0/marginal_variances) * means) / \
            np.sum(1.0/marginal_variances)
    sample_mu = np.random.normal(loc=mu_hat, scale=np.sqrt(total_variance))
    
    conditional_means = \
        ((1.0/(serrs ** 2)) * means + 1.0/(tau ** 2) * sample_mu) /\
        (1.0/(serrs ** 2) + 1.0/(tau ** 2))
    conditional_scales = np.sqrt(1.0 / (1.0/(serrs ** 2) + 1.0/(tau ** 2)))
    return np.random.normal(loc=conditional_means,
                            scale=conditional_scales)
        
sample_means_from_taus = \
        np.vectorize(sample_means_from_tau, otypes=[np.ndarray], excluded=[1,2])

samples_df = DataFrame.from_records(sample_means_from_taus(samples_tau, school_data.means, school_data.serrs),
                                    columns=school_data.index)

# <markdowncell>

# Now that we have posterior samples, we can reproduce Table 5.3. which shows quantiles of posterior samples:

# <codecell>

samples_df.quantile([0.025,0.25,0.5,0.75,0.975]).transpose().apply(np.round)

# <markdowncell>

# Lower and upper quantiles deviate a bit from the book, but this is probably due to small sample size (200).
# 
# Figure 5.8 (a) reproduced below shows posterior samples for the effect in school A.

# <codecell>

samples_df['A'].hist(bins=20)
pl.axes().set_xlabel(r'Effect in School A')
pl.xlim(-20,60)

# <markdowncell>

# Figure 5.8 (b) reproduced below shows posterior samples of the largest effect.

# <codecell>

pl.hist(samples_df.max(axis=1), bins=20)
pl.axes().set_xlabel(r'Largest Effect')
pl.xlim(-20,60)

# <markdowncell>

# $\text{Pr}(\max\{ \theta_j \} > 28.4 \mid y)$ can be calculated as:

# <codecell>

np.mean(samples_df.max(axis=1) > 28.4)

# <markdowncell>

# Since the probability is small, the accuracy with only 200 samples is quite poor.
# 
# We can also compute $\text{Pr}(\theta_1 > \theta_3 \mid y)$:

# <codecell>

np.mean(samples_df['A'] > samples_df['C'])

# <codecell>


