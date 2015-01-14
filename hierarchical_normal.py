# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Hierarchical Bayesian Inference of Normal Means #
# 
# This notebook is based on Chapter 5.4, 5.5 and 5.6 of 'Bayesian Data Analysis' (Gelman et. al, 3rd Edition).
# 
# Suppose we have $J$ observations $\bar{y}_{\cdot j} \mid \theta_j \sim \text{N}(\theta_j, \sigma_j^2)$, conditionally independent of each other given its parameters.  The variance parameter $\sigma_j^2$ is assumed to be known, and the mean parameter $\theta_j$'s are independently sampled from prior $\text{N}(\mu, \tau^2)$.  Finally, we assign uniform hyperprior distribution for $\mu$ given $\tau$:
# $$
# p(\mu,\tau) = p(\mu \mid \tau) p(\tau) \propto p(\tau).
# $$

# <codecell>

# prepare inline pylab
% pylab inline

# <markdowncell>

# ## Parallel Experiments in Eight Schools (BDA 5.5) ##
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
def sample_means_from_tau(tau, means, serrs, include_mu=False):
    if isinf(tau):
        if include_mu:
            raise ValueError("mu is not well defined when tau is infinity")
        else:
            return np.random.normal(loc=means, scale=serrs)

    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    # precision weighted average
    mu_hat = np.sum((1.0/marginal_variances) * means) / \
            np.sum(1.0/marginal_variances)
    sample_mu = np.random.normal(loc=mu_hat, scale=np.sqrt(total_variance))
    if tau == 0:
        if include_mu:
            return np.repeat(sample_mu, len(means) + 1)
        else:
            return np.repeat(sample_mu, len(means))
        
    conditional_means = \
        ((1.0/(serrs ** 2)) * means + 1.0/(tau ** 2) * sample_mu) /\
        (1.0/(serrs ** 2) + 1.0/(tau ** 2))
    conditional_scales = np.sqrt(1.0 / (1.0/(serrs ** 2) + 1.0/(tau ** 2)))
    if include_mu == True:
        return np.concatenate(([sample_mu],
                               np.random.normal(loc=conditional_means,
                                                scale=conditional_scales)))
    else:
        return np.random.normal(loc=conditional_means,
                                scale=conditional_scales)
        
sample_means_from_taus = \
        np.vectorize(sample_means_from_tau, otypes=[np.ndarray], excluded=[1,2,3])

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

# <markdowncell>

# ## Model Checking (BDA 6.5) ##
# 
# To check the model fit, we sample predictive data from posterior distribution and then measure its test statistics.

# <codecell>

samples_df.head(20)

# <codecell>

posterior_data_list = []
for row_index, row in samples_df.iterrows():
    posterior_data_list.append( np.random.normal(loc=row, scale=school_data.serrs) )
posterior_data_df = DataFrame.from_records(posterior_data_list,
                                           columns=school_data.index)

# <codecell>

pl.hist(posterior_data_df.max(axis=1), bins=20)
pl.axes().set_xlabel(r'T(y) = max(y)')
pl.axvline(school_data.means.max(), color='r', linestyle='dashed', linewidth=2)
print "p-value: %.2f" % (np.sum(posterior_data_df.max(axis=1) > school_data.means.max())/float(len(posterior_data_df)))

# <codecell>

pl.hist(posterior_data_df.min(axis=1), bins=20)
pl.axes().set_xlabel(r'T(y) = min(y)')
pl.axvline(school_data.means.min(), color='r', linestyle='dashed', linewidth=2)
print "p-value: %.2f" % (np.sum(posterior_data_df.min(axis=1) > school_data.means.min())/float(len(posterior_data_df)))

# <codecell>

pl.hist(posterior_data_df.mean(axis=1), bins=20)
pl.axes().set_xlabel(r'T(y) = mean(y)')
pl.axvline(school_data.means.mean(), color='r', linestyle='dashed', linewidth=2)
print "p-value: %.2f" % (np.sum(posterior_data_df.mean(axis=1) > school_data.means.mean())/float(len(posterior_data_df)))

# <codecell>

pl.hist(posterior_data_df.std(axis=1), bins=20)
pl.axes().set_xlabel(r'T(y) = sd(y)')
pl.axvline(school_data.means.std(), color='r', linestyle='dashed', linewidth=2)
print "p-value: %.2f" % (np.sum(posterior_data_df.std(axis=1) > school_data.means.std())/float(len(posterior_data_df)))

# <markdowncell>

# Surprisingly, these p-values are too close to what we have in the book, considering the small data size!

# <markdowncell>

# ## Model Comparison based on predictive performance (BDA 7.3) ##
# 
# (Sorry for extreme low quality code with a lot of redundancies here; I quickly began tosmell something bad, but was too lazy to go back.)

# <codecell>

from scipy.stats import *
# computing AIC
# first, compute MLE estimate
marginal_variances = school_data.serrs ** 2
total_variance = 1.0/np.sum(1.0/marginal_variances)
# precision weighted average
mle_complete_pooling = np.sum((1.0/marginal_variances) * school_data.means) / np.sum(1.0/marginal_variances)
mle_lpd_complete_pooling = norm.logpdf(school_data.means, loc=mle_complete_pooling, scale=school_data.serrs).sum()
mle_lpd_no_pooling = norm.logpdf(school_data.means, loc=school_data.means, scale=school_data.serrs).sum()
print "AIC for complete pooling: %.1f" % (-2 * mle_lpd_complete_pooling + 1 * 2)
print "AIC for no pooling: %.1f" % (-2 * mle_lpd_no_pooling + 8 * 2)

# <codecell>

# computing DIC
# here we increase sample number to improve the precision
sample_num = 2000
# sample 
complete_pooling_samples_df = DataFrame.from_records(sample_means_from_taus(np.repeat(0,sample_num), school_data.means, school_data.serrs),
                                                     columns=school_data.index)
no_pooling_samples_df = DataFrame.from_records(sample_means_from_taus(np.repeat(float('inf'),sample_num), school_data.means, school_data.serrs),
                                               columns=school_data.index)

samples_tau = np.random.choice(tau_knots, sample_num, p=tau_probs)
hierarchical_samples_df = DataFrame.from_records(sample_means_from_taus(samples_tau, school_data.means, school_data.serrs),
                                                 columns=school_data.index)
# first, compute expected mean from posterior distribution
pmean_no_pooling = no_pooling_samples_df.mean()
pmean_complete_pooling = complete_pooling_samples_df.mean()
pmean_hierarchical = hierarchical_samples_df.mean()

pmean_lpd_no_pooling = norm.logpdf(school_data.means, loc=pmean_no_pooling, scale=school_data.serrs).sum()
pmean_lpd_complete_pooling = norm.logpdf(school_data.means, loc=pmean_complete_pooling, scale=school_data.serrs).sum()
pmean_lpd_hierarchical = norm.logpdf(school_data.means, loc=pmean_hierarchical, scale=school_data.serrs).sum()

print "-2lpd with no pooling: %.1f" % (-2 * pmean_lpd_no_pooling)
print "-2lpd with complete pooling: %.1f" % (-2 * pmean_lpd_complete_pooling)
print "-2lpd with hierarchical pooling: %.1f" % (-2 * pmean_lpd_hierarchical)

pdic_no_pooling = 2 * (pmean_lpd_no_pooling - \
    no_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                      scale=school_data.serrs),
                                axis=1).sum(axis=1).mean())
pdic_complete_pooling = 2 * (pmean_lpd_complete_pooling - \
    complete_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                            scale=school_data.serrs), 
                                      axis=1).sum(axis=1).mean())
pdic_hierarchical = 2 * (pmean_lpd_hierarchical - \
    hierarchical_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                        scale=school_data.serrs), axis=1).sum(axis=1).mean())
print "p_DIC for no pooling: %.1f" % (pdic_no_pooling)
print "p_DIC for complete pooling: %.1f" % (pdic_complete_pooling)
print "p_DIC for hierarchical pooling: %.1f" % (pdic_hierarchical)

print "DIC for no pooling: %.1f" % (-2 * pmean_lpd_no_pooling + 2 * pdic_no_pooling)
print "DIC for complete pooling: %.1f" % (-2 * pmean_lpd_complete_pooling + 2 * pdic_complete_pooling)
print "DIC for hierarchical pooling: %.1f" % (-2 * pmean_lpd_hierarchical + 2 * pdic_hierarchical)

# <codecell>

# WAIC computation
# here I am not being careful in numerical precision; in principle I should've done these calculation in log space.
lppd_no_pooling = np.log(no_pooling_samples_df.apply(lambda x: norm.pdf(school_data.means, loc=x, 
                                                     scale=school_data.serrs), axis=1).mean(axis=0)).sum()
lppd_complete_pooling = np.log(complete_pooling_samples_df.apply(lambda x: norm.pdf(school_data.means, loc=x, 
                                                                 scale=school_data.serrs), axis=1).mean(axis=0)).sum()
lppd_hierarchical = np.log(hierarchical_samples_df.apply(lambda x: norm.pdf(school_data.means, loc=x, 
                                                                            scale=school_data.serrs), axis=1).mean(axis=0)).sum()
print "-2lppd with no pooling: %.1f" % (-2 * lppd_no_pooling)
print "-2lppd with complete pooling: %.1f" % (-2 * lppd_complete_pooling)
print "-2lppd with hierarchical pooling: %.1f" % (-2 * lppd_hierarchical)

pwaic_1_no_pooling = 2 * (lppd_no_pooling - 
                          no_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                      scale=school_data.serrs), axis=1).mean(axis=0).sum())
pwaic_1_complete_pooling = 2 * (lppd_complete_pooling - 
                                complete_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                                  scale=school_data.serrs), axis=1).mean(axis=0).sum())
pwaic_1_hierarchical = 2 * (lppd_hierarchical - 
                            hierarchical_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                          scale=school_data.serrs), axis=1).mean(axis=0).sum())

print "p_WAIC1 with no pooling: %.1f" % (pwaic_1_no_pooling)
print "p_WAIC1 with complete pooling: %.1f" % (pwaic_1_complete_pooling)
print "p_WAIC1 with hierarchical pooling: %.1f" % (pwaic_1_hierarchical)

pwaic_2_no_pooling = no_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                 scale=school_data.serrs), axis=1).var(axis=0).sum()
pwaic_2_complete_pooling = complete_pooling_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                 scale=school_data.serrs), axis=1).var(axis=0).sum()
pwaic_2_hierarchical = hierarchical_samples_df.apply(lambda x: norm.logpdf(school_data.means, loc=x, 
                                                 scale=school_data.serrs), axis=1).var(axis=0).sum()

print "p_WAIC2 with no pooling: %.1f" % (pwaic_2_no_pooling)
print "p_WAIC2 with complete pooling: %.1f" % (pwaic_2_complete_pooling)
print "p_WAIC2 with hierarchical pooling: %.1f" % (pwaic_2_hierarchical)

print "WAIC with no pooling: %.1f" % (-2 * (lppd_no_pooling - pwaic_2_no_pooling))
print "WAIC with complete pooling: %.1f" % (-2 * (lppd_complete_pooling - pwaic_2_complete_pooling))
print "WAIC with hierarchical pooling: %.1f" % (-2 * (lppd_hierarchical - pwaic_2_hierarchical))

# <markdowncell>

# LOOC is omitted since I am too lazy.

# <markdowncell>

# ## Clinical Trials of Beta-blockers (BDA 5.6) ##
# 
# Here, we perform a meta-analysis which estimates the effect of beta-blockers from 22 clinical trials.  Since a lot of plots are omitted in this chapter of the book, this notebook might be interesting for someone who wanted to take a deeper look on this analysis. 
# 
# The data can be retrieved from the BDA book website:

# <codecell>

blocker_data = read_csv("http://www.stat.columbia.edu/~gelman/book/data/meta.asc",
                        index_col='study', skiprows=3, delim_whitespace=True)

# <codecell>

blocker_data

# <markdowncell>

# Our estimand in this analysis is log odds ratio; this is estimated by empirical logit which sampling variance is estimated by eq (5.24).  We regard these values as means and variances of hierarchical normal means model.  First, let us calculate these values:

# <codecell>

blocker_data['means'] = np.log(blocker_data['treated.deaths']\
                               /(blocker_data['treated.total']-blocker_data['treated.deaths'])) \
                        - np.log(blocker_data['control.deaths']\
                               /(blocker_data['control.total']-blocker_data['control.deaths']))
blocker_data['serrs'] = np.sqrt(1.0/blocker_data['treated.deaths'] + 
                                1.0/(blocker_data['treated.total'] - 1.0/blocker_data['treated.deaths']) + 
                                1.0/blocker_data['control.deaths'] + 
                                1.0/(blocker_data['control.total'] - 1.0/blocker_data['control.deaths']))                                

# <codecell>

import pylab as pl
# grid points to evaluate evaluate density function
tau_min = 0; tau_max = 0.6; tau_grid_num = 1000
tau_knots = np.linspace(tau_min, tau_max, tau_grid_num)
lop_posterior_tau_densities = \
    log_posterior_tau(tau_knots, blocker_data.means, 
                      blocker_data.serrs, log_prior=lambda tau: 0)
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

# As mentioned in the book, the marginal posterior density function of $\tau$ peaks at nonzero value, but still values around zero are quite plausible.
# 
# The plot below shows mean effects conditioned on $\tau$; combined with the information from plot above, this indicates that moderate amount of shrinkage in estimates are necessary.

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
        pandas.DataFrame.from_records(mean_posterior_given_taus(tau_knots, blocker_data.means, blocker_data.serrs),
                                      index=tau_knots, columns=blocker_data.index)
    
sd_posterior_given_taus = \
    np.vectorize(sd_posterior_given_tau, otypes=[np.ndarray], excluded=[1])
tau_conditional_sds = \
        pandas.DataFrame.from_records(sd_posterior_given_taus(tau_knots, blocker_data.serrs),
                                      index = tau_knots, columns=blocker_data.index)
# I was not able to put inline labels in Python. 
# this page contains an example code for it: http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
tau_conditional_means.plot()
pl.legend(bbox_to_anchor=(1.05, 1), loc=2)
pl.ylim(-1,0.5)
pl.axes().set_xlabel(r'$\tau$')
pl.show()

# <markdowncell>

# Now let us sample from the joint posterior distribution.

# <codecell>

# seed the RNG
np.random.seed(1353135)
sample_num = 5000
# first sample tau's
tau_probs = posterior_tau_densities / posterior_tau_densities.sum()
samples_tau = np.random.choice(tau_knots, sample_num, p=tau_probs)
# now sample mu conditioned on tau
samples_df = DataFrame.from_records(sample_means_from_taus(samples_tau, blocker_data.means, 
                                                           blocker_data.serrs, include_mu=True),
                                    columns=np.concatenate((['mu'],blocker_data.index)))

# <codecell>

samples_columns_df = samples_df.drop('mu', 1)

# <markdowncell>

# Posterior quantiles in Table 5.4 can now be reproduced.

# <codecell>

samples_columns_df.quantile([0.025,0.25,0.5,0.75,0.975]).transpose().apply(lambda x:np.round(x,decimals=2))

# <markdowncell>

# Now let us sample predicted effects.

# <codecell>

samples_predicted_effect = np.zeros(len(samples_tau))
samples_predicted_effect[samples_tau > 0] = np.random.normal(loc=samples_df['mu'][samples_tau > 0],scale=samples_tau[samples_tau > 0])
samples_predicted_effect[samples_tau == 0] = samples_df['mu'][samples_tau == 0]

# <markdowncell>

# Now Table 5.5 can be reproduced.

# <codecell>

DataFrame({'mean':samples_df['mu'], 
           'standard deviation':samples_tau, 
           'predicted_effect':samples_predicted_effect}).quantile([0.025,0.25,0.5,0.75,0.975]).transpose().apply(lambda x:np.round(x,decimals=2))

# <markdowncell>

# Histograms of estimates, not shown in the book, are presented below.

# <codecell>

samples_columns_df.hist(figsize=(12,12),sharex=True, bins=20)

# <codecell>

def mean_posterior_of_mean_given_tau(tau, means, serrs):
    assert(len(means) == len(serrs))
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    # precision weighted average
    mu_hat = np.sum((1.0/marginal_variances) * means) / \
            np.sum(1.0/marginal_variances)
    return mu_hat

def sd_posterior_of_mean_given_tau(tau, means, serrs):
    assert(len(means) == len(serrs))
    marginal_variances = serrs ** 2 + tau ** 2
    total_variance = 1.0/np.sum(1.0/marginal_variances)
    return np.sqrt(total_variance)

mean_posterior_of_mean_given_taus = \
    np.vectorize(mean_posterior_of_mean_given_tau, otypes=[np.ndarray], excluded=[1,2])
sd_posterior_of_mean_given_taus = \
    np.vectorize(sd_posterior_of_mean_given_tau, otypes=[np.ndarray], excluded=[1,2])

# <markdowncell>

# Recall this was the posterior density of $\tau$:

# <codecell>

pl.plot(tau_knots, posterior_tau_densities, color='k', linestyle='-', linewidth=1)
pl.axes().set_xlabel(r'$\tau$')
# the y-axis of unnormalized posterior means nothing, so rather hide the scale
pl.axes().get_yaxis().set_visible(False)

# <markdowncell>

# Compared to the posterior quantile values of data, the conditional $\mu \mid \tau$ changes very little within most plausible region of $\tau$.

# <codecell>

pl.plot(tau_knots, mean_posterior_of_mean_given_taus(tau_knots, blocker_data.means, blocker_data.serrs))
#pl.ylim((-0.6,0.1))
pl.axes().set_xlabel(r'$E[\mu \mid \tau,y]$')

# <markdowncell>

# However, the standard deviation changes greatly.

# <codecell>

pl.plot(tau_knots, sd_posterior_of_mean_given_taus(tau_knots, blocker_data.means, blocker_data.serrs))
pl.axes().set_xlabel(r'sd$[\mu \mid \tau,y]$')

# <markdowncell>

# The book comments the value of $\text{sd}(\mu \mid \tau,y)$ at $\tau=0.13$:

# <codecell>

sd_posterior_of_mean_given_tau(0.13, blocker_data.means, blocker_data.serrs)

# <markdowncell>

# On the other hand, $\text{sd}(\mu \mid y)$, marginalized over $\tau$, can be computed from posterior samples as follows:

# <codecell>

samples_df['mu'].std()

# <markdowncell>

# Well this is a bit different from the value in the book, 0.071.  I wonder why...?

# <codecell>


