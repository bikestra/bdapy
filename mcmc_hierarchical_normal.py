# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # MCMC on Hierarchical Normal Model (BDA 11.6) #

# <codecell>

% pylab inline

# <codecell>

import numpy as np
np.random.seed(13579)

# first write down the data
data = {'A':np.array([62,60,63,59]),
        'B':np.array([63,67,71,64,65,66]),
        'C':np.array([68,66,71,67,68,68]),
        'D':np.array([56,62,60,61,63,64,63,59]),
        }

# <codecell>

# define sampler for inverse chi-square
def r_inv_chisquare(size, df, scale_sq):
    return df * scale_sq / np.random.chisquare(df, size)

# <codecell>

# compute the empirical mean in each group
means = {key: data[key].mean() for key in data.keys()}

# <codecell>

# initialize theta and mu parameters
thetas = {key: np.random.choice(data[key]) for key in data.keys()}
mu = mean(thetas.values())

# <codecell>

from collections import defaultdict
from pandas import *
from scipy import stats

# J in the book
group_num = len(data)
# n in the book
num_points = np.sum([len(value) for value in data.values()])

num_samples = 400
num_chains = 5

sub_sample_dfs = []
for chain_index in range(num_chains):
    samples = defaultdict(list)
    for sample_index in range(num_samples):
        # sample tau
        tauhat_sq = np.sum((np.array(thetas.values()) - mu) ** 2)/(group_num-1)
        tau_sq = r_inv_chisquare(1, group_num - 1, tauhat_sq)[0]
        # sample sigma
        sigmahat_sq = 0
        for key in data.keys():
            sigmahat_sq += np.sum((data[key] - thetas[key]) ** 2)
        sigmahat_sq /= num_points
        sigma_sq = r_inv_chisquare(1, num_points, sigmahat_sq)[0]
        # sample mu
        muhat = np.mean(thetas.values())
        mu = np.random.normal(loc=muhat, scale=np.sqrt(tau_sq/group_num))
        # sample thetas
        for key in data.keys():
            precision = 1.0/tau_sq + len(data[key])/sigma_sq
            thetahat = (1.0/tau_sq * mu + len(data[key])/sigma_sq * means[key]) / precision
            V_theta = 1.0/precision
            thetas[key] = np.random.normal(loc=thetahat, scale=np.sqrt(V_theta))
        samples['tau'].append(np.sqrt(tau_sq))
        samples['sigma'].append(np.sqrt(sigma_sq))
        for key in data.keys():
            samples['theta_' + key].append(thetas[key])
        # compute log joint posterior probability (up to some constant)
        log_joint_posterior = np.log(np.sqrt(tau_sq))
        for key in data.keys():
            log_joint_posterior += stats.norm.logpdf(thetas[key], loc=mu, scale=np.sqrt(tau_sq))
            log_joint_posterior += np.sum(stats.norm.logpdf(data[key], loc=thetas[key], scale=sigma_sq))
        samples['log_joint'].append(log_joint_posterior)
    # take second half of each chain
    second_halfs = {key: value[len(value)/2:] for key, value in samples.iteritems()}
    # to break second half of each chain into two sequences, define sequence indices accordingly
    second_halfs['seq_index'] = \
            np.concatenate([np.repeat(chain_index * 2, len(second_halfs['tau'])/2),
                      np.repeat(chain_index * 2 + 1, len(second_halfs['tau']) - len(second_halfs['tau'])/2)])
    sub_sample_dfs.append(DataFrame(second_halfs))
sample_dfs = pandas.concat(sub_sample_dfs)

# <markdowncell>

# Now, Table 11.3 is reproduced!

# <codecell>

sample_dfs.quantile([0.025, 0.25, 0.5, 0.75, 0.975]).transpose()

# <markdowncell>

# It turns out, however, that some of $\hat{R}$ values are smaller than 1.0.  Did I do something wrong?

# <codecell>

seq_len = num_samples / 4.0
W_values = sample_dfs.groupby('seq_index').var(ddof=1).mean(axis=0)
B_values = sample_dfs.groupby('seq_index').mean().var(axis=0,ddof=1) * (seq_len)
print "Rhat values:"
print sqrt((seq_len-1.0)/seq_len + 1.0/seq_len * (B_values / W_values))

