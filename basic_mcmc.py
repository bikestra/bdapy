# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Basic Markov-Chain Monte-Carlo (MCMC) Sampling #
# 
# ## Gibbs Sampling from Bivariate Normal Distribution (BDA 11.1) ##
# 
# Here, we sample from a bivariate normal distribution using Gibbs sampling, although it is not simple to draw from actual joint distribution.  The posterior distribution is assumed to be:
# $$
# \left( \begin{matrix} \theta_1 \\ \theta_2 \end{matrix} \middle) \right| y
# \sim
# \text{N}
# \left(
#   \left( \begin{matrix} y_1 \\ y_2 \end{matrix} \right),
#   \left( \begin{matrix} 1 & \rho \\ \rho & 1 \end{matrix} \right)
# \right).
# $$
# The conditional distributions are:
# $$
# \theta_1 \mid \theta_2, y \sim \text{N}(y_1 + \rho(\theta_2 - y_2), 1-\rho^2),
# $$
# $$
# \theta_2 \mid \theta_1, y \sim \text{N}(y_2 + \rho(\theta_1 - y_1), 1-\rho^2).
# $$

# <codecell>

# prepare inline pylab
% pylab inline
import numpy as np
import pylab as plt
np.random.seed(13531)

# <codecell>

def gibbs_bivariate(y1, y2, rho, start_theta1, start_theta2, num_samples):
    scale = np.sqrt(1.0 - rho ** 2)
    theta1_samples = [start_theta1]
    theta2_samples = [start_theta2]
    current_theta1 = start_theta1
    current_theta2 = start_theta2
    for i in xrange(num_samples):
        current_theta1 = np.random.normal(loc=y1 + rho * (current_theta2 - y2), scale=scale)
        theta1_samples.append(current_theta1)
        theta2_samples.append(current_theta2)
        
        current_theta2 = np.random.normal(loc=y2 + rho * (current_theta1 - y1), scale=scale)
        theta1_samples.append(current_theta1)
        theta2_samples.append(current_theta2)
    return theta1_samples, theta2_samples

# <markdowncell>

# This is Figure 11.2 (a).

# <codecell>

starting_points = [(2.5,2.5),(-2.5,2.5),(2.5,-2.5),(-2.5,-2.5),(0,0)]
plt.plot(zip(*starting_points)[0], zip(*starting_points)[1], 'ks')
for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = gibbs_bivariate(0, 0, 0.8, start_theta1, start_theta2, 10)
    plt.plot(theta1_samples, theta2_samples, color='k', linestyle='-', linewidth=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# This is Figure 11.2 (b).

# <codecell>

plt.plot(zip(*starting_points)[0], zip(*starting_points)[1], 'ks')
for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = gibbs_bivariate(0, 0, 0.8, start_theta1, start_theta2, 500)
    plt.plot(theta1_samples, theta2_samples, color='k', linestyle='-', linewidth=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# This is Figure 11.2 (c)

# <codecell>

for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = gibbs_bivariate(0, 0, 0.8, start_theta1, start_theta2, 500)
    plt.scatter(theta1_samples[len(theta1_samples)/2:], theta2_samples[len(theta2_samples)/2:], marker='.', s=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# ## Metropolis Sampling from Bivariate Normal (BDA 11.3) ##
# 
# Here, we are sampling from $p(\theta \mid y) = \text{N}(\theta \mid 0, I)$, where $I$ is a $2 \times 2$ identity matrix.  We use $J_t(\theta^* \mid \theta^{t-1}) = \text{N}(\theta^* \mid \theta^{t-1}, 0.2^2 I)$ as a proposal distribution, which is quite inefficient.

# <codecell>

import scipy.stats as stats
from scipy.stats import norm
def metropolis_bivariate(y1, y2, start_theta1, start_theta2, num_samples, 
                         include_start=True, include_loglik=False):
    if include_start:
        theta1_samples = [start_theta1]
        theta2_samples = [start_theta2]
    else:
        theta1_samples = []
        theta2_samples = []
        
    loglik_samples = []
    current_theta1 = start_theta1
    current_theta2 = start_theta2
    # somehow the book is using unnormalized log probability. don't know why.
    current_log_prob = norm.logpdf((current_theta1,current_theta2),loc=(0,0),scale=(1,1)).sum() \
            - norm.logpdf((0,0),loc=(0,0),scale=(1,1)).sum()
    
    for i in xrange(num_samples):
        proposal_theta1, proposal_theta2 = np.random.normal(loc=(current_theta1, current_theta2),
                                                            scale=(0.2,0.2))
        proposal_log_prob = norm.logpdf((proposal_theta1,proposal_theta2),
                                         loc=(0,0),scale=(1,1)).sum() \
                            - norm.logpdf((0,0),
                                         loc=(0,0),scale=(1,1)).sum()
        if proposal_log_prob > current_log_prob:
            flag_accept = True
        else:
            acceptance_prob = np.exp(proposal_log_prob - current_log_prob)
            if np.random.random() < acceptance_prob:
                flag_accept = True
            else:
                flag_accept = False
        
        if flag_accept:
            current_theta1 = proposal_theta1
            current_theta2 = proposal_theta2
            current_log_prob = proposal_log_prob
            
        theta1_samples.append(current_theta1)
        theta2_samples.append(current_theta2)
        loglik_samples.append(current_log_prob)
    
    if include_loglik:
        return theta1_samples, theta2_samples, loglik_samples
    else:
        return theta1_samples, theta2_samples

# <markdowncell>

# This is Figure 11.1 (a).

# <codecell>

starting_points = [(2.5,2.5),(-2.5,2.5),(2.5,-2.5),(-2.5,-2.5)]
plt.plot(zip(*starting_points)[0], zip(*starting_points)[1], 'ks')
for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = metropolis_bivariate(0, 0, start_theta1, start_theta2, 50)
    plt.plot(theta1_samples, 
             theta2_samples, 
             color='k', linestyle='-', linewidth=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# This is Figure 11.1 (b).

# <codecell>

starting_points = [(2.5,2.5),(-2.5,2.5),(2.5,-2.5),(-2.5,-2.5)]
plt.plot(zip(*starting_points)[0], zip(*starting_points)[1], 'ks')
for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = metropolis_bivariate(0, 0, start_theta1, start_theta2, 1000)
    plt.plot(theta1_samples, 
             theta2_samples, 
             color='k', linestyle='-', linewidth=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# This is Figure 11.1 (c).

# <codecell>

for start_theta1, start_theta2 in starting_points:
    theta1_samples, theta2_samples = metropolis_bivariate(0, 0, start_theta1, start_theta2, 1000)
    theta1_samples_tail = theta1_samples[len(theta1_samples)/2:]
    theta2_samples_tail = theta2_samples[len(theta2_samples)/2:]
    plt.scatter(theta1_samples_tail + np.random.uniform(low=-0.001, high=0.001, size=len(theta1_samples_tail)),
                theta2_samples_tail + np.random.uniform(low=-0.001, high=0.001, size=len(theta2_samples_tail)),
                marker='.', s=1)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axes().set_aspect('equal')

# <markdowncell>

# ## Assessing Convergence (BDA 11.4) ##
# 
# $\hat{R}$ can be used to monitor the convergence of MCMC sampling.  Table 11.1 is reproduced here:

# <codecell>

from pandas import *
for num_iter in [50,500,2000,5000]:
    seq_len = num_iter/2.0
    dfs = []
    print "When the number of iterations is: %d" % num_iter
    for sequence_num, (start_theta1, start_theta2) in enumerate(starting_points):
        theta1_samples, theta2_samples, loglik_samples =\
                metropolis_bivariate(0, 0, start_theta1, start_theta2, num_iter,
                                     False, True)
        # convert samples into dataframe
        theta1_samples_tail = theta1_samples[len(theta1_samples)/2:]
        theta2_samples_tail = theta2_samples[len(theta2_samples)/2:]
        loglik_samples_tail = loglik_samples[len(loglik_samples)/2:]
        seqnum = concatenate([repeat(sequence_num * 2, len(theta1_samples_tail)/2),
                              repeat(sequence_num * 2 + 1, 
                                     len(theta1_samples_tail) - len(theta1_samples_tail)/2)])
        df = DataFrame({'theta1':theta1_samples_tail,'theta2':theta2_samples_tail,
                        'loglik':loglik_samples_tail,'seqnum':seqnum})
        dfs.append(df)
    whole_df = pandas.concat(dfs)
    print whole_df.quantile([0.025,0.5,0.975])
    W_values = whole_df.groupby('seqnum').var().mean(axis=0)
    B_values = whole_df.groupby('seqnum').mean().var(axis=0) * (seq_len)
    print "Rhat values:"
    print sqrt((seq_len-1.0)/seq_len + 1.0/seq_len * (B_values / W_values))
    print "\n\n"

