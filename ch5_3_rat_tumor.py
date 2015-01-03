# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Hierarchical Bayesian Inference of Binomial Probabilities (BDA 5.3) #
# 
# This notebook attempts to reproduce analysis in Chapter 5.3 of the book 'Bayesian Data Analysis' by Gelman et al (3rd Edition).  
# 
# Here, we are given $J=71$ observations of binomial outcomes $(y_1,N_1), \ldots, (y_J,N_J)$ where $y_j \sim \text{Bin}(N_j, \theta_j)$.  We assume conjugate prior $\theta_j \sim \text{Beta}(\alpha,\beta)$, and a "reasonable" hyperprior density $p(\alpha,\beta) \propto (\alpha+\beta)^{-5/2}$; refer to the chapter regarding why this is a reasonable choice.  Given this hierarchical model, we perform fully Bayesian analysis.

# <codecell>

% pylab inline

# <codecell>

# load data from Andrew's homepage
from pandas import *
tumor_df = read_csv("http://www.stat.columbia.edu/~gelman/book/data/rats.asc",
                    index_col=False, skiprows=2, delim_whitespace=True)

# <codecell>

# the data roughly looks like follows
tumor_df.head(20)

# <markdowncell>

# The marginal posterior distribution for hyperparameters ($\alpha$ and $\beta$) can be derived as the ratio of prior and posterior partition functions, multipled by the hyperparameter prior: (equation 5.8)
# $$
# p(\alpha,\beta \mid y) = p(\alpha,\beta) \prod_{j=1}^J 
#     \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
#     \frac{\Gamma(\alpha+y_j) \Gamma(\beta+n_j-y_j)}{\Gamma(\alpha+\beta+n_j)}
# $$
# Now let us plot the marginal posterior density.  As instructed in the book, here we reparametrize hyperparameters to be $\log(\frac \alpha \beta)$ and $\log(\alpha + \beta)$.  As always, when computing a density function it is safer to do it in log-scale.

# <codecell>

from scipy.special import gammaln
import numpy as np
# calculate the log marginal hyperparamter posterior for each natural parameter value
def log_hyperparam_posterior_helper(num_trials, num_successes, natural_1, natural_2):
    '''
    log of marginal posterior of hyperparameters (eq 5.8) in natural parameterization
    ( log(alpha/beta) and log(alpha + beta) )
    '''
    assert(len(num_trials) == len(num_successes))
    # calculate original parameters
    beta = np.exp(natural_2) / (1.0 + np.exp(natural_1))
    alpha = beta * np.exp(natural_1)
    # now calculate the log density
    log_prior = np.log(alpha) + np.log(beta) - 2.5 * np.log(alpha + beta)
    # prior partition function part
    log_posterior_1 = len(num_trials) * (gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta))
    # posterior partition function part
    log_posterior_2 = sum(gammaln(alpha + num_successes) + gammaln(beta + num_trials - num_successes) - 
            gammaln(alpha + beta + num_trials))
    return log_prior + log_posterior_1 + log_posterior_2
# vectorize the function
local_log_hyperparam_posterior = \
    np.vectorize(log_hyperparam_posterior_helper, otypes=[np.float], excluded=[0,1])

# <markdowncell>

# Now we calculate the marginal posterior density for hyperparameters on a grid of points and plot it.

# <codecell>

import pylab as pl
import numpy as np
grid_num = 64
xmin = -2.3
xmax = -1.3
ymin = 1
ymax = 5
x = np.linspace(xmin, xmax, grid_num)
y = np.linspace(ymin, ymax, grid_num)
X, Y = np.meshgrid(x, y)
log_Z = local_log_hyperparam_posterior(tumor_df['N'], tumor_df['y'], X,Y)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z) * np.exp(max_log_Z)
levels=np.exp(max_log_Z) * np.linspace(0.05, 0.95, 10)
pl.axes().set_aspect((xmax-xmin)/(ymax-ymin))
pl.contourf(X, Y, Z, 8, alpha=.75, cmap='jet', 
            levels=np.exp(max_log_Z) * np.concatenate(([0], np.linspace(0.05, 0.95, 10), [1])))
pl.contour(X, Y, Z, 8, colors='black', linewidth=.5, levels=levels)
pl.axes().set_xlabel(r'$\log(\alpha/\beta)$')
pl.axes().set_ylabel(r'$\log(\alpha + \beta)$')

# <markdowncell>

# Above is the plot same to Figure 5.3 (a); Figure 5.2 can also be reproduced just by changing the range.  Now let us sample hyperparameters from the posterior distribution to generate Figure 5.3 (b), by approximating the posterior distribution as step functions on grid points.

# <codecell>

# seed the RNG
np.random.seed(13531)
# normalize the grid
probs = np.ravel(Z)/np.sum(Z)
# sample from the normalized distribution
throws = np.random.choice(len(probs), 1000, p=probs)
xs = x[throws % grid_num]
ys = y[throws / grid_num]
xgrid_size = (xmax-xmin)/(grid_num-1)
ygrid_size = (ymax-ymin)/(grid_num-1)
# add jittering
xs += np.random.random(len(xs)) * xgrid_size - 0.5 * xgrid_size
ys += np.random.random(len(ys)) * ygrid_size - 0.5 * ygrid_size
pl.scatter(xs, ys, marker='.', s=1)
pl.xlim(xmin, xmax)
pl.ylim(ymin, ymax)
pl.axes().set_aspect((xmax-xmin)/(ymax-ymin))
pl.axes().set_xlabel(r'$\log(\alpha/\beta)$')
pl.axes().set_ylabel(r'$\log(\alpha + \beta)$')
pl.show()

# <markdowncell>

# Now, let us sample corresponding $\theta_j$'s.

# <codecell>

# convert sampled natural parameters into original parameters
betas = np.exp(ys) / (1.0 + np.exp(xs))
alphas = betas * np.exp(xs)
# assign an array to store sampled parameters
thetas = np.zeros((len(tumor_df), len(alphas)))
# do the sampling
for i in range(len(alphas)):
    for j in range(len(tumor_df)):
        thetas[j][i] = np.random.beta(alphas[i] + tumor_df['y'][j], 
                                      betas[i] + tumor_df['N'][j] - tumor_df['y'][j])
# calculate statistics of parameters
theta_means = np.mean(thetas, axis=1)
theta_2_5s = np.percentile(thetas, 2.5, axis=1)
theta_97_5s = np.percentile(thetas, 97.5, axis=1)

# <markdowncell>

# Now let us draw Figure 5.4, which shows the 95% posterior intervals for each $\theta_j$.

# <codecell>

mles = tumor_df['y']/tumor_df['N']
# to reduce overlap, jitter
jittered_mles = mles + np.random.uniform(low=-0.01, high=0.01, size=len(mles))
ax = pl.scatter(jittered_mles, theta_means)
for j in range(len(tumor_df)):
    pl.plot([jittered_mles[j],jittered_mles[j]], [theta_2_5s[j], theta_97_5s[j]], color='k', linestyle='-', linewidth=1)
pl.plot([0,0.5], [0,0.5], color='k', linestyle='-', linewidth=1)
pl.xlim(-0.01,0.41)
pl.ylim(-0.01,0.41)
pl.axes().set_aspect('equal')
pl.axes().set_xlabel(r'observed rate, $y_j/N_j$')
pl.axes().set_ylabel(r'95% posterior interval for $\theta_j$')
pl.show()

