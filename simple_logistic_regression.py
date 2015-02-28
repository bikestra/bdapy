# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Simple Logistic Regression #
# 
# Here, we consider simple Bayesian logistic regression model which has only one covariate variable.  Let $x_1, x_2, \ldots, x_n$ be the covariate variable, $n_i$ be the number of trials in observation $i$, and $y_i$ be the number of "success".  The likelihood function can be written:
# $$
# p(y_i \mid \theta_1, \theta_2, n_i, x_i) \propto 
# [\text{logit}^{-1}(\theta_1 + \theta_2 x_i)]^{y_i}
# [1 - \text{logit}^{-1}(\theta_1 + \theta_2 x_i)]^{n_i - y_i}
# $$
# For simplicity, uniform prior is assumed on $\theta_1, \theta_2$.

# <markdowncell>

# ## Sampling from Grid (BDA 3.7) ##
# 
# Since parameters are only two-dimensional, we can afford to approximate the posterior distribution on grid points.

# <codecell>

% pylab inline
import numpy as np

# <markdowncell>

# First of all, here is the data we will use, from Racine et al. (1986).

# <codecell>

# Data from Racine et al. (1986) 
from pandas import *
bioassay_df = DataFrame({'dose':[-0.86,-0.30,-0.05,0.73],
                         'animals':[5,5,5,5],
                         'deaths':[0,1,3,5]
                         })

# <markdowncell>

# To produce Figure 3.3 (a), we define a function that evaluates log-posterior function.

# <codecell>

def log_posterior_helper(theta_1, theta_2, trials, successes, covariates, log_prior = lambda t1, t2: 0):
    ret = log_prior(theta_1, theta_2)
    # first compute prediction scores
    scores = theta_1 + theta_2 * covariates
    # then convert it to log-likelihood
    ret += np.sum(-np.log(1 + exp(-scores)) * successes - np.log(1 + exp(scores)) * (trials - successes ))
    return ret
log_posterior = \
    np.vectorize(log_posterior_helper, otypes=[np.float], excluded=[2,3,4,5])

# <codecell>

import pylab as pl
grid_num = 64
theta1_min = -4
theta1_max = 10
theta2_min = -10
theta2_max = 40
x = np.linspace(theta1_min, theta1_max, grid_num)
y = np.linspace(theta2_min, theta2_max, grid_num)
X, Y = np.meshgrid(x, y)
log_Z = log_posterior(X,Y,bioassay_df.animals, bioassay_df.deaths, bioassay_df.dose)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z) * np.exp(max_log_Z)
levels=np.exp(max_log_Z) * np.linspace(0.05, 0.95, 10)
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.contourf(X, Y, Z, 8, alpha=.75, cmap='jet', 
            levels=np.exp(max_log_Z) * np.concatenate(([0], np.linspace(0.05, 0.95, 10), [1])))
pl.contour(X, Y, Z, 8, colors='black', linewidth=.5, levels=levels)
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')

# <markdowncell>

# Above is Figure 3.3 (a).  Since the grid is given, sampling from it is trivial.

# <codecell>

# seed the RNG
np.random.seed(135791)
# normalize the grid
probs = np.ravel(Z)/np.sum(Z)
# sample from the normalized distribution
throws = np.random.choice(len(probs), 1000, p=probs)
theta1s = x[throws % grid_num]
theta2s = y[throws / grid_num]
xgrid_size = float(theta1_max-theta1_min)/(grid_num-1)
ygrid_size = float(theta2_max-theta2_min)/(grid_num-1)
# add jittering
theta1s += np.random.random(len(theta1s)) * xgrid_size - 0.5 * xgrid_size
theta2s += np.random.random(len(theta2s)) * ygrid_size - 0.5 * ygrid_size
pl.scatter(theta1s, theta2s, marker='.', s=1)
pl.xlim(theta1_min, theta1_max)
pl.ylim(theta2_min, theta2_max)
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')
pl.show()

# <markdowncell>

# It is trivial to convert these parameters to LD50 to get Figure 3.4.

# <codecell>

pl.hist(-theta1s/theta2s, bins=20)
pl.axes().set_xlabel('LD50')
pl.axes().get_yaxis().set_visible(False)

# <markdowncell>

# ## Asymptotic Approximation (BDA 4.1) ## 
# 
# Alternatively, we can find the mode of the distribution, and use normal approximation of posterior at the mode.  For small problems like this, we can even avoid computing the actual gradient/Hessian by using numerical methods.

# <codecell>

import scipy.optimize

# <codecell>

# since I am lazy, let's use gradient-free optimization although analytic formula is available
res = scipy.optimize.minimize(lambda x: -log_posterior(x[0],x[1],bioassay_df.animals, bioassay_df.deaths, bioassay_df.dose), 
                              (0,0))

# <codecell>

theta_mode = res.x

# <codecell>

from scipy import linalg
def log_asymp_posterior_helper(theta_1, theta_2):
    theta_vec = np.matrix([theta_1,theta_2])
    return -0.5 * (theta_vec - theta_mode) * linalg.solve(res.hess_inv,(theta_vec - theta_mode).transpose())

log_asymp_posterior = \
    np.vectorize(log_asymp_posterior_helper, otypes=[np.float])

# <markdowncell>

# Now Figure 4.1 (a) can be reproduced:

# <codecell>

import pylab as pl
grid_num = 64
theta1_min = -4
theta1_max = 10
theta2_min = -10
theta2_max = 40
x = np.linspace(theta1_min, theta1_max, grid_num)
y = np.linspace(theta2_min, theta2_max, grid_num)
X, Y = np.meshgrid(x, y)
log_Z = log_asymp_posterior(X,Y)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z) * np.exp(max_log_Z)
levels=np.exp(max_log_Z) * np.linspace(0.05, 0.95, 10)
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.contourf(X, Y, Z, 8, alpha=.75, cmap='jet', 
            levels=np.exp(max_log_Z) * np.concatenate(([0], np.linspace(0.05, 0.95, 10), [1])))
pl.contour(X, Y, Z, 8, colors='black', linewidth=.5, levels=levels)
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')

# <markdowncell>

# Since log-posterior of Gaussian distribution is only quadratic, the tilted shape of actual posterior distribution is not captured.
# 
# Now, sampling from Gaussian distribution is straightforward:

# <codecell>

theta_samples = np.random.multivariate_normal(theta_mode, res.hess_inv, 1000)
theta1s = theta_samples[:,0]
theta2s = theta_samples[:,1]
pl.scatter(theta1s, theta2s, marker='.', s=1)
pl.xlim(theta1_min, theta1_max)
pl.ylim(theta2_min, theta2_max)
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')
pl.show()

# <markdowncell>

# Histogram of posterior samples for LD50 in Figure 4.2 (a) can also be reproduced.  Gaussian approximation places much more probability mass of $\theta_2$ around zero, which results in much higher variance of LD50.

# <codecell>

pl.hist(-theta1s/theta2s, bins=1000)
pl.axes().set_xlabel('LD50')
pl.xlim(-2,2)
pl.axes().get_yaxis().set_visible(False)

# <markdowncell>

# Below is Figure 4.2 (b), which shows only central 95% of the distribution.

# <codecell>

ld50s = -theta1s/theta2s
pl.hist(ld50s[(ld50s > np.percentile(ld50s, 2.5)) & (ld50s < np.percentile(ld50s, 97.5))], bins=20)
pl.axes().set_xlabel('LD50')
pl.axes().get_yaxis().set_visible(False)

# <markdowncell>

# ## Expectation Propagation (BDA 13.8) ##
# 
# Expectation Propogation (EP) is one of the most successful inference method for Bayesian models, especially for classification problems.  Here, we apply EP on Bayesian logistic regression; note that EP is more frequently used with probit regression instead.

# <codecell>

import scipy.integrate
from scipy.stats import norm

# <codecell>

# first, form data matrix
data_matrix = np.column_stack([np.repeat(1, len(bioassay_df)), bioassay_df.dose])
trials = bioassay_df.animals
successes = bioassay_df.deaths

# <markdowncell>

# Now let us run Expectation Propagation!

# <codecell>

history_mu1s = []
history_mu2s = []
history_sigma1s = []
history_sigma2s = []
history_rhos = []
history_means = []
history_precisions = []

local_param_naturalmeans = [np.matrix(np.repeat(0, data_matrix.shape[1])).T 
                            for index in xrange(len(data_matrix))]
local_param_precisions = [np.identity(data_matrix.shape[1])
                          for index in xrange(len(data_matrix))]
global_param_naturalmean = np.matrix(np.repeat(0.0, data_matrix.shape[1])).T
global_param_precision = np.zeros((data_matrix.shape[1],data_matrix.shape[1]))
M_is = [0 for index in xrange(len(data_matrix))]
V_is = [linalg.norm(data_matrix[index,:]) for index in xrange(len(data_matrix))]
for i in xrange(len(data_matrix)):
    global_param_naturalmean += local_param_naturalmeans[i]
    global_param_precision += local_param_precisions[i]

for iter_num in xrange(10):
    for i in xrange(len(data_matrix)):
        trial = trials[i]
        success = successes[i]
        global_param_mean = np.matrix(solve(global_param_precision, global_param_naturalmean))
        global_param_variance = linalg.inv(global_param_precision)
        
        history_means.append(global_param_mean.copy())
        history_precisions.append(global_param_precision.copy())
        history_mu1s.append(global_param_mean[0,0])
        history_mu2s.append(global_param_mean[1,0])
        history_sigma1s.append(np.sqrt(global_param_variance[0,0]))
        history_sigma2s.append(np.sqrt(global_param_variance[1,1]))
        history_rhos.append(global_param_variance[0,1]/
                            np.sqrt(global_param_variance[0,0] * global_param_variance[1,1]))
        
        M_i = M_is[i]
        V_i = V_is[i]
        M = np.dot(np.matrix(data_matrix[i,:]), global_param_mean)
        V = np.dot(data_matrix[i,:], solve(global_param_precision, data_matrix[i,:]))
        V_del_i = 1.0/(1.0/V - 1.0/V_i)
        M_del_i = V_del_i * (M/V - M_i/V_i)
        scale_del_i = np.sqrt(V_del_i)

        def tilted_eta(eta, trial, success, moment):
            prob = 1.0/(1.0 + exp(-eta))
            return (eta ** moment) * norm.pdf(eta, loc=M_del_i, scale=scale_del_i) \
                * (prob ** success) * ((1.0-prob) ** (trial - success))
        moment_0 = scipy.integrate.quad(lambda x: tilted_eta(x, trial, success, 0), 
                                        M_del_i - 10 * scale_del_i, M_del_i + 10 * scale_del_i)[0]
        moment_1 = scipy.integrate.quad(lambda x: tilted_eta(x, trial, success, 1), 
                                        M_del_i - 10 * scale_del_i, M_del_i + 10 * scale_del_i)[0]
        moment_2 = scipy.integrate.quad(lambda x: tilted_eta(x, trial, success, 2), 
                                        M_del_i - 10 * scale_del_i, M_del_i + 10 * scale_del_i)[0]
        M = moment_1/moment_0
        V = (moment_2/moment_0) - ((moment_1/moment_0) ** 2)
        one_over_V_i = (1.0/V - 1.0/V_del_i)
        M_i_div_V_i = M/V - M_del_i/V_del_i
        V_i = 1.0/one_over_V_i
        M_i = M_i_div_V_i * V_i
        M_is[i] = M_i
        V_is[i] = V_i

        # subtract original message
        global_param_precision -= local_param_precisions[i]
        global_param_naturalmean -= local_param_naturalmeans[i]

        # compute new message
        local_param_precisions[i] = np.matrix(data_matrix[i,:]).T * data_matrix[i,:] * one_over_V_i
        local_param_naturalmeans[i] = np.matrix(M_i_div_V_i * data_matrix[i,:]).T

        # send new message
        global_param_precision += local_param_precisions[i]
        global_param_naturalmean += local_param_naturalmeans[i]

# <markdowncell>

# Now series of plots in Figure 13.7 can be obtained.

# <codecell>

plt.plot(history_mu1s)
plt.title(r'Updating of $\mu_1$')

# <codecell>

plt.plot(history_mu2s)
plt.title(r'Updating of $\mu_2$')

# <codecell>

plt.plot(history_sigma1s)
plt.title(r'Updating of $\sigma_1$')

# <codecell>

plt.plot(history_sigma2s)
plt.title(r'Updating of $\sigma_2$')

# <codecell>

plt.plot(history_rhos)
plt.title(r'Updating of $\rho$')

# <codecell>

global_param_mean = np.matrix(solve(global_param_precision, global_param_naturalmean))

# <markdowncell>

# First, let us draw the contour plot of approximated density we have been drawing.

# <codecell>

from scipy import linalg
def log_ep_posterior_helper(theta_1, theta_2):
    theta_vec = np.matrix([theta_1,theta_2])
    return -(theta_vec - global_param_mean.T) * global_param_precision * (theta_vec - global_param_mean.T).T

log_ep_posterior = \
    np.vectorize(log_ep_posterior_helper, otypes=[np.float])
import pylab as pl
grid_num = 64
theta1_min = -4
theta1_max = 10
theta2_min = -10
theta2_max = 40
x = np.linspace(theta1_min, theta1_max, grid_num)
y = np.linspace(theta2_min, theta2_max, grid_num)
X, Y = np.meshgrid(x, y)
log_Z = log_ep_posterior(X,Y)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z) * np.exp(max_log_Z)
levels=np.exp(max_log_Z) * np.linspace(0.05, 0.95, 10)
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.contourf(X, Y, Z, 8, alpha=.75, cmap='jet', 
            levels=np.exp(max_log_Z) * np.concatenate(([0], np.linspace(0.05, 0.95, 10), [1])))
pl.contour(X, Y, Z, 8, colors='black', linewidth=.5, levels=levels)
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')

# <markdowncell>

# Let us draw progress of the algorithm with 95% confidence region at each iteration, as in Figure 13.8 (a).

# <codecell>

import pylab as pl
grid_num = 64
theta1_min = -2
theta1_max = 4
theta2_min = -5
theta2_max = 25
x = np.linspace(theta1_min, theta1_max, grid_num)
y = np.linspace(theta2_min, theta2_max, grid_num)
X, Y = np.meshgrid(x, y)
interval = 4
for i in range(len(history_means)/interval):
    def log_ep_posterior_helper(theta_1, theta_2):
        theta_vec = np.matrix([theta_1,theta_2])
        return -0.5 * (theta_vec - history_means[i * interval].T) * history_precisions[i * interval] *\
            (theta_vec - history_means[i * interval].T).T

    log_ep_posterior = \
        np.vectorize(log_ep_posterior_helper, otypes=[np.float])
    log_Z = log_ep_posterior(X,Y)
    max_log_Z = log_Z.max()
    Z = np.exp(log_Z - max_log_Z)
    pl.contour(X, Y, Z, 8, colors='black', linewidth=.5, levels=[0.05])
    
pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')

# <markdowncell>

# Now compare 95% confidence region of different algorithms, as in Figure 13.8 (b).

# <codecell>

import pylab as pl
grid_num = 64
theta1_min = -2
theta1_max = 5
theta2_min = -5
theta2_max = 30
x = np.linspace(theta1_min, theta1_max, grid_num)
y = np.linspace(theta2_min, theta2_max, grid_num)
X, Y = np.meshgrid(x, y)

# exact posterior
log_Z = log_posterior(X,Y,bioassay_df.animals, bioassay_df.deaths, bioassay_df.dose)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z)
CS1 = pl.contour(X, Y, Z, colors='black', linestyles='dashed', linewidth=.5, levels=[0.05])

# EP 
def log_ep_posterior_helper(theta_1, theta_2):
    theta_vec = np.matrix([theta_1,theta_2])
    return -0.5 * (theta_vec - global_param_mean.T) * global_param_precision *\
            (theta_vec - global_param_mean.T).T
log_ep_posterior = \
    np.vectorize(log_ep_posterior_helper, otypes=[np.float])
log_Z = log_ep_posterior(X,Y)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z)
CS2 = pl.contour(X, Y, Z, colors='black', linewidth=.5, levels=[0.05])

# Asymptotic normal approximation
log_Z = log_asymp_posterior(X,Y)
max_log_Z = log_Z.max()
Z = np.exp(log_Z - max_log_Z) * np.exp(max_log_Z)
CS3 = pl.contour(X, Y, Z, colors='black', linestyles='dotted', linewidth=.5, levels=[0.05])

lines = [ CS1.collections[0], CS2.collections[0], CS3.collections[-1]]
labels = ['Exact posterior','Expectation Propagation','Mode-based approximation']
plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc=2)

pl.axes().set_aspect(float(theta1_max-theta1_min)/(theta2_max-theta2_min))
pl.axes().set_xlabel(r'$\theta_1$')
pl.axes().set_ylabel(r'$\theta_2$')

# <markdowncell>

# It can be seen that EP provides better approximation of exact posterior than mode-based approximation.

