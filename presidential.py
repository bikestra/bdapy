# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Forecasting Presidential Elections (BDA 15.2) #
# 
# In this notebook, we learn about the advantage of using hierarchical linear model as opposed to non-hierarchical model, by building a forecasting model for presidential elections.  

# <codecell>

% pylab inline
from pandas import *

np.random.seed(13571)

# define sampler for inverse chi-squared distribution
def r_inv_chisquare(size, df, scale_sq):
    return df * scale_sq / np.random.chisquare(df, size)

# <markdowncell>

# Again, the data is from BDA book's homepage.  Somehow the below command does not work if engine="c", which is a default value.  Very frustrating!

# <codecell>

# I could not figure out how to do this in "c" engine.  how frustrating!
presidental_df = read_csv("http://www.stat.columbia.edu/~gelman/book/data/presidential.asc",
                          skiprows=28, delimiter=r"\s+", engine="python")

# <codecell>

presidental_df.head()

# <markdowncell>

# The data looks like above.  For detailed description of each column, please look at skipped rows at the top of the file and the corresponding book chapter in BDA book; here we just consider ``constant``, ``n1`` to ``r6`` columns as given covariates.
# 
# Here, we want to reproduce Figure 15.1, but unfortunately we cannot infer mapping from state indices to actual state names, so I am just showing points here.  The below plot shows how the share of democratic vote in 1984 is related to that in 1988; almost linear relationship is observed.

# <codecell>

df_1984 = presidental_df.loc[(presidental_df['year'] == 1984),('Dvote','state')]
df_1984.rename(columns={'Dvote':'Dvote1984'}, inplace=True)
df_1988 = presidental_df.loc[(presidental_df['year'] == 1988),('Dvote','state')]
df_1988.rename(columns={'Dvote':'Dvote1988'}, inplace=True)
df_1984_1988 = merge(df_1984, df_1988, on=('state'))
fig, ax = plt.subplots()
df_1984_1988.plot(x='Dvote1984',y='Dvote1988', kind='scatter', ax=ax)
ax.set_xlabel('Dem vote by state (1984)')
ax.set_ylabel('Dem vote by state (1988)')
ax.set_aspect(1)

# <markdowncell>

# Below, the same analysis done for 1972 & 1976 year pair.  Here it is shown that the pattern is not very linear, but it is also pointed out in the text that outlier states are in Jimmy Carter's home region, hinting us that we may have to take home regions as features of the model.

# <codecell>

df_1972 = presidental_df.loc[(presidental_df['year'] == 1972),('Dvote','state')]
df_1972.rename(columns={'Dvote':'Dvote1972'}, inplace=True)
df_1976 = presidental_df.loc[(presidental_df['year'] == 1976),('Dvote','state')]
df_1976.rename(columns={'Dvote':'Dvote1976'}, inplace=True)
df_1972_1976 = merge(df_1972, df_1976, on=('state'))
fig, ax = plt.subplots()
df_1972_1976.plot(x='Dvote1972',y='Dvote1976', kind='scatter', ax=ax)
ax.set_xlabel('Dem vote by state (1972)')
ax.set_ylabel('Dem vote by state (1976)')
ax.set_aspect(1)

# <markdowncell>

# Now, let us do the actual regression analysis to form a model.  We confine ourselves to years before 1992, and drop every row that contains missing values:

# <codecell>

df_before1988 = presidental_df[presidental_df['year'] <= 1988].dropna(axis=0)

# <markdowncell>

# I generally do not like one-letter variables, but when doing regression analysis it is very tempting...!

# <codecell>

# let us predict percentages instead of fractions
X = matrix(df_before1988.iloc[:,4:].as_matrix())
y = matrix(df_before1988.loc[:,('Dvote')].as_matrix()).T * 100

# <markdowncell>

# For linear regression, posterior distribution can be simply described by MLE.

# <codecell>

def linear_regression(X,y):
    XtX = X.T * X
    Xty = X.T * y
    Q,R = numpy.linalg.qr(X)
    R_inv = np.linalg.inv(R)
    beta_hat = np.linalg.solve(R, Q.T * y)
    n = X.shape[0]
    k = X.shape[1]
    s_sq = (np.asarray(y - X * beta_hat) ** 2).sum() / (n-k)
    return beta_hat, s_sq

beta_hat, s_sq = linear_regression(X,y)

# <markdowncell>

# Now we will be using national RMSE (root mean squared error) as test statistic for model checking.  Please refer to the book for the precise definition.

# <codecell>

def rmse_nationwide(X, y, beta):
    return sqrt(np.mean(Series(np.squeeze(np.asarray(y - X * beta)), 
                               np.asarray(df_before1988['year'])).groupby(level=0).mean() ** 2))
rmse_nationwide(X, y, beta_hat)

# <markdowncell>

# Now we sample 200 $\beta, \sigma^2$ parameters from the posterior distribution, generate new dependent variable $y^{\text{rep}}$, and compare how the test statistic changes as $y$ is switched to $y^{\text{rep}}$.

# <codecell>

sample_num = 200
n = X.shape[0]; k = X.shape[1]
Q,R = numpy.linalg.qr(X)
R_inv = np.linalg.inv(R)

original_rmses = []
replicated_rmses = []

for sample_index in range(sample_num):
    sigma_sq = r_inv_chisquare(1, n-k, s_sq)[0]
    sampled_beta = beta_hat + R_inv * np.asmatrix(np.random.normal(size=k)).T * sqrt(sigma_sq)
    original_rmses.append(rmse_nationwide(X, y, sampled_beta))
    sampled_y = X * sampled_beta + sqrt(sigma_sq) * np.asmatrix(np.random.normal(size=n)).T
    replicated_rmses.append(rmse_nationwide(X, sampled_y, sampled_beta))

# <codecell>

plt.scatter(original_rmses, replicated_rmses, marker='.', s=1)
plt.xlim(0,1.5)
plt.ylim(0,1.5)
plt.plot([0,3], [0,3], color='k', linestyle='-', linewidth=1)
plt.axes().set_aspect(1)
plt.xlabel(r'$T(y,\theta)$')
plt.ylabel(r'$T(y^{rep},\theta)$')

# <markdowncell>

# Above plot corresponds to Figure 15.2 in the book.  It can be seen that realized test variable is much smaller than that from replicated data; in other words, we might be underestimating the magnitude of prediction error.  Authors argue that this might be due to correlation of dependent variable within observations in the same year.  This motivates us to move on to hierarchical model.
# 
# Note that the scale of this plot is not the same to that of Figure 15.2.  Please let me know if you find errors in my code!

# <codecell>


