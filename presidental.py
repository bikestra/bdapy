# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Forecasting Presidental Elections (BDA 15.2) #
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

# The data looks like above.  For detailed description of each column, please read skipped rows at the top of the file.
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

# <codecell>

df_before1988 = presidental_df[presidental_df['year'] <= 1988].dropna(axis=0)

# <codecell>

X = matrix(df_before1988.iloc[:,4:].as_matrix())
y = matrix(df_before1988.loc[:,('Dvote')].as_matrix()).T

# <codecell>

XtX = X.T * X
Xty = X.T * y
Q,R = numpy.linalg.qr(X)
R_inv = np.linalg.inv(R)
beta_hat = np.linalg.solve(R, Q.T * y)
n = X.shape[0]
k = X.shape[1]
s_sq = (np.asarray(y - X * beta_hat) ** 2).sum() / (n-k)

# <codecell>

def rmse_nationwide(X, beta):
    return sqrt(np.mean(Series(np.squeeze(np.asarray(y - X * beta)), 
                               np.asarray(df_before1988['year'])).groupby(level=0).mean() ** 2))
rmse_nationwide(X, beta_hat)

# <codecell>

sample_num = 200
for sample_index in range(sample_num):
    sigma_sq = r_inv_chisquare(1, n-k, s_sq)[0]
    sampled_beta = beta_hat + R_inv * np.asmatrix(np.random.normal(size=k)).T * sqrt(sigma_sq)
    print rmse_nationwide(X, sampled_beta)

# <codecell>

plt.hist(y - X * beta_hat)

# <codecell>

plt.scatter(X * beta_hat, y - X * beta_hat)

# <codecell>


