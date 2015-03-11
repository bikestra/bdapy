# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Incumbency in Congressional Elections (BDA 14.3) #
# 
# This notebook page reproduces analysis done in BDA 14.3, which investigates the advantage of incumbents in congressional elections.

# <codecell>

% pylab inline
from pandas import *

# <markdowncell>

# Here, we read congressional election results from 1902 to 1992 from BDA book website.

# <codecell>

# construct a list of DataFrames for each year
dfs = []
# it seems like years prior to 1990 were not analyzed in the book
for year in range(1900, 1994, 2):
    year_df = read_csv("http://www.stat.columbia.edu/~gelman/book/data/incumbency/%d.asc" % year,
             delimiter=r"\s+", header=None)    
    year_df['year'] = year
    dfs.append(year_df)
incumbency_df = concat(dfs)
incumbency_df.rename(columns={0:'state',1:'district',2:'incumbency',3:'democratic',4:'republican'},
                     inplace=True)
# compute proportion of democratic votes
incumbency_df['demoprop'] = incumbency_df['democratic'] / \
        (incumbency_df['democratic'] + incumbency_df['republican'])

# <markdowncell>

# Preprocessed data look like belows.  `state` and `district` columns identify the district, while `democratic` and `republican` are the number of votes each party received.  `incumbency` is +1 if Democrats controled the seat and -1 if Republicans controled the seat before the election.

# <codecell>

incumbency_df.head()

# <markdowncell>

# For illustration, let us look at how the election result in 1986 is correlated with that in 1988.  For this, we create a separate DataFrame for each year.

# <codecell>

df_1986 = incumbency_df.loc[(incumbency_df['year'] == 1986),('state','district','democratic','republican','demoprop')]
df_1988 = incumbency_df.loc[incumbency_df['year'] == 1988,('state','district','democratic','republican','demoprop','incumbency')]
df_1986 = df_1986.rename(columns={'demoprop':'demoprop1986','democratic':'democratic1986','republican':'republican1986'})
df_1988 = df_1988.rename(columns={'demoprop':'demoprop1988','incumbency':'incumbency1988','democratic':'democratic1988','republican':'republican1988'})

# <markdowncell>

# Then, we merge them by district.

# <codecell>

df_1986_1988 = merge(df_1986, df_1988, on=('state', 'district'))

# <markdowncell>

# We are only interested in cases Democratic and Republican candidates competed with each other.

# <codecell>

# filter out cases 
filtered_df_1986_1988 = df_1986_1988.loc[(df_1986_1988['democratic1986'] > 0) & (df_1986_1988['republican1986'] > 0) &\
        (df_1986_1988['democratic1988'] > 0) & (df_1986_1988['republican1988'] > 0)]

# <codecell>

filtered_df_1986_1988.head(20)

# <markdowncell>

# Figure 14.1 which shows the proportion of Democratic vote as a function of that in 1988 is reproduced below.  Two variables seems to have very strong linear relationship, and it seems critical to use previous year's result as a covariate variable.

# <codecell>

fig, ax = plt.subplots()
filtered_df_1986_1988[filtered_df_1986_1988['incumbency1988'] != 0].plot('demoprop1986', 'demoprop1988', kind='scatter', marker='.', ax=ax)
filtered_df_1986_1988[filtered_df_1986_1988['incumbency1988'] == 0].plot('demoprop1986', 'demoprop1988', kind='scatter', marker='o', facecolors='none', ax=ax)
ax.set_xlim(-.05,1.05)
ax.set_ylim(-.05,1.05)
ax.set_xlabel('Democratic vote in 1986')
ax.set_ylabel('Democratic vote in 1988')
ax.set_aspect('equal')
plt.show()

# <markdowncell>

# Now, we learn a linear model for each year's election result to estimate the effect of incumbency.

# <codecell>

# first, come up with the list of years we will learn linear model on
years = unique(incumbency_df['year'])

# <codecell>

medians = []
for index in range(1, len(years)):
    now_year = years[index]
    prev_year = years[index - 1]
    df_prev = incumbency_df.loc[(incumbency_df['year'] == prev_year),('state','district','demoprop','democratic','republican')]
    df_now = incumbency_df.loc[incumbency_df['year'] == now_year,('state','district','demoprop','incumbency','democratic','republican')]
    df_prev = df_prev.rename(columns={'demoprop':'demoprop_prev','democratic':'democratic_prev','republican':'republican_prev'})
    df_now = df_now.rename(columns={'demoprop':'demoprop_now','democratic':'democratic_now','republican':'republican_now'})
    df_now['is_incumbent'] = abs(df_now['incumbency'])
    df_now['constant'] = 1    
    df_prev_now = merge(df_prev, df_now, on=('state', 'district'))
    df_prev_now = df_prev_now.loc[(df_prev_now['democratic_now'] > 0) & (df_prev_now['republican_now'] > 0) & \
                              (df_prev_now['democratic_prev'] > 0) & (df_prev_now['republican_prev'] > 0)]
    df_prev_now['incumbent_party'] = (df_prev_now['demoprop_prev'] > 0.5).map({True:1,False:-1})
    
    df_prev_now['prevprop'] = df_prev_now['demoprop_prev']
    df_prev_now['nowprop'] = df_prev_now['demoprop_now']
    df_prev_now.loc[df_prev_now['demoprop_prev'] < 0.5, 'prevprop'] = 1.0 - df_prev_now['demoprop_prev'][df_prev_now['demoprop_prev'] < 0.5]
    df_prev_now.loc[df_prev_now['demoprop_prev'] < 0.5, 'nowprop'] = 1.0 - df_prev_now['demoprop_now'][df_prev_now['demoprop_prev'] < 0.5]
    
    X = matrix(df_prev_now.loc[:,('constant', 'is_incumbent', 'incumbent_party','prevprop')].as_matrix())
    y = matrix(df_prev_now.loc[:,('nowprop')].as_matrix()).T
    XtX = X.T * X
    Xty = X.T * y
    beta_hat = linalg.solve(XtX, Xty)
    medians.append(beta_hat[1,0])
    if now_year == 1988:
        print beta_hat
        print medians
        #break

# <codecell>

plt.scatter(years[1:], medians)

# <codecell>

df_prev_now['residual'] = y - X * beta_hat
fig, ax = plt.subplots()
df_prev_now[df_prev_now['is_incumbent'] == 1].plot('prevprop', 'residual', kind='scatter', marker='.', ax=ax)
df_prev_now[df_prev_now['is_incumbent'] == 0].plot('prevprop', 'residual', kind='scatter', marker='o', facecolors='none', ax=ax)
#filtered_df_1986_1988[filtered_df_1986_1988['incumbency1988'] == 0].plot('demoprop1986', 'demoprop1988', kind='scatter', marker='o', facecolors='none', ax=ax)
#ax.set_xlim(-.05,1.05)
#ax.set_ylim(-.05,1.05)
#ax.set_xlabel('Democratic vote in 1986')
#ax.set_ylabel('Democratic vote in 1988')
#ax.set_aspect('equal')
plt.show()

# <codecell>

df_prev_now['prevprop']

# <codecell>

df_prev_now['residual'] = y - X * beta_hat

# <codecell>

df_prev_now.head()

# <codecell>

import statsmodels.api as sm

# <codecell>

est = sm.OLS(y, X)

# <codecell>

est = est.fit()
est.summary()

# <codecell>

prev_year

# <codecell>

df_prev_now['prevprop'] = df_prev_now['demoprop_prev']
df_prev_now['nowprop'] = df_prev_now['demoprop_now']
df_prev_now['prevprop'][df_prev_now['demoprop_prev'] < 0.5] = 1.0 - df_prev_now['demoprop_prev'][df_prev_now['demoprop_prev'] < 0.5]
df_prev_now['nowprop'][df_prev_now['demoprop_prev'] < 0.5] = 1.0 - df_prev_now['demoprop_now'][df_prev_now['demoprop_prev'] < 0.5]

# <codecell>

df_prev_now

# <codecell>

df_prev_now['demoprop_prev'] > 0.5

# <codecell>

df_prev_now

# <codecell>

prev_year

# <codecell>

beta_hat

# <codecell>

now_year

# <codecell>

sum(incumbency_df['democratic'] < 0)

# <codecell>

sum(incumbency_df['republican'] < 0)

# <codecell>

unique(df_1986_1988['democratic1986'])

# <codecell>


