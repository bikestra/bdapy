# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Incumbency in Congressional Elections (BDA 14.3) #
# 
# This notebook page reproduces analysis done in BDA 14.3, which investigates the advantage of incumbents in congressional elections.

# <codecell>

% pylab inline
from pandas import *

# <codecell>

# construct a list of DataFrames for each year
dfs = []
# it seems like years prior to 1990 were not analyzed in the book
for year in range(1902, 1994, 2):
    year_df = read_csv("http://www.stat.columbia.edu/~gelman/book/data/incumbency/%d.asc" % year,
             delimiter=r"\s+", header=None)    
    year_df['year'] = year
    dfs.append(year_df)
incumbency_df = concat(dfs)
incumbency_df.rename(columns={0:'state',1:'district',2:'incumbency',3:'democratic',4:'republican'},
                     inplace=True)
# filter out irrelevant cases
# incumbency_df = incumbency_df[(incumbency_df['democratic'] == 0) | (incumbency_df['democratic'] == 0)]

# <codecell>

# compute proportion of democratic votes
incumbency_df['demoprop'] = incumbency_df['democratic'] / \
        (incumbency_df['democratic'] + incumbency_df['republican'])

# <codecell>

plt.scatter(incumbency_df[incumbency_df['year'] == 1986]['demoprop'],
            incumbency_df[incumbency_df['year'] == 1988]['demoprop'])

# <codecell>

df_1986 = incumbency_df.loc[incumbency_df['year'] == 1986,('state','district','demoprop')]
df_1988 = incumbency_df.loc[incumbency_df['year'] == 1988,('state','district','demoprop')]
df_1986 = df_1986.rename(columns={'demoprop':'demoprop1986'})
df_1988 = df_1988.rename(columns={'demoprop':'demoprop1988'})

# <codecell>

df_1986.join(other=df_1988, on=('state','district'))

# <codecell>

df_1986.join(on=['state','district'])

# <codecell>

df_1988.head()

# <codecell>

df_1986.head()

# <codecell>


