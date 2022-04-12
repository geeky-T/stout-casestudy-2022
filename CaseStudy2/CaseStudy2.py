#!/usr/bin/env python
# coding: utf-8

# # Base Setup

# In[100]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None


# In[101]:


df = pd.read_csv('./casestudy.csv')
df = df.dropna()
df.head()


# # Information Extraction

# • Total revenue for the current year
# 

# In[102]:


year_wise_total_revenue = df.groupby('year')['net_revenue'].sum()
print(year_wise_total_revenue)


# • New Customer Revenue e.g. new customers not present in previous year only
# 

# In[103]:


year_2015_data = df.query('year == 2015')
year_2016_data = df.query('year == 2016')
year_2017_data = df.query('year == 2017')

new_customer_2015 = year_2015_data
new_customer_2016 = year_2016_data[~year_2016_data.customer_email.isin(
    year_2015_data.customer_email)]
new_customer_2017 = year_2017_data[~year_2017_data.customer_email.isin(
    year_2016_data.customer_email)]

print(new_customer_2015.net_revenue.sum())
print(new_customer_2016.net_revenue.sum())
print(new_customer_2017.net_revenue.sum())


# • Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
# 

# In[104]:


customer_growth_2015 = 0
customer_growth_2016 = year_2016_data.net_revenue.sum() -     year_2015_data.net_revenue.sum()
customer_growth_2017 = year_2017_data.net_revenue.sum() -     year_2016_data.net_revenue.sum()
print(customer_growth_2015)
print(customer_growth_2016)
print(customer_growth_2017)


# • Revenue lost from attrition
# 

# In[105]:


lost_customer_2015 = 0
lost_customer_2016 = year_2015_data[~year_2015_data.customer_email.isin(
    year_2016_data.customer_email)]
lost_customer_2017 = year_2016_data[~year_2016_data.customer_email.isin(
    year_2017_data.customer_email)]

print(lost_customer_2015)
print(lost_customer_2016.net_revenue.sum())
print(lost_customer_2017.net_revenue.sum())


# • Existing Customer Revenue Current Year
# • Existing Customer Revenue Prior Year

# In[106]:


revenue_2015 = year_2015_data.net_revenue.sum()
revenue_2016 = year_2016_data.net_revenue.sum()
revenue_2017 = year_2017_data.net_revenue.sum()

print(revenue_2015)
print(revenue_2016)
print(revenue_2017)


# • Total Customers Current Year
# • Total Customers Previous Year
# 

# In[107]:


customers_2015 = len(year_2015_data.customer_email.unique())
customers_2016 = len(year_2016_data.customer_email.unique())
customers_2017 = len(year_2017_data.customer_email.unique())

print(customers_2015)
print(customers_2016)
print(customers_2017)


# • New Customers
# 

# In[108]:


print(new_customer_2015.customer_email.unique())
print(new_customer_2016.customer_email.unique())
print(new_customer_2017.customer_email.unique())


# • Lost Customers
# 

# In[109]:


print(lost_customer_2015)
print(lost_customer_2016.customer_email.unique())
print(lost_customer_2017.customer_email.unique())


# # Insight Visualization

# #### Top 50 Existing Customers Purchase Trend
# This chart shows how the top 50 customers who generated revenue for all the three years and their expense per year trend. (Customer Purchase Pattern).

# In[110]:


year_2015_data = df.query('year == 2015')
year_2016_data = df.query('year == 2016')
year_2017_data = df.query('year == 2017')

customers_2015 = year_2015_data
existing_customer_2016 = year_2016_data[year_2016_data.customer_email.isin(
    year_2015_data.customer_email.unique())]
existing_customer_2017 = year_2017_data[year_2017_data.customer_email.isin(
    year_2016_data.customer_email.unique())]
all_time_existing_customer_data = df[df.customer_email.isin(
    existing_customer_2017.customer_email.unique())]

top_50_revenue_generating_all_time_existing_customers = all_time_existing_customer_data.groupby(
    ['customer_email'])['net_revenue'].sum().reset_index(name='net_revenue').nlargest(50, columns=['net_revenue'])

top_50_all_time_existing_customer_data = pd.DataFrame(all_time_existing_customer_data[all_time_existing_customer_data.customer_email.isin(
    top_50_revenue_generating_all_time_existing_customers.customer_email)])


fig, axes = plt.subplots(5, 1, figsize=(20, 40))
plt.tight_layout(pad=3)
fig.suptitle('Top 50 Existing Customers Purchase Trend')

data = top_50_all_time_existing_customer_data[top_50_all_time_existing_customer_data.customer_email.isin(
    top_50_all_time_existing_customer_data.customer_email.unique()[0:10])]
sns.pointplot(ax=axes[0], hue='customer_email',
              y='net_revenue', data=data, palette='rainbow', x='year')
axes[0].set_xlabel('Customer 1-10')
axes[0].set_ylabel('Revenue')

data = top_50_all_time_existing_customer_data[top_50_all_time_existing_customer_data.customer_email.isin(
    top_50_all_time_existing_customer_data.customer_email.unique()[10:20])]
sns.pointplot(ax=axes[1], hue='customer_email',
              y='net_revenue', data=data, palette='rainbow', x='year')
axes[1].set_xlabel('Customer 11-20')
axes[1].set_ylabel('Revenue')

data = top_50_all_time_existing_customer_data[top_50_all_time_existing_customer_data.customer_email.isin(
    top_50_all_time_existing_customer_data.customer_email.unique()[20:30])]
sns.pointplot(ax=axes[2], hue='customer_email',
              y='net_revenue', data=data, palette='rainbow', x='year')
axes[2].set_xlabel('Customer 21-30')
axes[2].set_ylabel('Revenue')

data = top_50_all_time_existing_customer_data[top_50_all_time_existing_customer_data.customer_email.isin(
    top_50_all_time_existing_customer_data.customer_email.unique()[30:40])]
sns.pointplot(ax=axes[3], hue='customer_email',
              y='net_revenue', data=data, palette='rainbow', x='year')
axes[3].set_xlabel('Customer 31-40')
axes[3].set_ylabel('Revenue')

data = top_50_all_time_existing_customer_data[top_50_all_time_existing_customer_data.customer_email.isin(
    top_50_all_time_existing_customer_data.customer_email.unique()[40:50])]
sns.pointplot(ax=axes[4], hue='customer_email',
              y='net_revenue', data=data, palette='rainbow', x='year')
axes[4].set_xlabel('Customer 41-50')
axes[4].set_ylabel('Revenue')


# #### Existing vs New Customers Revenue Analysis
# Revenue Generation Analysis over existing customers versus new customers every year.

# In[111]:


year_2015_data = df.query('year == 2015')
year_2016_data = df.query('year == 2016')
year_2017_data = df.query('year == 2017')

new_customer_2016 = year_2016_data[~year_2016_data.customer_email.isin(
    year_2015_data.customer_email)]
new_customer_2016.insert(1, 'type', 'new')
existing_customer_2016 = year_2016_data[year_2016_data.customer_email.isin(
    year_2015_data.customer_email.unique())]
existing_customer_2016.insert(1, 'type', 'existing')

new_customer_2017 = year_2017_data[~year_2017_data.customer_email.isin(
    year_2016_data.customer_email)]
new_customer_2017.insert(1, 'type', 'new')
existing_customer_2017 = year_2017_data[year_2017_data.customer_email.isin(
    year_2016_data.customer_email.unique())]
existing_customer_2017.insert(1, 'type', 'existing')

data_2016 = pd.concat(
    [new_customer_2016, existing_customer_2016], ignore_index=True)
data_2017 = pd.concat(
    [new_customer_2017, existing_customer_2017], ignore_index=True)

data = pd.concat([data_2016, data_2017], ignore_index=True)

data = data.groupby(['year', 'type'])[
    'net_revenue'].sum().reset_index(name='net_revenue')


plt.figure(figsize=(5, 9))
plt.tight_layout(pad=3)
plt.suptitle('Existing Vs New Customer Revenue Trend')
sns.pointplot(hue='year',y='net_revenue',data=data, palette='rainbow', x='type')
plt.xlabel('Customer Type')
plt.ylabel('Revenue Generated')


# #### Top 50 Customers Who left in 2017 and degraded between year 2015-16
# This chart shows how customers who degraded on purchase in between 2016 and 2015 and completely stopped in the year 2017.\
# (helps us ask for feedback from the customers or possible problems they may have faced)

# In[112]:


year_2015_data = df.query('year == 2015')
year_2016_data = df.query('year == 2016')
year_2017_data = df.query('year == 2017')

customers_2015 = year_2015_data
existing_customer_2016 = year_2016_data[year_2016_data.customer_email.isin(
    year_2015_data.customer_email.unique())]
customers_left_2017 = existing_customer_2016[~existing_customer_2016.customer_email.isin(
    year_2017_data.customer_email.unique())]


data = df[df.customer_email.isin(
    customers_left_2017.customer_email.unique())]

data_2015 = data.query('year == 2015')
data_2015.rename(columns={'net_revenue': 'net_revenue_2015'}, inplace=True)

data_2016 = data.query('year == 2016')
data_2016.rename(columns={'net_revenue': 'net_revenue_2016'}, inplace=True)


data_merged = pd.merge(data_2015, data_2016, how="left", on=["customer_email"])
degrading_customers = data_merged[data_merged.net_revenue_2015 >
                                  data_merged.net_revenue_2016]

data = df[df.customer_email.isin(
    degrading_customers.customer_email.unique())]

top_50_revenue_generating_among_degrading_customers = data[data.customer_email.isin(data.groupby(
    ['customer_email'])['net_revenue'].sum().reset_index(name='net_revenue').nlargest(50, columns=['net_revenue']).customer_email.unique())]


fig, axes = plt.subplots(5, 1, figsize=(20, 40))
plt.tight_layout(pad=3)
fig.suptitle('Top 50 Customers Who left in 2017 and degraded between year 2015-16')

data = top_50_revenue_generating_among_degrading_customers[top_50_revenue_generating_among_degrading_customers.customer_email.isin(
    top_50_revenue_generating_among_degrading_customers.customer_email.unique()[0:10])]
g = sns.pointplot(ax=axes[0], hue='customer_email',
                  y='net_revenue', data=data, palette='rainbow', x='year', xlim=(2015, 2017))
axes[0].set_xlabel('Customer 1-10')
axes[0].set_ylabel('Revenue')
g.set_xticks(range(3+1))
g.set_xticklabels(['2015', '2016', '2017', ''])

data = top_50_revenue_generating_among_degrading_customers[top_50_revenue_generating_among_degrading_customers.customer_email.isin(
    top_50_revenue_generating_among_degrading_customers.customer_email.unique()[10:20])]
g = sns.pointplot(ax=axes[1], hue='customer_email',
                  y='net_revenue', data=data, palette='rainbow', x='year')
axes[1].set_xlabel('Customer 11-20')
axes[1].set_ylabel('Revenue')
g.set_xticks(range(3+1))
g.set_xticklabels(['2015', '2016', '2017', ''])

data = top_50_revenue_generating_among_degrading_customers[top_50_revenue_generating_among_degrading_customers.customer_email.isin(
    top_50_revenue_generating_among_degrading_customers.customer_email.unique()[20:30])]
g = sns.pointplot(ax=axes[2], hue='customer_email',
                  y='net_revenue', data=data, palette='rainbow', x='year')
axes[2].set_xlabel('Customer 21-30')
axes[2].set_ylabel('Revenue')
g.set_xticks(range(3+1))
g.set_xticklabels(['2015', '2016', '2017', ''])

data = top_50_revenue_generating_among_degrading_customers[top_50_revenue_generating_among_degrading_customers.customer_email.isin(
    top_50_revenue_generating_among_degrading_customers.customer_email.unique()[30:40])]
g = sns.pointplot(ax=axes[3], hue='customer_email',
                  y='net_revenue', data=data, palette='rainbow', x='year')
axes[3].set_xlabel('Customer 31-40')
axes[3].set_ylabel('Revenue')
g.set_xticks(range(3+1))
g.set_xticklabels(['2015', '2016', '2017', ''])

data = top_50_revenue_generating_among_degrading_customers[top_50_revenue_generating_among_degrading_customers.customer_email.isin(
    top_50_revenue_generating_among_degrading_customers.customer_email.unique()[40:50])]
g = sns.pointplot(ax=axes[4], hue='customer_email',
                  y='net_revenue', data=data, palette='rainbow', x='year')
axes[4].set_xlabel('Customer 41-50')
axes[4].set_ylabel('Revenue')
g.set_xticks(range(3+1))
g.set_xticklabels(['2015', '2016', '2017', ''])

