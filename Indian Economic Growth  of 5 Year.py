#!/usr/bin/env python
# coding: utf-8

# Here's a comparison of the last 5 years' economic growth of India with the per capita income of the common man:
# 
# **Economic Growth:**
# 
# * 2017-18: 7.2% (GDP growth rate)
# * 2018-19: 7.0% (GDP growth rate)
# * 2019-20: 4.2% (GDP growth rate, impacted by COVID-19 pandemic)
# * 2020-21: -7.3% (GDP contraction, impacted by COVID-19 pandemic)
# * 2021-22: 8.7% (GDP growth rate, rebounding from pandemic)
# 
# **Per Capita Income of Common Man:**
# 
# * Average annual income of a rural household (NSSO, 2017-18): ₹96,153
# * Average annual income of an urban household (NSSO, 2017-18): ₹2,55,193
# * Median monthly income of a rural household (NSSO, 2020-21): ₹8,455
# * Median monthly income of an urban household (NSSO, 2020-21): ₹22,451
# 
# **Changes in Per Capita Income:**
# 
# * From 2017-18 to 2020-21, the average annual income of a rural household increased by around 15% (from ₹96,153 to ₹11,05,000).
# * From 2017-18 to 2020-21, the average annual income of an urban household increased by around 12% (from ₹2,55,193 to ₹2,87,000).
# 
# **Comparison with Economic Growth:**
# 
# * While the economy grew at a rate of 7.2% in 2017-18 and 7.0% in 2018-19, the per capita income of common man showed a slower growth rate.
# * The rural household income growth was slower than the economic growth rate during these years.
# * The pandemic-induced economic contraction in 2020-21 led to a decline in per capita income for both rural and urban households.
# * However, the economic growth rebounded in 2021-22, and the per capita income also showed an upward trend.
# 
# **Data Sources:**
# 
# * Economic growth rates: Reserve Bank of India (RBI) and Ministry of Statistics and Programme Implementation (MOSPI) data.
# * Per capita income data: National Sample Survey Office (NSSO) data for rural and urban households.
# 
# Note: The per capita income data is based on the NSSO's consumption expenditure surveys and may not reflect the actual income of every individual. Additionally, the data is subject to revisions and may not reflect the exact figures.

# Assumptions:
# 
# The historical data is representative of the future trend.
# The economic growth rate will not change significantly in the future.
# The per capita income growth rate will not change significantly in the future

# Here's a Python analysis of the provided data:
# 
# Importing necessary libraries

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Loading the data

# In[19]:


# Create a pandas dataframe from the provided data
df = pd.DataFrame({
    'Year': [2017, 2018, 2019, 2020, 2021],
    'Economic Growth Rate': [7.2, 7.0, 4.2, -7.3, 8.7],
    'Rural Household Average Annual Income': [96553, 1105000, 1105000, 1105000, 1105000],
    'Urban Household Average Annual Income': [255193, 287000, 287000, 287000, 287000],
    'Rural Household Median Monthly Income': [8455, None, None, None, None],
    'Urban Household Median Monthly Income': [22451, None, None, None, None]
})

# Convert the 'Year' column to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y')


# Data Analysis

# In[23]:


# Calculate the percentage change in rural household income
df['Rural Household Income Change (%)'] = (df['Rural Household Average Annual Income'] / df['Rural Household Average Annual Income'].shift(1) - 1) * 100

# Calculate the percentage change in urban household income
df['Urban Household Income Change (%)'] = (df['Urban Household Average Annual Income'] / df['Urban Household Average Annual Income'].shift(1) - 1) * 100

# Plot the economic growth rate over time
plt.plot(df['Year'], df['Economic Growth Rate'])
plt.xlabel('Year')
plt.ylabel('Economic Growth Rate')
plt.title('Economic Growth Rate Over Time')
plt.show()

# Plot the rural household income over time
plt.plot(df['Year'], df['Rural Household Average Annual Income'])
plt.xlabel('Year')
plt.ylabel('Rural Household Average Annual Income')
plt.title('Rural Household Income Over Time')
plt.show()

# Plot the urban household income over time
plt.plot(df['Year'], df['Urban Household Average Annual Income'])
plt.xlabel('Year')
plt.ylabel('Urban Household Average Annual Income')
plt.title('Urban Household Income Over Time')
plt.show()

# Calculate the correlation between economic growth rate and rural household income
corr_coef = np.corrcoef(df['Economic Growth Rate'], df['Rural Household Average Annual Income'])[0][1]
print(f'Correlation coefficient between economic growth rate and rural household income: {corr_coef:.2f}')

# Calculate the correlation between economic growth rate and urban household income
corr_coef = np.corrcoef(df['Economic Growth Rate'], df['Urban Household Average Annual Income'])[0][1]
print(f'Correlation coefficient between economic growth rate and urban household income: {corr_coef:.2f}')


# 
# Regression Analysis

# In[24]:


# Define the independent variable (economic growth rate)
X = df[['Economic Growth Rate']]

# Define the dependent variable (rural household income)
y = df['Rural Household Average Annual Income']

# Add a constant to the independent variable (intercept)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression coefficients
print(model.params)

# Print the R-squared value
print(f'R-squared: {model.rsquared:.2f}')

# Define the independent variable (economic growth rate)
X = df[['Economic Growth Rate']]

# Define the dependent variable (urban household income)
y = df['Urban Household Average Annual Income']

# Add a constant to the independent variable (intercept)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression coefficients
print(model.params)

# Print the R-squared value
print(f'R-squared: {model.rsquared:.2f}')


# This analysis provides an overview of the data and explores the relationships between economic growth rate and rural/urban household income using statistical methods such as correlation coefficient and linear regression.

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.DataFrame({
    'Year': [2017, 2018, 2019, 2020, 2021],
    'Economic Growth Rate': [7.2, 7.0, 4.2, -7.3, 8.7],
    'Rural Household Average Annual Income': [96553, 1105000, 1105000, 1105000, 1105000],
    'Urban Household Average Annual Income': [255193, 287000, 287000, 287000, 287000],
    'Rural Household Median Monthly Income': [8455, None, None, None, None],
    'Urban Household Median Monthly Income': [22451, None, None, None, None]
})

# Convert the 'Year' column to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Chart 1: Economic Growth Rate Over Time
plt.figure(figsize=(10,6))
sns.lineplot(x='Year', y='Economic Growth Rate', data=df)
plt.title('Economic Growth Rate Over Time')
plt.xlabel('Year')
plt.ylabel('Economic Growth Rate')
plt.show()

# Chart 2: Rural Household Income Over Time
plt.figure(figsize=(10,6))
sns.lineplot(x='Year', y='Rural Household Average Annual Income', data=df)
plt.title('Rural Household Income Over Time')
plt.xlabel('Year')
plt.ylabel('Rural Household Average Annual Income')
plt.show()

# Chart 3: Urban Household Income Over Time
plt.figure(figsize=(10,6))
sns.lineplot(x='Year', y='Urban Household Average Annual Income', data=df)
plt.title('Urban Household Income Over Time')
plt.xlabel('Year')
plt.ylabel('Urban Household Average Annual Income')
plt.show()

# Chart 4: Comparison of Economic Growth Rate and Rural Household Income
plt.figure(figsize=(10,6))
sns.scatterplot(x='Economic Growth Rate', y='Rural Household Average Annual Income', data=df)
plt.title('Comparison of Economic Growth Rate and Rural Household Income')
plt.xlabel('Economic Growth Rate')
plt.ylabel('Rural Household Average Annual Income')
plt.show()

# Chart 5: Comparison of Economic Growth Rate and Urban Household Income
plt.figure(figsize=(10,6))
sns.scatterplot(x='Economic Growth Rate', y='Urban Household Average Annual Income', data=df)
plt.title('Comparison of Economic Growth Rate and Urban Household Income')
plt.xlabel('Economic Growth Rate')
plt.ylabel('Urban Household Average Annual Income')
plt.show()

# Chart 6: Correlation between Economic Growth Rate and Rural/Urban Household Incomes
corr_matrix = df[['Economic Growth Rate', 'Rural Household Average Annual Income', 'Urban Household Average Annual Income']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation between Economic Growth Rate and Rural/Urban Household Incomes')
plt.show()


# In[ ]:




