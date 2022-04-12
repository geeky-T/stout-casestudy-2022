#!/usr/bin/env python
# coding: utf-8

# # Describe Data
# 
# Dataset represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals.
# 
# 
# <b><u>Variable Description</u></b>
# 
# emp_title: Job title. \
# emp_length: Number of years in the job, rounded down. \ If longer than 10 years, then this is represented by the value 10. \
# state: Two-letter state code. \
# home_ownership: The ownership status of the applicant's residence. \
# annual_income: Annual income. \
# verified_income: Type of verification of the applicant's income. \
# debt_to_income: Debt-to-income ratio. \
# annual_income_joint: If this is a joint application, then the annual income of the two parties applying. \
# verification_income_joint: Type of verification of the joint income. \
# debt_to_income_joint: Debt-to-income ratio for the two parties. \
# delinq_2y: Delinquencies on lines of credit in the last 2 years. \
# months_since_last_delinq: Months since the last delinquency. \
# earliest_credit_line: Year of the applicant's earliest line of credit. \
# inquiries_last_12m: Inquiries into the applicant's credit during the last 12 months. \
# total_credit_lines: Total number of credit lines in this applicant's credit history. \
# open_credit_lines: Number of currently open lines of credit. \
# total_credit_limit: Total available credit, e. \g. \ if only credit cards, then the total of all the credit limits. \ This excludes a mortgage. \
# total_credit_utilized: Total credit balance, excluding a mortgage. \
# num_collections_last_12m: Number of collections in the last 12 months. \ This excludes medical collections. \
# num_historical_failed_to_pay: The number of derogatory public records, which roughly means the number of times the applicant failed to pay. \
# months_since_90d_late: Months since the last time the applicant was 90 days late on a payment. \
# current_accounts_delinq: Number of accounts where the applicant is currently delinquent. \
# total_collection_amount_ever: The total amount that the applicant has had against them in collections. \
# current_installment_accounts: Number of installment accounts, which are (roughly) accounts with a fixed payment amount and period. \ A typical example might be a 36-month car loan. \
# accounts_opened_24m: Number of new lines of credit opened in the last 24 months. \
# months_since_last_credit_inquiry: Number of months since the last credit inquiry on this applicant. \
# num_satisfactory_accounts: Number of satisfactory accounts. \
# num_accounts_120d_past_due: Number of current accounts that are 120 days past due. \
# num_accounts_30d_past_due: Number of current accounts that are 30 days past due. \
# num_active_debit_accounts: Number of currently active bank cards. \
# total_debit_limit: Total of all bank card limits. \
# num_total_cc_accounts: Total number of credit card accounts in the applicant's history. \
# num_open_cc_accounts: Total number of currently open credit card accounts. \
# num_cc_carrying_balance: Number of credit cards that are carrying a balance. \
# num_mort_accounts: Number of mortgage accounts. \
# account_never_delinq_percent: Percent of all lines of credit where the applicant was never delinquent. \
# tax_liens: a numeric vector. \
# public_record_bankrupt: Number of bankruptcies listed in the public record for this applicant. \
# loan_purpose: The category for the purpose of the loan. \
# application_type: The type of application: either individual or joint. \
# loan_amount: The amount of the loan the applicant received. \
# term: The number of months of the loan the applicant received. \
# interest_rate: Interest rate of the loan the applicant received. \
# installment: Monthly payment for the loan the applicant received. \
# grade: Grade associated with the loan. \
# sub_grade: Detailed grade associated with the loan. \
# issue_month: Month the loan was issued. \
# loan_status: Status of the loan. \
# initial_listing_status: Initial listing status of the loan. \ (I think this has to do with whether the lender provided the entire loan or if the loan is across multiple lenders). \
# disbursement_method: Dispersement method of the loan. \
# balance: Current balance on the loan. \
# paid_total: Total that has been paid on the loan by the applicant. \
# paid_principal: The difference between the original loan amount and the current balance on the loan. \
# paid_interest: The amount of interest paid so far by the applicant. \
# paid_late_fees: Late fees paid by the applicant. \

# # Project Setup

# In[207]:


import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

data = pd.read_csv('loans_full_schema.csv')
data.head()


# # Exploratory Analysis

# Exploring features and its data types. 

# In[208]:


data.info()


# Finding Null counts of each features in the dataset.

# In[209]:


data.isnull().sum()


# Finding Categorical Values among the features with non-numerical data type explored previously.

# In[210]:


# Considering only non-numeric features.
data[
    [
        "emp_title",
        "state",
        "homeownership",
        "verified_income",
        "verification_income_joint",
        "loan_purpose",
        "application_type",
        "grade",
        "sub_grade",
        "issue_month",
        "loan_status",
        "initial_listing_status",
        "disbursement_method", ]
].nunique()


# # Data Visualization

# Status of currently lent resources.

# In[211]:


from matplotlib import pyplot as plt
import seaborn as sns

a = data.groupby(['loan_status']).size().reset_index(name='count').sort_values(ascending=False, by='count')
plt.figure(figsize=(18,10))
g = sns.barplot(x='loan_status', y='count', data=a, palette='rainbow')
g.set_xlabel('Loan Status')
g.set_ylabel('Count')
plt.title("Loan Status by Income Verification")


# Loan Records by their purpose.

# In[212]:



a = data.groupby(['loan_purpose']).size().reset_index(name='count')
fig, ax = plt.subplots(figsize=(16, 16))
ax.pie(a['count'].to_numpy(), labels=a['loan_purpose'].to_numpy(),
       autopct='%.1f%%', textprops={'fontsize': 8}, explode=[0.01]*a['loan_purpose'].nunique())
ax.set_title('Loans by Purpose', fontdict={'fontsize': 24})
plt.tight_layout()


# Loan Interest By Purpose

# In[213]:


plt.figure(figsize=(18,8))
sns.boxplot(x='loan_purpose',y='interest_rate',data=data, palette='rainbow')
plt.title("Loan Interest Rate by Loan Purpose, Lender's Club")
plt.xlabel('Loan Purpose')
plt.ylabel('Interest Rate')


# In[214]:


a = data.groupby(['verified_income', 'loan_status']).size().reset_index(name='count')
plt.figure(figsize=(18,10))
sns.barplot(x='loan_status',y='count',data=a, palette='rainbow', hue='verified_income')
plt.title("Loan Status by Income Verification")


# Average Interest Rate by State

# In[215]:


plt.figure(figsize=(22,14))
sns.pointplot(x='state',y='interest_rate',data=data)
plt.title("Average Interest Rate by State")


# # Modelling & Training

# ## Data Preprocessing

# In[216]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Feature Engineering Dataset

# Removing rows/data based on null condition of certain columns
# remove rows with missing debt_to_income value (since the numbers are very less it shouldn't effect the accuracy much) - only 24 such rows found in exploration stage.
data = data[data['debt_to_income'].notna()]

# Merging Columns of data based information gained during exploratory analysis stage.

# Replacing null values in annual_income_joint with corresponding values from annual_income column;
data.annual_income_joint.fillna(data.annual_income, inplace=True)
# renaming merged column annual_income_joint as annual_income_merged
data.rename({'annual_income_joint': 'annual_income_merged'}, inplace=True)

# Removing meaningless to the motive features from dataframe
data.drop(columns=['emp_title', 'verification_income_joint', 'annual_income', 'debt_to_income_joint',
          'num_accounts_120d_past_due'], axis=1, inplace=True)  # remove unnecessary columns
# renaming merged column annual_income__merged to annual_income
data.rename({'annual_income_merged': 'annual_income'}, inplace=True)


# Label Encoding categorical values & removing the categorical columns from the dataset
labelEncoder = LabelEncoder()

data['state_cat'] = labelEncoder.fit_transform(data['state'])
data['homeownership_cat'] = labelEncoder.fit_transform(data['homeownership'])
data['verified_income_cat'] = labelEncoder.fit_transform(
    data['verified_income'])
data['loan_purpose_cat'] = labelEncoder.fit_transform(data['loan_purpose'])
data['application_type_cat'] = labelEncoder.fit_transform(
    data['application_type'])
data['grade_cat'] = labelEncoder.fit_transform(data['grade'])
data['sub_grade_cat'] = labelEncoder.fit_transform(data['sub_grade'])
data['issue_month_cat'] = labelEncoder.fit_transform(data['issue_month'])
data['loan_status_cat'] = labelEncoder.fit_transform(data['loan_status'])
data['initial_listing_status_cat'] = labelEncoder.fit_transform(
    data['initial_listing_status'])
data['disbursement_method'] = labelEncoder.fit_transform(
    data['disbursement_method'])


data.drop(columns=[
    'state',
    'homeownership',
    'verified_income',
    'loan_purpose',
    'application_type',
    'grade',
    'sub_grade',
    'issue_month',
    'loan_status',
    'initial_listing_status',
    'disbursement_method',
], inplace=True)



scaler = MinMaxScaler().fit(data.values)
transform = scaler.transform(data.values)

data[data.columns] = transform


# Replacing -1 in case it isn't available (replacement with constant method).
data['months_since_90d_late'].replace(
    np.nan, -1, inplace=True)
data['emp_length'].replace(
    np.nan, -1, inplace=True)
data['months_since_last_credit_inquiry'].replace(
    np.nan, -1, inplace=True)
data['months_since_last_delinq'].replace(
    np.nan, -1, inplace=True)


# In[217]:


# data_without_nulls.head()
print(data.info())
print(data.isnull().sum())
data.head()


# ## Feature Selection

# <font color="red"> We find strong correlation of 7 fields with the interest_rate. </font>

# In[218]:


plt.figure(figsize=(120, 80))
cor = data.corr()
cor_interest_rate = cor['interest_rate']
sns.heatmap(cor, annot=True, cmap='viridis')
print(cor_interest_rate[(cor_interest_rate >= 0.2) | (cor_interest_rate <= -0.2)])


# ## Feature Extraction

# In[219]:


feature_data = data[["total_debit_limit",
                     "term",
                     "interest_rate",
                     "paid_interest",
                     "verified_income_cat",
                     "grade_cat",
                     "sub_grade_cat", ]]
feature_data.isna().sum()


# ## Separating Feature & Target Variable Data

# In[220]:


X, Y = pd.DataFrame(feature_data.loc[:, feature_data.columns != 'interest_rate']), pd.DataFrame(feature_data.loc[:, feature_data.columns == 'interest_rate'])


# ## Training

# Splitting test and training data.

# In[221]:


from sklearn.model_selection import train_test_split
# splitting test and training data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)


# ### XGBoost Regression

# In[222]:


import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as MSE

# fit model on training data
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)

y_pred = xg_reg.predict(X_test)
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE: %f" % (rmse))


# #### XGBoost Model Prediction vs Actual Visualization

# In[223]:


y_test_viz = y_test.to_numpy()
plt.figure(figsize=(10,10))
plt.scatter(y_test_viz, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test_viz))
p2 = min(min(y_pred), min(y_test_viz))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.suptitle('XGBoost Prediction vs Actual Visualization')
plt.title("RMSE: %f" % (rmse))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()


# ### Random Forest Regressor

# In[224]:


from sklearn.ensemble import RandomForestRegressor
import numpy as np
# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
 
# fit the regressor with x and y data
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE: %f" % (rmse))


# #### RandomForestRegressor Prediction vs Actual Visualization

# In[225]:


plt.figure(figsize=(10,10))
plt.scatter(y_test_viz, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test_viz))
p2 = min(min(y_pred), min(y_test_viz))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.suptitle('RandomForestRegressor Prediction vs Actual Visualization')
plt.title("RMSE: %f" % (rmse))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()


# ### MLPRegressor

# In[226]:


from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("MLPRegressor Score: %f" % (regr.score(X_test, y_test)))


# #### MLPRegressor Prediction vs Actual Visualization

# In[227]:


plt.figure(figsize=(10,10))
plt.scatter(y_test_viz, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test_viz))
p2 = min(min(y_pred), min(y_test_viz))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.suptitle('MLPRegressor Prediction vs Actual Visualization')
plt.title("MLPRegressor Score: %f" % (regr.score(X_test, y_test)))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()


# # Summary
# 
# Linear Regression Model still needs some tuning to get more accurate results when compared to ForestRegressor and Neural Network Regressor. \
# As all the steps performed in this analysis and prediction are convention. If provided more time I would:
# <ul>
#   <li>tweak XGB linear regressor model parameters to tune it for better accuracy.</li>
#   <li>go beyond these conventional methods to build more efficient model and inclined to this specific use case.</li>
# </ul>
# 
# Assumptions:
# <ul>
#   <li>emp_title field is irrelevant to the interest rate as most of them are different and contains many null values.</li>
#   <li>24 rows in debt_to_income contains null, so removed those records as they represent less than 1% of the complete dataset.</li>
#   <li>removed verification_income_joint field completely as it is a categorical value with enough nulls that if handled might corrupt the data.</li>
#   <li>removed debt_to_income_joint field from consideration as debt_to_income is defined for the complete dataset i.e without any missing values, so considered just that.</li>
#   <li>num_accounts_120d_past_due discarded from consideration for features for this prediction as it either contains null values or 0's which won't be of any use to this prediction.</li>
#   <li>Replaced null values with -1 (as constant) for the missing values in months_since_90d_late, emp_length, months_since_last_credit_inquiry, months_since_last_delinq properties from the dataset.</li>
#   <li>annual_income_joint supersedes annual_income. annual_income considered when annual_income_joint is null.
# </ul>
