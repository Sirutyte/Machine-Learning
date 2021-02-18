#!/usr/bin/env python
# coding: utf-8

# # Tree Growth Analysis
# 
# ### 1. Methodology
# 
# The aim of the assignment is to examine the relationships between the outcome (Y) and predictor variables (X). The list of variables and corresponding descriptions provided below:
# 
# ##### Outcome variables (Y):
# 
# 1. Tree Survival Rate - % of trees survived/became mature.
# 2. Diameter -  average DBH (cm)
# 3. Height - average height (meters)
# 4. Stand Volume - volume column refers to standing tree volume by cubic meters of all trees
# at the farm at that given age
# 
# ##### Performance drivers (X):
# 
# 1. FA Zone - facilitator zone indicates field staff management variability and agro-ecological
# growing conditions.
# 2. Field Manager - a proxy for management talent.
# 3. Management Quality - farmer classification designated by our operations team of how
# effectively a farmer delivered key actions (weeding, etc.) that are known to impact tree survival and growth.
# 4. Suitability - a high-level classification defined by an external consultant, primarily based on
# digitized versions of 1930’s maps on soil type, combined with his knowledge of silviculture.
# 5. Planting Year - we can expect significant year-to-year variability, mostly driven by fluctuation in
# rainfall timing/volume, plus other longtail factors (e.g. seedling quality or management quality
# within a given year).
# 6. Location - area name. 
# 7. SiteIndexClass - growth rate of the farm.
# 8. Area - total farm land area.
# 9. Age - age of the farm in years. 
# 10. Stems per hectare - number of stems/trees per hectare.
# 
# Since all the outcomes variables are continuous, I will be using regression analysis to estimate the importance of predictors on the outcomes variables. I will create 4 model version using different outcomes variable (Y) and the same list of performance drivers including the control variables listed in the dataset, which were not highlighted for review. In case those variables are significant, it will allow me to reduce the errors and improve precision of insights when it comes to the variables of interest.
# 
# In addition, I will be using different regression algorithms such as Linear, Lasso, Ridge and Random Forest to evaluate the best model fit. 
# 
# ### 2. Challenges and Limitations
# 
# The list of provided variables consists of a number of categorical variables with high cardinality, which makes it’s challenging to use it in the regression analysis (each category is a separate dummy variable and it’s not advisable to have a large number of columns). 
# 
# When it comes to outliers, using a z-score method, I have found that my sample size dropped to N=3327 after removing suspected data points. This means that there’s relatively many data points that need to be further investigated. The process of data collection in the field might, in particular, suggest data quality issues. However, removing outliers did not significantly affect my model results and, taken the fact that I am not able to investigate this further to confirm whether these are natural outliers or data quality issues, I have decided to keep it in the analysis. 
# 
# ### 3. Results
# 
# When it comes to the tree size indicators: volume, diameter and height - all the predictors of interest (FA Zone, field manager, management quality, suitability, planting year) are significant drivers of the performance. Random Forest Regression showed to be the best model fit with the Adj R - Squared for the volume outcome - 0.61, diameter - 0.72 and  height - 0.92. In addition, p-values for all the variables in the list were less than 0.05, showing that all indicators have significant relationships with the outcomes variables. When it comes to SiteIndexClass, it has a positive association with all the tree size indicators, meaning higher index results in larger volume, diameter and height. Similarly, tree age (or year of planting) has a positive association with all the tree size outcomes. On the other hand, area has a negative relationship with volume and diameter, but positive association with height. This is a surprising finding since I would expect all the tree size metrics would be affected similarly by the size of the area planted. When it comes to the interpretation of the categorical variables, it’s less straightforward and would require additional analysis output. 
# 
# In addition, the model for the tree survival rate returned very weak results - Adj R-Squared is at 0.11 (only 11% of variation in the tree survival rate is explained by the X variables), indicating that the model is missing crucial drivers of tree survival rate. This could be factors such as weather conditions (rainfall, hours of sunlight), land fertility indicators etc. Due to such low model predictive power, it's not advisable to comment on the impact of each variable on the outcome variable, however I would notice that land suitability and FA zone seem particularly unrelated to tree survival rate. 
# 
# 
# ### 4. Recommendations
# 
# - Further research is required to better understand the drivers behind the tree survival rate. This is important since the tree survival rate is an important factor influencing business performance. 
# - Further invest in the field manager recruitment and training processes, as they prove to be playing an important role in the productivity of farm. 
# - Similarly, since farmer management quality has benefited the farm performance, additional farmer extension services could help further improve business performance and drive towards profitability.  
# 

# ### The workbook contains of two notebooks: Exploratory Data Analysis (EDA) is presented in a separate file - 3.1. EDA Data Science Irma Sirutyte.html (pandas profile report can only be exported separately). 

# ### Importing libraries

# In[1]:


### Importing the libraries needed for the the analysis
import pandas as pd
from pandas_profiling import ProfileReport

import numpy as np
import seaborn as sb

from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from functools import reduce

from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# ### Data Import

# In[2]:


### import the dataset and look at a preview
df_komaza = pd.read_csv('enumeration_data_komaza.csv')
df_komaza.head()


# ## 1. Exploratory Data Analysis (EDA)
# First step in the analysis is to better understand your data. I will be using the profile pandas library to generate a comprehensive report. The full report is also attached to the main file for you to review. 

# In[3]:


profile = ProfileReport(df_komaza)
#profile.to_widgets()
profile.to_file(output_file='EDA_output.html')


# In[4]:


df_komaza.describe(include='all')


# The dataset consists of 3738 rows which represent farms. 
# - Location: All farms are located in Kenya, Kilifi County, across 27 locations and 137 FA Zones.  
# - Management: The farms are managed by 9 field managers. In addition, there are 4 farm management quality levels (evaluation of farmer's farm management skills).
# - Farms: There are 6 different site index classes and 6 suitability levels of farms. The average size of farms is 0.11 (area unit) and, on average, it's 7.12 years old.
# - Trees: There's only one type of species planted - Eucalyptus GC. Volume of all trees is at 43.3, while the avg tree diameter and height is 12.19 and 12.63 consecutively. Average percentage of tree survival is at 68.5%.     

# Insights when it comes to data preparation for the modeling:
# 
# - County and Species variables are constant, therefore should be droppped.
# - Planting year and age are highly correlated, therefore only one of them can be used in the model. I will be using age (as a proxy for year), as regression analysis handles continuous variables better. 
# - Also stems per hectare should be dropped since it's highly correlated with the tree survival rate.
# - FA Zone variable has high cardinality, which means I should consider reducing dimensions. 
# - Variable area has 3.2% values, which are zero. This might be interesting to investigate later on, though should not significantly affect relationships between variables. 

# Next, I will be changing the column names into consistent lower case and concatenated style, which makes it easier to reference column names in the later steps.

# In[5]:


original_cols = df_komaza.columns
def adjust_columns(col):
    '''
    Function that removes spaces betwen column names, turns them to lower case and concatenates them 
    with underscores
    '''
    ## turn column to lower case, split it by the spaces and then concatenate the words using an underscore
    return "_".join(col.lower().split())
df_komaza.rename(adjust_columns, axis=1, inplace=True) ### Apply the above function to the columns 
df_komaza.head(3)


# # 2. Modeling 

# ### Data Preparation
# 
# - I will assume that tree survival rate can't be higher than 100% and replace all values > 100 with 100%.
# - I will drop the first two character of siteindexclass and transform it to numerical variable (float). I am doing this to avoid creating more dummies (I already have too many high cardinality categorical variables in the model). 
# - Constant variables such as county, species and stand_id, as well as highly correlated variables planting_year and stems_per_hectare will be dropped.

# In[6]:


# Replacing tree survival rates higher than 100%. 
df_komaza.loc[df_komaza['tree_survival_rate'] > 100, 'tree_survival_rate'] = 100

#transforming siteindexclass variable into numeric float: dropping the 2 front letter and changing data type
df_komaza['siteindexclass'] = df_komaza['siteindexclass'].str[2:]
df_komaza['siteindexclass'] = df_komaza['siteindexclass'].astype(float)

#dropping county, species, stand_id, stems_per_hectare and planting_year variables
df_komaza.drop(['stand_id', 'county', 'species','stems_per_hectare','planting_year'], axis=1, inplace=True)
df_komaza.head()


# #### Generating a correlation matrix for all the numeric variables (pearson correlation coeficient)  

# Before adding the variables to the model, it's important to investigate correlations to avoid multicollinearity. 

# In[7]:


data = pd.concat([df_komaza], axis='columns')
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sb.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sb.heatmap(corr, mask=mask,annot=True, cmap=cmap, vmax=.8, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 1});


# Siteindex class seems highly correlated with the tree size outcomes - height, diameter and volume. It should be an important factor while predicting these outcomes. In addition, there are no high correlations between the predictors, therefore no multicollinerity issues. 

# ### 2.1. Regression model (y - volume):
#     
# #### y - volume
# 
# #### X - siteindexclass, area, age, suitability, mgmthquality, field_manager, location, fa_zone.   
# 

# In[8]:


#remove variables that will not be needed in the independent variable list 
cols_independent_drop=df_komaza.drop(['volume','height','tree_survival_rate','diameter'], axis='columns')
cols_independent_drop.tail()

#creating dummies for categorical variables
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])

#assigning volume as y - dependent variable
y=df_komaza.volume

#view of the final X - independend variable list
X.tail()


# Next, I'm splitting my data into train (75%) and test (25%). Train will be used for training the model and test dataset will be used to evaluate the model predictive power at the end of the analysis. Basically, it will give me more confidence in the model accuracy and ensures that there's no issues with my results. 

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[10]:


X_train.head()


# In[11]:


X.info()


# #### Linear Regression (OLS) (using stats library) 

# In[12]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_train)
model.summary()


# - fa_zone variable is not significant. It seems like location and field_manager variables are more dominant and might have explained most it's variation.
# - field manager, area, age and siteindexclass are all significant.

# Model accuracy is quite weak - Adj R-Squared is at 0.65. Therefore, I will continue to investigate whether other algorithms such as Ridge, Lasso or Random Forest Regressor could be a better fit. In addition, I will perform feature transformation and examine if it can help to improve the model accuracy.

# In[13]:


# Creating a method to run the cross validation of the model. 

def run_cross_validation(model, X_train, y_train, cv=10):
    scores = cross_val_score(model, X_train, y_train, cv=cv)

    print('List of fold scores is: {} '.format(scores))
    print('\n')
    print('The mean score is {}'.format(np.mean(scores)))


# #### Linear Regression

# In[14]:


lr = LinearRegression()

run_cross_validation(lr, X_train, y_train)


# In[15]:


from matplotlib import pyplot
model = LinearRegression()
model.fit(X_train, y_train)
# get importance
importance = model.coef_
# summarize feature importance
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.2f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# #### Ridge Regression

# In[16]:


ridge=Ridge()

run_cross_validation(ridge, X_train, y_train)


# #### Lasso Regression

# In[17]:


lasso = Lasso()

run_cross_validation(lasso, X_train, y_train)


# #### Random Forest Regression

# In[18]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# Comparing the results for Linear Regression (0.63), Lasso Regression (0.57), Ridge Regression (0.63) and Random Forest Regression (0.67), I can see that Random Forest Regression performed the best (0.67 R-Adjusted). Lasso and Ridge performed equally or lower than the Linear Regression model. Therefore, I will only use Random Forest Regression and Linear Regression in the following steps. In addition, overall the model predictive power is relatively low. I will investigate several potential solutions:
# - scaling the variables
# - examining the outliers
# - reducing the dimensions for high cardinality variables, such as fa_zone and location.
# - potentially adding polynomial features

# ### Scaling the numeric variables

# In[19]:


numeric_cols = ['siteindexclass', 'area', 'age']
scaler = StandardScaler()

scaled_train = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns = numeric_cols)
scaled_test = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns = numeric_cols)

X_train.reset_index(drop=True, inplace=True)
X_train.drop(numeric_cols, axis=1, inplace=True)
X_train = pd.concat([X_train, scaled_train], axis=1)


X_test.reset_index(drop=True, inplace=True)
X_test.drop(numeric_cols, axis=1, inplace=True)
X_test = pd.concat([X_test, scaled_test], axis=1)


# #### Repeating the Random Forest Regression to see if scaling the variables has helped with the predictive model 

# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# Scaling the numeric variables has a significant effect on the Random Forest Regression accuracy - Adj R-Squared increased up to 0.67.

# ### Dimension reduction
# 
# I will be reducing the number of categories for variables that have high cardinality. I will maintain 10 most common categories and merge the remaining ones into one category called Other.

# In[ ]:


# reducing the number of dimensions up to 10 categories

df_komaza['location'] = df_komaza['location'].astype('category')
others = df_komaza['location'].value_counts().index[10:]
label = 'Other'
df_komaza['location'] = df_komaza['location'].cat.add_categories([label])
df_komaza['location'] = df_komaza['location'].replace(others, label)
df_komaza['location'].value_counts(normalize=True)


# reducing the number of dimensions up to 10 categories
df_komaza['fa_zone'] = df_komaza['fa_zone'].astype('category')
others = df_komaza['fa_zone'].value_counts().index[10:]
label = 'Other'
df_komaza['fa_zone'] = df_komaza['fa_zone'].cat.add_categories([label])
df_komaza['fa_zone'] = df_komaza['fa_zone'].replace(others, label)
df_komaza['fa_zone'].value_counts(normalize=True)


# In[ ]:


#remove variables that will not be needed in the independent variable list 
cols_independent_drop=df_komaza.drop(['volume','height','tree_survival_rate','diameter'], axis='columns')
cols_independent_drop.tail()

#creating dummies for categorical variables
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])

#assigning volume as y - dependent variable
y=df_komaza.volume

#view of the final X - independend variable list
X.tail()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[ ]:


X_train.head()


# Reduced the number of columns from 186 columns to 44 columns.

# #### Linear Regression

# In[ ]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_train)
model.summary()


# #### Random Forest Regression

# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_test, y_test, cv=5)


# In[ ]:


rf.fit(X_train, y_train)
rf.feature_importances_


# #### Outlier Investigation

# Looking at pandas profile report and the scatterplots, I can see that there are some obvious outliers. Since this data is based on the measurements in the field, data quality issues might be a likely concern. However, I will investigate what effect does it have on the modeling and decide if it's apppropriate to drop these outliers.  

# In[ ]:


#Investogating outliers in relation to the dependent variable - volume

fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(15, 4))
axes = np.ravel(axes)
col_name = ['diameter','height','age','area','tree_survival_rate']
for i, c in zip(range(5), col_name):
    df_komaza.plot.scatter(ax=axes[i], x=c, y='volume', sharey=True, colorbar=False, c='g')


# #### Removing outliers
# In order to examine how the model would perform without outliers, I will drop them and create a new dataset. 

# In[ ]:


#separate numerical variables for outlier investigation
columns_num = ['diameter','volume','height','area']
df_numeric=df_komaza[columns_num]

# Thereshold for considering a point an outlier will be 2 standard deviations 
z = np.abs(stats.zscore(df_numeric))

#print(z)
threshold = 2

#print(np.where(z > 2))

#I will craete a new dataframe which will have no outliers. This will allow me to run the two model versions (full dataset vs. dataset with no outliers) and compare the results.  
df_komaza_o = df_komaza[(z < 2).all(axis=1)]
df_komaza_o.shape


# In[ ]:


#remove variables that will not be needed in the independent variable list 
cols_independent_drop=df_komaza_o.drop(['volume','height','tree_survival_rate','diameter'], axis='columns')
cols_independent_drop.tail()

#creating dummies for categorical variables
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])

#assigning volume as a dependent variable
y=df_komaza_o.volume
X.tail()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# #### Linear Regression

# In[ ]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_train)
model.summary()


# #### Random Forest Regression

# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# Removing outliers has almost no impact on the linear regression model, however it has reduced the Random Forest Regression predictive power. In addition, I do not have sufficient information about the outliers to decide whether these points are errors or just natural outliers that might be useful for my analysis. 

# ### 2. Regression model (y - diameter)
#     
# #### y - diameter
# 
# #### X - area, age, fa_zone, suitability, mgmthquality, siteindexclass, field_manager, location   
#    

# In[ ]:


cols_independent_drop=df_komaza.drop(['diameter','volume','height','tree_survival_rate'], axis='columns')
cols_independent_drop.tail()
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])
X.tail()
y=df_komaza.diameter
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# #### Linear Regression

# In[ ]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X)
model.summary()


# #### Random Forest Regression

# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_test, y_test, cv=5)


# ### 3. Regression model (y - height)
#     
# #### y - height
# 
# #### X - area, age, fa_zone, suitability, mgmthquality, siteindexclass, field_manager, location   
# 

# In[ ]:


cols_independent_drop=df_komaza.drop(['diameter','volume','tree_survival_rate','height'], axis='columns')
cols_independent_drop.tail()
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])
X.tail()
y=df_komaza.height
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[ ]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X)
model.summary()


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_test, y_test, cv=5)


# ### 4. Regression model (y - tree survival rate)
#     
# #### y - tree survival rate
# 
# #### X - area, age, fa_zone, suitability, mgmthquality, siteindexclass, field_manager, location   

# In[ ]:


cols_independent_drop=df_komaza.drop(['diameter','volume','tree_survival_rate','height'], axis='columns')
cols_independent_drop.tail()
X=pd.get_dummies(cols_independent_drop, columns=['suitability','mgmtquality','field_manager','location','fa_zone'])

y=df_komaza.tree_survival_rate
#X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[ ]:


model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X)
model.summary()


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_train, y_train, cv=5)


# In[ ]:


rf = RandomForestRegressor(n_estimators=50, random_state=0)
print('Running cross validation .....')
run_cross_validation(rf, X_test, y_test, cv=5)

