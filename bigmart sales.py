#!/usr/bin/env python
# coding: utf-8

# In[270]:


# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[271]:


# importing dataset
data = pd.read_csv("C:/Users/hritw/Downloads/data sci pdf/ML/train_bm.csv")


# In[272]:


data.head()


# In[273]:


data.shape


# # Performing Exploratory Data Analysis

# In[274]:


# checking variables
data.dtypes


# UNIVARIATE ANALYSIS

# In[275]:


# Measures of Central Tendency : Mean, Median, Mode


# In[276]:


# 1. Mean


# In[277]:


# mean of item MRP
data['Item_MRP'].mean()


# In[278]:


# mean of sales
data['Item_Outlet_Sales'].mean()


# In[279]:


# mean of item weight
data['Item_Weight'].mean()


# In[280]:


# 2. Median 


# In[281]:


# median of item MRP
data['Item_MRP'].median()


# In[282]:


# median of total sales
data['Item_Outlet_Sales'].median()


# In[283]:


# median of item weight
data['Item_Weight'].median()


# In[284]:


# 3. Mode


# In[286]:


# outlet size with highest frequency
data['Outlet_Size'].mode()


# In[287]:


# checking the frequencies 
data['Outlet_Size'].value_counts()


# In[288]:


# outlet type with highest frequency
data['Outlet_Type'].mode()


# In[289]:


# checking frequencies 
data['Outlet_Type'].value_counts()


# In[290]:


# item type with highest frequency
data['Item_Type'].mode()


# In[291]:


# checking the frequencies
data['Item_Type'].value_counts()


# In[292]:


# Measures of Dispersion : Range, Quartile, IQR, Variance, Standard Deviation


# In[293]:


# 1. Range


# In[294]:


# storing all the numerical columns
numerical_cols = data.select_dtypes(include=['int','float']).columns
numerical_cols


# In[295]:


# range for all the numerical columns
for col in numerical_cols:
    print("range of{}{}{}{}{}{}{}{}".format(col,":"," ","[",data[col].min(),",",data[col].max(),"]"))


# In[296]:


# 2. standard deviation


# In[297]:


data['Item_MRP'].std()


# In[298]:


# 3. variance


# In[299]:


data['Item_MRP'].var()


# In[300]:


data.describe()


# In[301]:


# graphical representation


# In[302]:


# setting img resolution :
plt.figure(figsize= (8,4), dpi = 140)
# plotting histogram 
plt.hist(data.Item_MRP, bins=20, color='lightblue')
#axes label:
plt.xlabel('Item_MRP')
plt.ylabel('frequency')
plt.title('Price of Items')


# In[303]:


# setting img resolution :
plt.figure(figsize= (8,4), dpi = 120)
# plotting histogram 
plt.hist(data.Outlet_Establishment_Year, bins=10)
#axes label:
plt.xlabel('Outlet_Establishment_Year')
plt.ylabel('frequency')
plt.title('year of outlet establishment')


# In[304]:


# plotting KDE plot with descriptive
plt.figure(dpi=140)
sns.kdeplot(data.Item_Weight,shade=True)
sns.scatterplot([data.Item_Weight.mean()],[0],color='red',label='mean')
sns.scatterplot([data.Item_Weight.median()],[0],color='green',label='median')
plt.xlabel('Item_Weight')
plt.ylabel('density')
plt.title('item weight mean & median')
plt.show()


# In[305]:


plt.figure(dpi = 140)
ax = sns.barplot(x=data['Outlet_Type'].value_counts().index, y = data['Outlet_Type'].value_counts().values)
plt.xlabel("Outlet Types")
plt.ylabel("value counts")
plt.title("Bar Plot for Outlet Types")

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.3,p.get_height())),
    color='black'


# In[306]:


plt.figure(dpi = 140)
ax = sns.barplot(x=data['Outlet_Size'].value_counts().index, y = data['Outlet_Size'].value_counts().values)
plt.xlabel("Outlet Size")
plt.ylabel("value counts")
plt.title("Bar Plot for Outlet Size")

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.3,p.get_height())),
    color='black'


# In[307]:


data.head()


# In[308]:


meaniv= data['Item_Visibility'].mean()
mediv= data['Item_Visibility'].median()
modeiv= data['Item_Visibility'].mode()
meaniv, mediv, modeiv


# In[309]:


# plotting KDE plot with descriptives:
plt.figure(dpi=140)
sns.kdeplot(data['Item_Visibility'],shade=True)
sns.scatterplot([meaniv],[0],color='red',label='mean')
sns.scatterplot([mediv],[0],color='green',label='median')
sns.scatterplot([modeiv[0]],[0],color='blue',label='mode')


# In[310]:


data['Item_Visibility'].skew()


# In[311]:


data['Item_Visibility'].kurtosis()


# - we can say from above plot and calculation that item visibility is positively skewed where mean>median>mode.
# - positive kurtosis indicates leptokurtic distribution of the tails
# - which means high presence of extreme values.

# BIVARIATE ANALYSIS

# In[312]:


# bivariate ananlysis broadly divided into 3 categories :
# 1. Numerical - Numerical variables
# 2. Numerical - Categorical variables
# 3. Categorical - Categorical variables


# In[313]:


# 1. Numerical - Numerical variables : analysis can be done using correlation and covariance


# In[314]:


data.corr()


# In[315]:


# view top 10 highest correlation b/w two variables :
c = data.corr().abs()
s = c.unstack()
top_corr = s.sort_values(kind='quicksort',ascending=False)
corr_df = pd.DataFrame(top_corr, columns=['Pearson correlation'])


# In[316]:


corr_df[corr_df['Pearson correlation']<1].head(10)


# In[317]:


# plotting heatmap
plt.figure(figsize=(36,6),dpi=140)
for j,i in enumerate (['pearson','kendall','spearman']):
    plt.subplot(1,3,j+1)
    correlation = data.dropna().corr(method=i)
    sns.heatmap(correlation, linewidth=2)
    plt.title(i,fontsize=18)


# - above correlation table and heat map clearly shows that only Item_MRP and Item_Outlet_sales has moderate degree of correlation other variables have either low degree or no correlation b/w them.

# In[318]:


# 2. Numerical - Categorical variables


# In[319]:


# barplot
sns.barplot(x=data['Item_Outlet_Sales'], y=data['Outlet_Size'])
plt.title('categorical numerical : barplot')


# In[320]:


sns.boxplot(x=data['Item_Outlet_Sales'], y=data['Outlet_Size'])
plt.title('categorical-numerical : boxplot')


# In[321]:


# 3. categorical - categorical variables


# In[322]:


pd.crosstab(data['Outlet_Size'],data['Item_Type'])


# In[323]:


pd.crosstab(data['Outlet_Location_Type'],data['Outlet_Size'])


# In[324]:


# plotting grouped plot
sns.countplot(x=data['Outlet_Location_Type'], hue=data['Outlet_Size'])
plt.title('categorical-categorical : grouped barplot')


# In[325]:


ax1 = data.groupby('Outlet_Location_Type').Outlet_Size.value_counts(normalize=True).unstack()
ax1.plot(kind='bar', stacked='True',title=str(ax1))


# MULTIVARIATE ANANLYSIS

# In[326]:


# tabular method for multivariate analysis is pivot table. Graphical method can be scatter plot, grouped boxplot, etc.


# In[327]:


table= pd.pivot_table(data,index=['Item_Type','Outlet_Size'], values='Item_MRP', aggfunc='mean')
table


# In[328]:


sns.boxplot(x=data.Outlet_Location_Type, y=data.Item_Outlet_Sales, hue=data.Outlet_Size, orient='v')
plt.title('Boxplot')


# In[329]:


data.head()


# In[330]:


sns.relplot(x="Item_MRP", y="Item_Outlet_Sales", hue="Item_Type", data=data[:200])


# In[331]:


sns.relplot(x="Item_MRP", y="Item_Outlet_Sales",data=data[:200], kind="scatter", size="Item_Visibility", hue="Item_Visibility")


# MISSING VALUE TREATMENT

# In[332]:


# checking missing values
data.isna().sum()


# In[333]:


mean_val = data['Item_Weight'].mean()
mean_val


# In[334]:


data['Item_Weight']= data['Item_Weight'].fillna(value=mean_val)


# In[335]:


mode_val = data['Outlet_Size'].mode()
mode_val


# In[336]:


data['Outlet_Size'] = data['Outlet_Size'].fillna(value= 'Medium')


# In[337]:


data.isna().sum()


# OUTLIER IDENTIFICATION

# In[338]:


sns.boxplot(x=data.Item_Weight)


# In[339]:


sns.boxplot(x=data.Item_Visibility)


# In[340]:


sns.boxplot(x=data.Item_MRP)


# In[341]:


sns.boxplot(x=data.Item_Outlet_Sales)


# In[342]:


sns.boxplot(x=data.Outlet_Establishment_Year)


# # Predictive Modeling

# BENCHMARK MODEL

# In[344]:


from sklearn.utils import shuffle

# Shuffling the Dataset
data = shuffle(data, random_state = 42)

#creating 4 divisions of dataset
div = int(data.shape[0]/4)

# 3 parts to train set and 1 part to test set
train = data.loc[:3*div+1,:]
test = data.loc[3*div+1:]


# In[345]:


train.shape, test.shape


# In[346]:


# SIMPLE MEAN
# making prediction model
# storing simple mean in a new column in the test set as "simple_mean"
test['simple_mean'] = train['Item_Outlet_Sales'].mean()


# In[347]:


#calculating mean absolute error
from sklearn.metrics import mean_absolute_error as MAE

simple_mean_error = MAE(test['Item_Outlet_Sales'] , test['simple_mean'])
simple_mean_error


# In[348]:


# Mean Item Outlet Sales with respect to Outlet_Type
out_type = pd.pivot_table(train, values='Item_Outlet_Sales', index = ['Outlet_Type'], aggfunc=np.mean)
out_type


# In[349]:


# initializing new column to zero
test['Out_type_mean'] = 0

# For every unique entry in Outlet_Identifier
for i in train['Outlet_Type'].unique():
    # Assign the mean value corresponding to unique entry
    test['Out_type_mean'][test['Outlet_Type'] == str(i)] = train['Item_Outlet_Sales'][train['Outlet_Type'] == str(i)].mean()


# In[350]:


#calculating mean absolute error
out_type_error = MAE(test['Item_Outlet_Sales'] , test['Out_type_mean'])
out_type_error


# In[351]:


# Mean Item_Outlet_Sales with respect to both Outlet_Location_Type and Outlet_Establishment_Year
combo = pd.pivot_table(train, values = 'Item_Outlet_Sales', index = ['Outlet_Location_Type','Outlet_Establishment_Year'], aggfunc = np.mean)
combo


# In[352]:


# Initiating new empty column
test['Super_mean'] = 0

# Assigning variables to strings ( to shorten code length)
s2 = 'Outlet_Location_Type'
s1 = 'Outlet_Establishment_Year'

# For every Unique Value in s1
for i in test[s1].unique():
   # For every Unique Value in s2
    for j in test[s2].unique():
       # Calculate and Assign mean to new column, corresponding to both unique values of s1 and s2 simultaneously
        test['Super_mean'][(test[s1] == i) & (test[s2]==str(j))] = train['Item_Outlet_Sales'][(train[s1] == i) & (train[s2]==str(j))].mean()


# In[353]:


#calculating mean absolute error
super_mean_error = MAE(test['Item_Outlet_Sales'] , test['Super_mean'] )
super_mean_error


# KNN MODEL

# In[354]:


data = pd.get_dummies(data.drop(['Item_Identifier'],axis=1))
data.head(10)


# In[355]:


# SEGREGATING INDEPENDENT AND DEPENDENT VARIABLES
x = data.drop(['Item_Outlet_Sales'], axis=1)
y = data['Item_Outlet_Sales']
x.shape, y.shape


# In[356]:


# SCALING THE DATA USING MINMAXSCALER
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[357]:


x = pd.DataFrame(x_scaled)


# In[358]:


# IMPORT TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y, random_state=56)


# In[359]:


# IMPLEMENTING KNN REGRESSOR
# importing KNN regressor and metric MSE
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as MSE


# In[360]:


# creating instance of KNN
reg = KNN(n_neighbors=5)
# fitting the model
reg.fit(train_x,train_y)
# predicting over the train set and calculating MSE
test_predict = reg.predict(test_x)
k = MSE(test_predict, test_y)
print('test MSE', k)


# In[361]:


# ELBOW FOR REGRESSOR
def elbow(k):
    # initiating empty list
    test_MSE = []
    
    # training model for every value of k 
    for i in k:
        # instance of KNN
        reg = KNN(n_neighbors=i)
        reg.fit(train_x,train_y)
        # appending f1 score to empty list calculated using predictions
        tmp = reg.predict(test_x)
        tmp = MSE(tmp,test_y)
        test_MSE.append(tmp)
    return test_MSE


# In[362]:


# defining k range
k = range(1,40)


# In[363]:


# calling above defined fn
test = elbow(k)


# In[364]:


# plotting the values in curve
plt.plot(k,test)
plt.xlabel('k neighbors')
plt.ylabel('mean squared error')
plt.title('elbow curve for test')


# In[366]:


# creating instance of KNN
reg = KNN(n_neighbors=7)
# fitting the model
reg.fit(train_x,train_y)
# predicting over the train set and calculating MSE
test_predict = reg.predict(test_x)
k = MSE(test_predict, test_y)
print('test MSE', k)


# LINEAR MODEL

# In[367]:


data.head()


# In[368]:


# Segregating variables: Independent and Dependent Variables
x = data.drop(['Item_Outlet_Sales'], axis=1)
y = data['Item_Outlet_Sales']
x.shape, y.shape


# In[369]:


# Splitting the data into train set and the test set
# Importing the train test split function
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[370]:


# Implementing Linear Regression
#importing Linear Regression and metric mean square error
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae


# In[371]:


# Creating instance of Linear Regresssion
lr = LR(normalize = True)
# Fitting the model
lr.fit(train_x, train_y)


# In[372]:


# Predicting over the Train Set and calculating error
train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Training Mean Absolute Error', k )


# In[373]:


# Predicting over the Test Set and calculating error
test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Test Mean Absolute Error', k )


# In[374]:


plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient plot')


# In[ ]:




