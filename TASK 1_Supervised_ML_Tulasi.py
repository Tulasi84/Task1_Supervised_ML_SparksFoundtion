#!/usr/bin/env python
# coding: utf-8

# **AUTHOR NAME : TULASI VENKATA SIRI**
# 
# **ORGANISATION  : Sparks Foundation**

# # Data Science and Business Analytics Internship at The Sparks Foundation

# # TASK 1: Prediction using Supervised Machine Learning
#     
# **The aim of the task is to predict the percentage of a student based on the no. of study hours using the linear Regression Supervised machine learning algorithm.** 
# 
# **Steps to be followed :**
#     
# **STEP 1:** Importing the dataset
# 
# **STEP 2:** Visualizing the dataset
# 
# **STEP 3:** Data Preparation
# 
# **STEP 4:** Training the algorithm
# 
# **STEP 5:** Visualizing the model
# 
# **STEP 6:** Making Predictions
# 
# **STEP 7:** Evaluating the model

# # STEP 1: Importing the libraries and dataset
# 
# **In this step we will import the required libraries and dataset through the link given**

# In[101]:


#Importing all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


#Reading the data from the link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# In[103]:


#Lets observe the dataset
data.head(5)


# In[104]:


data.tail(6)


# In[105]:


# To find the shape of the dataset
data.shape


# In[106]:


data.describe()


# In[107]:


#To check whether there exist any null or missing values in the dataset
data.isnull().sum()


# # STEP 2: Visualizing the dataset
# 
# **We will plot the dataset and check whether there exist any relation between the variables**

# In[108]:


#Plotting the dataset
plt.rcParams["figure.figsize"]=[11,7]
data.plot(x = 'Hours', y = 'Scores', style='*', color = 'brown', markersize = 8)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# By observing the above graph we found that there exist a linear relationship between "hours studied" and "percentage score". so, we will use linear regression supervised machine model and predict the coming values. 

# In[109]:


#To find correlation between variables
data.corr()


# # STEP 3: Data Preparation

# 
# **In the step we will divide the data into inputs and outputs and then divide the whole dataset into two parts**
# 1) Testing data
# 
# 2) Training data

# In[113]:


#here we used iloc function in order to divide the data
X = data.iloc[:, :1].values
Y = data.iloc[:, 1:].values


# In[114]:


X


# In[115]:


Y


# In[116]:


#Splitting the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# # STEP 4: Training the Algorithm
# 
# **we have split the data into training and testing sets, and now its finally the time to train algorithm**
#     

# In[94]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)


# # STEP 5: Visualizing the data
# **Visualizing the model after training it**

# In[95]:


line = model.coef_*X + model.intercept_

#plotting for the training data
plt.rcParams["figure.figsize"]=[11,7]
plt.scatter(X_train, Y_train, color='red')
plt.plot(X, line, color='blue');
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[97]:


plt.rcParams["figure.figsize"]=[11,7]
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X, line, color = 'blue')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# # STEP 6: Making Predictions
# 
# **After training the algorithm, then we will make some predictions**

# In[98]:


Y_predicted = model.predict(X_test)


# In[99]:


#comparing the actual and predicted data
data = pd.DataFrame({'Actual score':Y_test, 'Predicted score':Y_predicted})
data


# In[100]:


hrs = 9.25
own_prediction = model.predict([[hrs]])
print("The predicted score if a person studies for",hrs , "hours is", own_prediction[0])


# If a student studies for 9.25 hours then the predicted score is 93.69173248737538

# # STEP 7: Evaluating the model
# 
# **Final Step, we are going to evaluate our trained model by calculating mean absolute error**

# In[50]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predicted))


# In[ ]:




