#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# Que1: What is CNN? How does it work behind the scenes?
# 
# Que2: What are Stride, Padding, Kernel Filters, and Pooling?
# 
# Que3: Why does Overfitting happen in CNN, and how can you avoid it?
# 
# Que4: Why is InceptionNet better than VGG?
# 
# Que5: What is Augmentation?
# 
# Que6: Can you explain the concept of feature maps in CNNs?

# In[1]:


1) 
CNN stands for convolutional neural nework used extensively when dealing wih images.Basically it extracts the features from the images.
To extract the features,we have to convolve the image with filters.Accordingly,we have so many filters examples,Horizontal kernel filters also called as slobe filters,vertical filters..etc.
after convolving,we get the respective output for the image.say for example,if we have 6X6 image and 3x3 filter, we will get 4x4 size of output.


quetion 2
 Stride is nothing but step size ie the number of steps move forward.
    eg if stride=1 move the filter one step 
          stride =2 move the filter 2 step forward
        
    In padding it adds additional pixels for both side,so that we can avoid lossing the important features in the images wen the image size is reduced.     
   In padding there are 2 types
        a.VALID aslo called as 0 padding
        b.SAME
        
        
    Kernel filters are basically used to extract the features from the images,we have lots of kernel filters and sizes
    eg,horizontal,vertical based on we have many size like 3x3,5x5,11x11 etc
    
    Pooling is used to reduce the dimentionality of the image.
    In pooling we have 3 types.
    a.maxpooling
    b.avg pooling
    c.minpooling
    
question3.

As the number of layers increases,it also increase the parameter.it may leads to an overfitting.To avoid overfitting we have many techniques,
1. Batch normalization
2.Early stopping
3.Dropout
4.Weight initialization.

qns4

Even though the VGG has small size filters(3x3) and reduced parameters,still it has a overfitting problems,to avoid overfitting,
another model inceptional net is used,here,it goes deeper as well as wider.instead of convolving directly with the 3x3,5x5 filter first convolve with 1x1 filter,so that it reduces the dimension as well.


qns 5

Data Augmentation,is basically used to increase the size of the dataset.
eg-cropping,tilt the image,random up and down etc.

        



# # Machine Learning Techniques
# 
# ### Problem statement and Objective
# 
# #### Black Friday Project
# 
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month. Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.
# 
# 
# 
# 
# 

# ### Data Variable Definition 
# • User_ID User ID 
# 
# 
# • Product_ID Product ID 
# 
# 
# • Gender Sex of User 
# 
# 
# • Age Age in bins 
# 
# 
# • Occupation Occupation (Masked)
# 
# 
# • City_Category Category of the City (A,B,C) 
# 
# 
# • Stay_In_Current_City_Years Number of years stay in current city 
# 
# 
# • Marital_Status Marital Status 
# 
# 
# • Product_Category_1 Product Category (Masked)
# 
# 
# • Product_Category_2 Product may belongs to other category also (Masked) 
# 
# 
# • Product_Category_3 Product may belongs to other category also (Masked)
# 
# 
# • Purchase Purchase Amount (Target Variable) 
# 
# 

# ### Goal
# 
# Our goal is to predict the purchase amount of customers for various products after completing all the necessary preprocessing steps. Additionally, hyperparameter tuning and cross validation is essential. We also need to apply feature selection techniques such as SelectKBest, VIF, and PCA. 

# #Dataset Link
# 
# 
# https://raw.githubusercontent.com/s4sauravv/Datasets/main/Black%20Friday.csv

# 
# You have to use multiple algorithms to build the model, and whichever algorithm performs the best, you have to do hyperparameter tuning for it. After tuning the hyperparameters, you also need to plot its best fit line.

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[3]:


purchase_dataset=pd.read_csv('https://raw.githubusercontent.com/s4sauravv/Datasets/main/Black%20Friday.csv')


# In[4]:


purchase_dataset.head()


# In[5]:


purchase_dataset.shape


# In[6]:


purchase_dataset.info()


# In[7]:


purchase_dataset.isnull().sum()


# In[8]:


purchase_dataset.drop_duplicates(inplace=True)


# In[9]:


purchase_dataset['Product_Category_2']=purchase_dataset['Product_Category_2'].fillna(purchase_dataset['Product_Category_2'].mean())


# In[10]:


purchase_dataset['Product_Category_3']=purchase_dataset['Product_Category_3'].fillna(purchase_dataset['Product_Category_3'].mean())


# In[11]:


purchase_dataset.isnull().sum()


# In[12]:


purchase_dataset.drop(columns=['User_ID','Occupation','Stay_In_Current_City_Years'],axis=1,inplace=True)


# In[13]:


purchase_dataset.shape


# In[14]:


purchase_dataset.head()


# In[15]:


purchase_dataset['Age'].value_counts()


# In[18]:


purchase_dataset['Age']=purchase_dataset['Age'].replace({'26-35':1,'36-45':2,'18-25':3,'46-50':4,'51-55':5,'55+':6,'0-17':7})


# In[19]:


purchase_dataset.head()


# In[20]:


purchase_dataset['Gender'].value_counts()


# In[21]:


purchase_dataset['Gender']=purchase_dataset['Gender'].replace({'M':0,'F':1})


# In[22]:


purchase_dataset.head()


# In[23]:


purchase_dataset['City_Category'].value_counts()


# In[24]:


purchase_dataset['City_Category']=purchase_dataset['City_Category'].replace({'A':1,'B':2,'C':3})


# In[25]:


purchase_dataset.head()


# In[26]:


purchase_dataset.info()


# In[27]:


purchase_dataset.drop(columns='Product_ID',axis=1,inplace=True)


# In[28]:


purchase_dataset.info()


# In[32]:


plt.figure(figsize=(10,10),facecolor='orange')
number = 1
for column in purchase_dataset:
    if number <=7:
        ax=plt.subplot(3,3,number)
        sns.boxplot(purchase_dataset[column])
        plt.xlabel(column,fontsize=20)
        number +=1
plt.tight_layout()


# In[35]:


purchase_dataset.head()


# In[37]:


x=purchase_dataset.drop(columns='Purchase',axis=1)
y=purchase_dataset.Purchase


# In[54]:


purchase_dataset['Product_Category_2'].fillna(-999, inplace=True)
purchase_dataset['Product_Category_3'].fillna(-999, inplace=True)


# In[63]:


purchase_dataset.info()


# In[55]:


print(x)


# In[56]:


print(y)


# In[57]:


scaler=StandardScaler()


# In[58]:


x_scaler=scaler.fit_transform(x)


# In[67]:


x=x_scaler
y=purchase_dataset.Purchase


# In[68]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[69]:


print(x.shape,x_train.shape,x_test.shape)


# In[70]:


model=LogisticRegression()


# In[72]:


model.fit(x_train,y_train)


# model.fit(x_train,y_train)

# In[71]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[66]:


y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2: {r2}')


# In[ ]:




