#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Libraries used in the project
import numpy as np # linear algebra
import pandas as pd # data processing
import os
import matplotlib.pyplot as plot
import seaborn as sb


# In[150]:


for directory, _, filenames in os.walk('C:\\Users\\LaptopCheckout\\Downloads\\creditcard.csv\\'):
    for file in filenames:
        print(os.path.join(directory, file))
import warnings
warnings.filterwarnings("ignore")


# In[13]:


#Read the datafile
df = pd.read_csv("C:\\Users\\LaptopCheckout\\Downloads\\creditcard.csv\\creditcard.csv")


# In[14]:


#Read the header of the datafile
df.head()


# In[15]:


#Number of observation and columns
df.shape


# In[16]:


#Summary of the data
df.describe()


# In[152]:


#Checking null values
df.isnull().any()


# In[35]:


#Exploratory Data Analysis
#For Graphical details for bar chart
sb.histplot(df['Amount'],bins =150,color="blue")
sb.set_style("darkgrid")
# As we can see there is no significant distribution in the data, so we need to transform the data distribution. So we will use log of amount.


# In[39]:


#Using log distribution
sb.set_style("whitegrid")
df["log_amount"] = np.log2(df["Amount"]+0.05)
sb.displot(x = "log_amount",bins = 30, kde = True, hue = "Class", data=df)


# In[40]:


#Boxplot graphs to understand the Amount vs Class and Log_Amount vs class
sb.set_style("ticks")
fig,ax  = plot.subplots(ncols = 2,nrows =1,figsize = (15,15))
ax.flatten()
sb.boxplot(x = "Class", y = "Amount", data=df, ax = ax[0])
sb.boxplot(x = "Class",y = "log_amount", data =df, ax = ax[1])


# In[20]:


#Fraud vs not-fraud in bar graph 
sb.set_style("darkgrid")
plot.figure(figsize = (10,6))
sb.countplot(x = "Class", data=df)


# In[21]:


#Number of fraud and not-fraud 
fraud = df[df["Class"]==1]
not_fraud = df[df["Class"]==0]
print(fraud.shape,not_fraud.shape)


# In[22]:


#Dropping axis 1 of class
x = df.drop(["Class"], axis = 1)
y = df["Class"]
x.head()


# In[42]:


#The data is highly imbalanced since the ratio of fraud vs not-fraud is not in the equal proportion. To solve this problem we have three options 1. over populize the fraud, 2 - reduce not=fraud and 3 - using SMOTE we can generate synthetic values for fraud cases
from imblearn.under_sampling import NearMiss
nm = NearMiss()
x_nm, y_nm = nm.fit_resample(x, y)
print(x_nm.shape,y_nm.shape)


# In[153]:


#Standarizing the dataset 
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x_nm)


# In[44]:


#Creating new models for the problem and comparising two models
from sklearn.model_selection import train_test_split, cross_val_score
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_nm, test_size = 0.25)


# In[132]:


Model_Comparison = {}
Accuracy_Value = []
Cross_Validation_values = []
def model(model):
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    print("Accuracy: {}".format(score))
    cv_score = cross_val_score(model,x_train,y_train,cv=5)
    print("Cross Val Score: {}".format(np.mean(cv_score)))
    Accuracy_Value.append(score)
    Cross_Validation_values.append(np.mean(cv_score))


# In[133]:


from sklearn.linear_model import LogisticRegression
Model1 = LogisticRegression()
model(Model1)


# In[134]:


from sklearn.ensemble import RandomForestClassifier
Model2= RandomForestClassifier()
model(Model2)


# In[135]:


from sklearn.tree import DecisionTreeClassifier
Model3 = DecisionTreeClassifier()
model(Model3)


# In[136]:


from sklearn.neighbors import KNeighborsClassifier
Model4 = KNeighborsClassifier()
model(Model4)


# In[137]:


from sklearn.svm import SVC
Model5= SVC()
model(Model5)


# In[138]:


from sklearn.naive_bayes import GaussianNB
Model6 = GaussianNB()
model(Model6)


# In[139]:


from sklearn.ensemble import AdaBoostClassifier
Model7 = AdaBoostClassifier()
model(Model7)


# In[140]:


from sklearn.ensemble import GradientBoostingClassifier
Model8 = GradientBoostingClassifier()
model(Model8)


# In[141]:


models = ["LogisticRegression","RandomForestClassifier",
          "DecisionTreeClassifier","KNeighborsClassifier","SVC",
          "GaussianNB","AdaBoostClassifier","GradientBoostingClassifier"]


# In[142]:


Accuracy_Value


# In[143]:


Cross_Validation_values


# In[144]:


Model_Comparison = { "Model Name" : models , "Accuracy Score" : Accuracy_Value, "Cross val Score": Cross_Validation_values}


# In[145]:


Comparsion = pd.DataFrame(Model_Comparison)


# In[146]:


Comparsion


# In[154]:


plot.figure(figsize = (20,10))
sb.barplot(x = "Model Name", y = "Accuracy Score", data=Comparsion)
plot.title("Model Comparison Versus Accuracy Score")


# In[158]:


plot.figure(figsize = (20,10))
sb.barplot(x = "Model Name", y = "Cross val Score", data=Comparsion)
plot.title("Model Comparision versus Cross validation")


# In[ ]:





# In[ ]:




