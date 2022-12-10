#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']=(13,9)
sns.set_style('darkgrid')


# In[2]:


raw_data=pd.read_csv('creditcard.csv')
raw_data


# In[3]:


raw_data.columns


# In[4]:


raw_data.shape


# In[5]:


raw_data.describe()


# In[6]:


raw_data.isnull().sum()


# In[7]:


val=raw_data[raw_data['Class'] == 0]
fra=raw_data[raw_data['Class'] == 1]
outlier= len(fra)/float(len(val))
print(outlier)
print('Total_FraudCases: {}'.format(len(raw_data[raw_data['Class'] == 1])))
print('Total Valid_Transactions: {}'.format(len(raw_data[raw_data['Class'] == 0])))


# In[8]:


val


# In[9]:


val.Amount.describe()


# In[10]:


fra


# In[11]:


fra.Amount.describe()


# In[23]:


sns.distplot(fra['Amount'],color='red')


# In[24]:


sns.distplot(val['Amount'],color='purple')


# In[25]:


correlation=raw_data.corr()
sns.heatmap(correlation, vmax = .8,square = True,cmap='YlGnBu')


# In[14]:


X=raw_data.drop(['Class'], axis = 1)
Y=raw_data["Class"]
print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values


# In[15]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
  
n_outliers = len(fra)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")
  
acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))
  
prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))
  
rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))
  
f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))
  


# In[29]:


labels = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix,xticklabels=labels, 
            yticklabels=labels, annot = True, fmt ="d",cmap='YlGnBu');
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




