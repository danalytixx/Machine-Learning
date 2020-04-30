# # Ensemble Learning

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree  
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ## Importing the data and preprocessing

# In[2]:


#Importing traing data
dataset = pd.read_csv('C:/Users/Harry/Desktop/Analytics/diabetes_.csv')
X = dataset.iloc[:, 1:9].values
y = dataset.iloc[:, 9:10].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Preview the data
dataset.head()


# ## 1. DT: Fitting and evaluate the model

# In[3]:


# Create Decision Tree classifer object
model = DecisionTreeClassifier(random_state=0)

# Train Decision Tree Classifer
model = model.fit(X_train,y_train)

#evaluate
model.score(X_test,y_test)


# ## 2. RF: Fitting and evaluate the model

# In[4]:


# Create RF classifer 
rf = RandomForestClassifier(n_estimators=100)

# Train 
rf = rf.fit(X_train,y_train)

rf.score(X_test,y_test)


# ## 3. Bagging: Fitting and evaluate the model

# In[5]:


# Create Bagging classifer 
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 10)

# Train 
bag = bag.fit(X_train,y_train)

bag.score(X_test,y_test)


# ## 4. Boosting: Fitting and evaluate the model

# In[6]:


# Create AdaBoost classifer 
boost = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 10, learning_rate = 1)

# Train 
boost = boost.fit(X_train,y_train)

boost.score(X_test,y_test)


# ## 5. Voting with multiple models: Fitting and evaluate the model

# In[7]:


# Voting Classifier - Multiple Model Ensemble 
lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 2 )

evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')

# Train 
evc = evc.fit(X_train,y_train)

evc.score(X_test,y_test)

