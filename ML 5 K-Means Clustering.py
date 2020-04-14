#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# ## Importing the packages

# In[10]:


import numpy as np
import warnings
import pandas as pd
from sklearn.datasets import load_iris
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


# ## Importing the data and preprocessing

# In[11]:


#Importing traing data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

#feature columns
feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'sepal width (cm)', 'petal width (cm)']#,'Categories']

# slicing by using a two-dim dataset
X2 =X.iloc[:, 2:4]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.3, random_state = 0)

#create the a ID set for plotting the x axis
ID = pd.DataFrame(range(1, len(y_test) + 1 ,1))

#Preview the data
X2.head()


# #  Plotting the input data

# In[13]:


plt.rcParams["figure.figsize"] = (10,5)
# we only take two features

Xpetal_length=X.iloc[:, 2:3]
ypetal_width=X.iloc[:, 3:4]

#choosing the rigt columns for testing data
X_test_petal_length=X_test.iloc[:, :1]
y_test_petal_width=X_test.iloc[:, 1:2]


plt.scatter(Xpetal_length,ypetal_width)
plt.title('Iris Clustering')
plt.xlabel('Pedal length')
plt.ylabel('Pedal width')
plt.show()


# ## Fitting the model and predicting the result

# In[16]:


# Create KMeans object
model = KMeans(n_clusters=3)
#model = KMeans()


# Train Decision Tree Classifer
model = model.fit(X_train,y_train)


# Predict the response for test dataset
y_pred = model.predict(X_test)


# Reshaping the inputs to fit to the prediction
X_test_petal_length1=np.array(X_test_petal_length).reshape(-1)
y_test_petal_width1=np.array(y_test_petal_width).reshape(-1)


# ## Plotting the Clustering

# In[17]:


plt.rcParams["figure.figsize"] = (10,5)

plt.scatter((X_test_petal_length1),(y_test_petal_width1),c=y_pred, s=100, cmap='viridis',edgecolor='black',)

centers=model.cluster_centers_

#centers are defined 
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);
plt.title('Iris Clustering')
plt.xlabel('Pedal length')
plt.ylabel('Pedal width')
plt.show()

