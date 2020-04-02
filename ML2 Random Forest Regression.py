# # Random Forest Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")


# ## Importing the data and preprocessing

# In[31]:


# Splitting the dataset into the Training set and Test set

#Importing traing data
datasettrain = pd.read_csv('C:/Users/Harry/Desktop/Analytics/TSLA_train3.csv')
X_train = datasettrain.iloc[:, :1].values
y_train = datasettrain.iloc[:, 4:5].values

#Importing testing data
datasettest = pd.read_csv('C:/Users/Harry/Desktop/Analytics/TSLA_test3.csv')
X_test = datasettest.iloc[:, :1].values
y_test = datasettest.iloc[:, 4:5].values


#Preview the data
datasettrain.head()


# ## Choosing the model

# In[28]:


from sklearn.ensemble import RandomForestRegressor
  
 # create RF regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0,min_samples_leaf=1) 


# ## Fitting the model and predicting the result

# In[29]:


# Fitting RF Regression to the Training set
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(y_test)


# ## Plotting the graph 

# In[30]:


plt.rcParams["figure.figsize"] = (20,8)



plt.plot(X_train, y_train, color = 'red', label="Actual Train")
plt.plot(X_test,y_test, color='green', label="Actual Test")
plt.plot(X_test,y_pred, color='blue', label="Prediction")

plt.title('Tesla Stock Prices 2015-Today')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()





