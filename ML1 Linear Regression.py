
# # Linear Regression

# ## Importing the libraries

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split


# ## Importing the data and preprocessing

# In[62]:


# Importing the dataset
dataset = pd.read_csv('C:/Users/Harry/Desktop/Analytics/TSLA_stock.csv')
X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 4:5].values



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#View the data
dataset.head()


# ## Choosing the model

# In[63]:



from sklearn.linear_model import LinearRegression

regressor = linear_model.LinearRegression()


# ## Fitting the model and predicting the result

# In[79]:


# Fitting Simple Linear Regression to the Training set
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# ## Checking the accuracy of the model 

# In[80]:


acc = regressor.score(X_test, y_test)
print("Accuracy: " + str(acc))


# ## Plotting the graph 

# In[83]:



plt.rcParams["figure.figsize"] = (20,8)
plt.scatter(X_train, y_train, color = 'red', label="Actual")
plt.plot(X_test,y_pred, color='blue', label="Linear Regression")

plt.title('Tesla Stock Prices 2015-Today')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()







