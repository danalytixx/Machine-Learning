# # Random Forest Classifier

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


# ## Importing the data and preprocessing

# In[2]:


# Splitting the dataset into the Training set and Test set

#Importing traing data
dataset = pd.read_csv('C:/Users/Harry/Desktop/Analytics/diabetes_.csv')
X = dataset.iloc[:, 1:9].values
y = dataset.iloc[:, 9:10].values
ID=dataset.iloc[:, 0:1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#create the a ID set for plotting the x axis
ID=dataset.iloc[:len(y_test),0:1]

#Preview the data
dataset.head()


# ## Choosing the model

# In[3]:


from sklearn.ensemble import RandomForestClassifier

  
# create RF regressor object 
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0,criterion='gini' ,max_depth=3) 


# ## Fitting the model and predicting the result

# In[4]:


# Fitting RF Regression to the Training set
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)



# ## Evaluating the model 

# In[5]:


#evaluate accuracy
eval_model=classifier.score(X_train, y_train)
print("Accuracy: ",eval_model)


# In[6]:


# showing the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# ## Plotting the result 

# In[7]:


plt.rcParams["figure.figsize"] = (20,8)


fig,axs = plt.subplots(2)
fig.suptitle('Diabetes prediction')

axs[0].scatter(ID, y_test,color = 'red', label="Actual")
axs[1].scatter(ID,y_pred, color='blue', label="Prediction")



for ax in axs.flat:
    ax.set(xlabel='Points', ylabel='Outcome')
    

plt.legend()
plt.show()



