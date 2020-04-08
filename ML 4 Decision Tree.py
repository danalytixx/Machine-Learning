
# # Decision Tree

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pydot
import pickle
from sklearn import tree
from sklearn.tree import export_graphviz     # this is for export graphviz 
from sklearn.externals.six import StringIO   
from sklearn import metrics 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
sns.set()
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pydotplus
import pydot
from io import StringIO
import os     
os.environ["PATH"] ='C:/Program Files (x86)/Graphviz2.38/bin/'      #the path to Graphiz files


# ## Importing the data and preprocessing

# In[2]:


#Importing traing data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

#feature columns
feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'sepal width (cm)', 'petal width (cm)']#,'Categories']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#create the a ID set for plotting the x axis
ID = pd.DataFrame(range(1, len(y_test) + 1 ,1))
#Preview the data
X.head()


# ## Fitting the model and predicting the result

# In[3]:


# Create Decision Tree classifer object
classifier = DecisionTreeClassifier()

# Train Decision Tree Classifer
classifier = classifier.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)


# ## Evaluating the model 

# In[4]:


#evaluate accuracy
eval_model=metrics.accuracy_score(y_test, y_pred)

print("Accuracy: ",eval_model)


# In[5]:


# showing the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# ## Plotting the Decision Tree 

# In[15]:


dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  


# creating an Image
Image(graph.create_png())


# ## Plotting the prediction 

# In[16]:


plt.rcParams["figure.figsize"] = (20,8)


fig,axs = plt.subplots(2)
fig.suptitle('IRIS prediction')

axs[0].scatter(ID, y_test,color = 'red', label="Actual")
axs[1].scatter(ID,y_pred, color='blue', label="Prediction")



for ax in axs.flat:
    ax.set(xlabel='Points', ylabel='Iris')
    

plt.legend()
plt.show()




