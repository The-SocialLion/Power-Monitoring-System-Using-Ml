#!/usr/bin/env python
# coding: utf-8

# # ENERGY METER USING ML

# # Loading Dataset

# In[ ]:


from pandas import read_csv


# In[ ]:


url = "Energy Meter.csv"#reading the dataset
names = ['Voltage', 'Current', 'Power', 'class']# declaring class labels
dataset = read_csv(url, names=names)


# # Summarize Dataset

# In[ ]:


print(dataset.shape)#returns shape of dataset
print(dataset.head(20))#reads the first 20 rows in the dataset
print(dataset.describe())#displays the various values /information/properties with their column names
print(dataset.groupby('class').size())# groubs the data based on class


# # Visualize Data

# In[ ]:


from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[ ]:

# plotting bar .histogram and scatter plot for various values of the data
dataset.plot(kind='bar',subplots=True,layout=(2,2))
pyplot.title('BAR PLOT')
pyplot.show()

dataset.hist()
pyplot.title('HISTOGRAM PLOT')
pyplot.show()

scatter_matrix(dataset)
pyplot.title('SCATTER PLOT')
pyplot.show()


# # Evaluating various ML Algorithm

# In[ ]:


# 6 ML Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.model_selection import train_test_split#splitting train and validation data
from sklearn.model_selection import cross_val_score#evaluation
from sklearn.model_selection import StratifiedKFold#evaluation


# In[ ]:

#load the values from the dataset
array = dataset.values
X = array[:,0:3]#all rows,3 columns (0,1,2)
y = array[:,3]# all rows ,4th column(dependent value)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)#splitting value intrain and val
#,taking 20% of data
# In[ ]:

#initialising the model
models = []#empty list and appending the models in it and evaluating it when needed
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[ ]:

#stroing the accuracy in the result 
results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)#splitting 10 group of data to the alg
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')#just like model.fit like recording the accuracy of each algorithm
    results.append(cv_results)#accuracy in results
    names.append(name)#class name 
    res.append(cv_results.mean())#mean accuracyt of each row
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.ylim(.990, .999)#plotting accuracy vs overall accuracy in graph
pyplot.bar(names, res, color ='maroon', width = 0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()

