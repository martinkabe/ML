# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:46:25 2019

@author: Martin
"""

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')

print(data.head())
print(data.columns)

plt.hist(data['Class'], facecolor='blue', alpha=0.75)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title(r'Distribution of counts for Class variable')
plt.figure(figsize=(2,3))
plt.show()

plt.hist(data['Class'], facecolor='blue', alpha=0.75)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title(r'Distribution of counts for Class variable')
fraction = len(data[data['Class']==1])/float(len(data[data['Class']==0]))
plt.text(0.5, 150000, "Fraudulent cases: {0}\nValid cases: {1}\nOutlier fraction is {2}".format(len(data[data['Class']==1]), 
         len(data[data['Class']==0]),
         str(round(len(data[data['Class']==1])/float(len(data[data['Class']==0]))*100, 3)) + ' %'), horizontalalignment='center', verticalalignment='center')
plt.show()

data['Class'].value_counts()[1]


## Split the data into train and test
train_data = data.sample(frac = 0.1, random_state = 1)
train_data.shape

# correlation matrix
cormat = train_data.corr()
sns.heatmap(cormat, vmax = .8, square = True)
plt.show()

## Determine dependent and independent variables
columns = train_data.columns.tolist()
columns = [c for c in columns if c not in ['Class']]
target = 'Class'
X = train_data[columns]
Y = train_data[target]

## Modelling part
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# define a random state
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=fraction),
    "K-NN": KNeighborsClassifier(n_neighbors=20),
    "Logistic Regression": LogisticRegression(random_state=state, solver='lbfgs',
                                              multi_class='multinomial'),
    "Naive Bayes": GaussianNB()
}

classifier_knn = KNeighborsClassifier(n_neighbors=20)
classifier_knn.fit(X,Y)
y_pred = classifier_knn.predict(X)
(y_pred != Y).sum()

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X, Y)
    pred_i = knn.predict(X)
    error.append(np.mean(pred_i != Y))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')


## Fit the model
n_outliers = len(data[data['Class']==1])

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    elif clf_name == "Isolation Forest":
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    elif clf_name == 'K-NN':
        clf.fit(X, Y)
        y_pred = clf.predict(X)
    elif clf_name == 'Logistic Regression':
        clf.fit(X, Y)
        y_pred = clf.predict(X)
    elif clf_name == 'Naive Bayes':
        clf.fit(X, Y)
        y_pred = clf.predict(X)
        
    
    # Reshape the prediction values to 0 for valid, 1 for fraud.
    if clf_name == "Local Outlier Factor" or clf_name == "Isolation Forest":
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
    
    




