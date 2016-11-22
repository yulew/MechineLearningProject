__author__ = 'Yule'
import numpy as np
import pandas as pd
from sklearn import linear_model, datasets


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = 0.5
#MFCC starts from 874, ends at 886.
Start=874
End=886
Target=1947
second=20000
Percentage_train=0.7
DATASET_PATH_BASE = '/Users/Yule/Documents/Machine Learning/Project/data'
ms_array = [5000, 10000, 20000, 30000]
location = ['Corner Stone', 'Lecture room', 'busloop', 'lake', 'Dinning hall', 'aq', 'gym', 'lib' ]
def logistic_regression(second,Percentage_train):
    second_path_base = DATASET_PATH_BASE + '/' + str(second) + '/'
    data_path = second_path_base + location[0] + '/data_shuffle.csv'
    data = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
    X = data.values[0:int(Percentage_train*len(data)),Start:End+1]
    T = data.values[0:int(Percentage_train*len(data)),Target]

    for l in location[1:8]:
        data_path = second_path_base + l + '/data_shuffle.csv'
        data = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
        #shuffle the dataframe
        #data=data.reindex(np.random.permutation(data.index))
        x = data.values[0:int(Percentage_train*len(data)),Start:End+1]
        t = data.values[0:int(Percentage_train*len(data)),Target]
        X=np.concatenate((X, x), axis=0)
        T=np.concatenate((T, t), axis=0)



    logreg = linear_model.LogisticRegression(C=1e5)
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, T)

    data_path = second_path_base + location[0] + '/data_shuffle.csv'
    data_test = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
    X_test = data_test.values[int(Percentage_train*len(data_test)):,Start:End+1]
    T_test = data_test.values[int(Percentage_train*len(data_test)):,Target]

    for l in location[1:8]:
        data_path = second_path_base + l + '/data_shuffle.csv'
        data_test = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")

        x = data_test.values[int(Percentage_train*len(data_test)):,Start:End+1]
        t = data_test.values[int(Percentage_train*len(data_test)):,Target]
        X_test=np.concatenate((X_test, x), axis=0)
        T_test=np.concatenate((T_test, t), axis=0)



    print("Total training accuracy:",logreg.score(X,T))
    print("Total test accuracy:",logreg.score(X_test,T_test))
    print("Accuracy for each location:")
    for l in location:
        data_path = second_path_base + l + '/data_shuffle.csv'
        data_test = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
        X_test = data_test.values[int(Percentage_train*len(data_test)):,Start:End+1]
        T_test = data_test.values[int(Percentage_train*len(data_test)):,Target]
        print(l,":",logreg.score(X_test,T_test))


logistic_regression(second,Percentage_train)