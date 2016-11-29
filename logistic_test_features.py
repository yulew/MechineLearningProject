__author__ = 'Yule'
import numpy as np
import scipy.special as sps
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = 0.5
#MFCC starts from 874, ends at 886.
Start=874
End=886
Target=1947
second=30000
Percentage_train=0.7
DATASET_PATH_BASE = '/Users/Yule/Documents/Machine Learning/Project/data'
ms_array = [5000, 10000, 20000, 30000]
location = ['Corner Stone', 'Lecture room', 'busloop', 'lake', 'Dinning hall', 'aq', 'gym', 'lib' ]
def logistic_regression(second,Percentage_train,sndFeature_list,feature):
    Column_number=list(range(Start,End+1))+sndFeature_list
    second_path_base = DATASET_PATH_BASE + '/' + str(second) + '/'
    data_path = second_path_base + location[0] + '/data_shuffle.csv'
    data = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
    X = data[Column_number].values[0:int(Percentage_train*len(data))]
    T = data.values[0:int(Percentage_train*len(data)),Target]

    for l in location[1:8]:
        data_path = second_path_base + l + '/data_shuffle.csv'
        data = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
        #shuffle the dataframe
        #data=data.reindex(np.random.permutation(data.index))
        x=data[Column_number].values[0:int(Percentage_train*len(data))]
        t = data.values[0:int(Percentage_train*len(data)),Target]
        X=np.concatenate((X, x), axis=0)
        T=np.concatenate((T, t), axis=0)



    logreg = linear_model.LogisticRegression(C=1e5)
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, T)
    err_train=logreg.score(X,T)
    data_path = second_path_base + location[0] + '/data_shuffle.csv'
    data_test = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
    X_test = data_test[Column_number].values[int(Percentage_train*len(data_test)):]

    T_test = data_test.values[int(Percentage_train*len(data_test)):,Target]
    for l in location[1:8]:
        data_path = second_path_base + l + '/data_shuffle.csv'
        data_test = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")

        x = data_test[Column_number].values[int(Percentage_train*len(data_test)):]

        t = data_test.values[int(Percentage_train*len(data_test)):,Target]
        X_test=np.concatenate((X_test, x), axis=0)
        T_test=np.concatenate((T_test, t), axis=0)


##########################"l2"regularization#####################
    logreg = linear_model.LogisticRegression(C=1e5,penalty='l2')
    # we create an instance of Neighbours Classifier and fit the data.
    T_pred=logreg.fit(X, T).predict(X_test)
    cnf_matrix = confusion_matrix(T_test, T_pred)
    cla_rep=classification_report(T_test, T_pred)

    print("Confusion Matrix:\n%s" %cnf_matrix)
    print("Classfication report:\n%s" %cla_rep)
    print("The training error is %s" %err_train)
#########################cross validation##########################
    scores = cross_val_score(logreg, X, T, cv=10)
    ###########################################
    print("Cross Validation:\n%s" %scores)

    try:
        text_file.write("The name of the group of features (+MFCC): %s (column %s-%s)\n" %(feature,sndFeature_list[0],sndFeature_list[-1]))
    except IndexError:
        text_file.write("The name of the group of features (+MFCC): %s \n" %(feature))

    text_file.write("Confusion Matrix:\n%s\n" %cnf_matrix)
    text_file.write("Classfication report:\n%s" %cla_rep)
    text_file.write("The training accuracy is %s\n" %err_train)
    text_file.write("Cross Validation:\n%s.\n" %scores)
    text_file.write("############################################################################\n")




features_dict={"lsf":list(range(0,10)),"ac":list(range(10,59)),"am":list(range(59,67)),"obsir":list(range(67,76)),"obsi":list(range(76,86)),"sroll":list(range(86,87)),"ess":list(range(87,91)),"derivate":list(range(91,92)),"mels":list(range(92,132)),"sflux":list(range(132,133)),"env":list(range(133,297)),"energy":list(range(297,298)),"lx":list(range(298,322)),"sfpb":list(range(322,345)),"psh":list(range(345,346)),"mags":list(range(346,859)),"sss":list(range(859,863)),"hi":list(range(863,873)),"psp":list(range(873,874)),"scfp":list(range(887,910)),"sv":list(range(910,911)),"tss":list(range(911,915)),"si":list(range(915,916)),"stati":list(range(916,918)),"cdod":list(range(918,919)),"lpc":list(range(919,921)),"sf":list(range(921,922)),"sd":list(range(922,923)),"frame":list(range(923,1946)),"mfcc_only":[]}


text_file = open("FeaturesPlusMFCC_scores_"+str(second)+".txt", "w")

for feature in list(features_dict.keys()):

    sndFeature_list=features_dict[feature]

    logistic_regression(second,Percentage_train,sndFeature_list,feature)
text_file.close()