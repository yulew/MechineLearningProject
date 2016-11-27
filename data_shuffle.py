__author__ = 'Yule'
import numpy as np
import pandas as pd
from bottleneck import nanmean
DATASET_PATH_BASE = '/Users/Yule/Documents/Machine Learning/Project/data'
second=10000
ms_array = [5000, 10000, 20000, 30000]
location = ['Corner Stone', 'Lecture room', 'busloop', 'lake', 'Dinning hall', 'aq', 'gym', 'lib' ]
def data_shuffle(second):
    second_path_base = DATASET_PATH_BASE + '/' + str(second) + '/'
    for l in location[0:8]:
        data_path = second_path_base + l + '/data.csv'
        data = pd.read_csv(data_path, na_values='_',encoding="ISO-8859-1")
        #shuffle the dataframe
        data_shuffle=data.reindex(np.random.permutation(data.index))

        values = data_shuffle.values
        # Convert to numpy matrix for real.
        values = np.asmatrix(values,dtype='float64')

        # Modify NaN values (missing values).
        mean_vals = nanmean(values, axis=0)

        inds = np.where(np.isnan(values))
        values[inds] = np.take(mean_vals, inds[1])



        data_path_shuffle=second_path_base + l + '/data_shuffle.csv'
        data_shuffle.to_csv(data_path_shuffle,index=False)
data_shuffle(second)
