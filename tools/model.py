from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time



exp_data = pd.read_csv('./exp_data/clean_sample/clean_sample.csv', delimiter=',')
#exp_data_labels = pd.read_csv('./exp_data/clean_sample/clean_sample_labels.csv', delimiter=',', dtype=np.int32)

print exp_data.info()

exp_data_col_names = list(exp_data)

"""
nan_index = []

for i in exp_data_col_names:
    for j in exp_data[i]:
        if np.isnan(j) == True:
            nan_index.append((i,j))

print nan_index
"""

exp_data_labels = exp_data["is_booking"]
exp_data_data = exp_data.drop("is_booking", axis=1)

print exp_data_data.info()


exp_data_labels = exp_data_labels.as_matrix()
exp_data_data = exp_data.as_matrix()
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.1, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
