
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import scipy as sp
from time import time


exp_data = np.genfromtxt('./exp_data/sample1.csv", delimiter=',',names= True)


is_booking = "is_booking"

column_names = list(exp_data.dtype.names)
column_names.remove(is_booking)

print column_names

exp_data_labels = np.array(exp_data["is_booking"])
exp_data_data = np.array(exp_data[column_names])

print exp_data_data




features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.1, random_state=42)

print labels_train

from sklearn import tree

clf = tree.DecisionTreeClassifier()

features_test = preprocessing.LabelEncoder()
features_train = preprocessing.LabelEncoder()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)