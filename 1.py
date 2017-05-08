
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from preprocess_tools import strip
from preprocess_tools import date_subtract
from preprocess_tools import strip_other
from datetime import datetime
from preprocess_tools import adult_per_room


exp_data = pd.read_csv('./exp_data/sample_mini.csv', delimiter=',')

exp_data["srch_ci"].replace("", np.nan, inplace=True)
exp_data["srch_co"].replace("", np.nan, inplace=True)
exp_data = exp_data.dropna(subset=["srch_ci"], how="all")
exp_data = exp_data.dropna(subset=["srch_co"], how="all")


#Creating vatiables
exp_data["event_date"] = exp_data.apply(lambda row: strip(row["date_time"], "date"), axis=1)
exp_data["event_time"] = exp_data.apply(lambda row: strip(row["date_time"], "time"), axis=1)
exp_data["room_night"] = exp_data.apply(lambda row: date_subtract(row["srch_ci"], row["srch_co"]), axis=1)

exp_data["adult_per_room"] = exp_data.apply.map(lambda x: adult_per_room(exp_data["srch_adult_cnt"][x], exp_data["srch_rm_cnt"][x]))


exp_data.head()
"""
exp_data["adult_per_room"] = exp_data.apply(lambda row: adult_per_room(row["srch_adult_cnt"], row["srch_rm_cnt"]), axis=0)

"""



print exp_data.head()


"""# Create tabel variable to pass train_test_split
exp_data_labels = exp_data.is_booking.as_matrix()

# Remove column is_booking and orig_destination_distance from data frame
data_train = exp_data.drop("is_booking",axis=1)
data_train_1 = data_train.drop("orig_destination_distance",axis=1)"""




"""features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.1, random_state=42)

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
print accuracy_score(pred, labels_test)"""