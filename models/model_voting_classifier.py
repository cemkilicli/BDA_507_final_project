from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd


#Load clean sample data
exp_data = pd.read_csv('../exp_data/3_preprocessed_sample/clean_sample_balanced.csv', delimiter=',')
exp_data = exp_data.drop("orig_destination_distance", axis=1)

#Set column names
exp_data_col_names = list(exp_data)

#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]

class_names = ["is book", "is not book"]

scale_labels = ["srch_destination_id", "user_location_region", "user_location_city", "user_id"]

drop_labels = ["is_booking"]

for names in drop_labels:
    exp_data_data = exp_data.drop(names, axis=1)

# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)

#Prepare votig classifier "Hard"
log_clf = LogisticRegression(class_weight='balanced')
rnd_clf = RandomForestClassifier(class_weight="balanced")

print "classifiers loaded"

voting_clf = VotingClassifier(estimators=[("lr", log_clf), ("rf", rnd_clf)],
                              voting="hard")

print "voting classifiers loaded"

voting_clf.fit(features_train,labels_train)
print " classifiers fit"


for clf in (log_clf, rnd_clf, voting_clf):
    clf.fit( features_train, labels_train)
    y_pred = clf.predict(features_test)
    print( clf.__class__.__name__, "Hard Voting", accuracy_score( labels_test, y_pred))

### Initialy it is inteded to use SCV also in voting classifier but removed due to login processing time
### The implementation is based on libsvm. The fit time complexity is more than quadratic with the number
### of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.


"""
#Prepare votig classifier "soft"

log_s_clf = LogisticRegression(class_weight='balanced')
rnd_s_clf = RandomForestClassifier(class_weight="balanced")
svm_s_clf = KNeighborsClassifier(n_neighbors=100)

voting_s_clf = VotingClassifier(estimators=[("lr", log_clf),("rf", rnd_clf),("svc", svm_clf)],
                              voting="soft")

voting_s_clf.fit(features_train,labels_train)

for clf in (log_s_clf, rnd_s_clf, svm_s_clf, voting_s_clf):
    clf.fit( features_train, labels_train)
    y_pred = clf.predict(features_test)
    print( clf.__class__.__name__, "Soft Voting", accuracy_score( labels_test, y_pred))
"""


