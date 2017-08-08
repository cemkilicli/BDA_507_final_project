from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


#Load clean sample data
exp_data = pd.read_csv('../exp_data/3_preprocessed_sample/clean_sample_balanced.csv', delimiter=',')
exp_data = exp_data.drop("orig_destination_distance", axis=1)


#Print data frame information
print exp_data.info()
print exp_data.shape

#Set column names
exp_data_col_names = list(exp_data)


#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]

from sklearn.preprocessing import MinMaxScaler
import numpy as np

class_names = ["is book", "is not book"]

scale_labels = ["srch_destination_id", "user_location_region", "user_location_city", "user_id"]


def scaler_MinMax(scale_labels):
    scaler = MinMaxScaler()
    for name in scale_labels:
        data = np.array(exp_data[name])
        scaled_values = scaler.fit_transform(data)
        exp_data[name] = scaled_values


scaler_MinMax(scale_labels)


drop_labels = ["is_booking"]

for names in drop_labels:
    exp_data_data = exp_data.drop(names, axis=1)


# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)


from sklearn.ensemble import RandomForestClassifier

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=50, class_weight="balanced")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
clf_probs = clf.predict_proba(features_test)
score = log_loss(labels_test, clf_probs)

print score


"""
# Print Confusion Matrix
cnf_matrix = confusion_matrix(labels_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization - KNN')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix - KNN')
plt.show()

"""
# Print Accuracy Score
print "Accuracy is", accuracy_score(labels_test,pred)
print "The number of correct predictions is", accuracy_score(labels_test, pred, normalize=False)
print "Total sample used is", len(pred)  # number of all of the predictions
