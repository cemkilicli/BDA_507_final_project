import pandas as pd

# Load Data
destinations = pd.read_csv("../exp_data/destinations.csv")
test = pd.read_csv("../exp_data/test.csv")
train = pd.read_csv("../exp_data/train.csv")

# Check how much data there is
print train.shape
print test.shape
print destinations.shape

# Explore the first 5 rows of the train data:
print train.head(5)

# Explore the first 5 rows of the test data:
print train.head(5)

# Exploring hotel clusters
print train["hotel_cluster"].value_counts()

# Exploring train and test user ids
# Create a set of all the unique test user & train user ids.
test_ids = set(test.user_id.unique())
train_ids = set(train.user_id.unique())

# Check if test user id count match with train user id count
intersection_count = len(test_ids & train_ids)
print intersection_count == len(test_ids)

train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

import random

unique_users = train.user_id.unique()

sel_user_id = random.sample(unique_users,10000)
sel_train = train[train.user_id.isin(sel_user_id)]

t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]

t2 = t2[t2.is_booking == True]

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

predictions = [most_common_clusters for i in range(t2.shape[0])]

import ml_metrics as metrics
target = [[l] for l in t2["hotel_cluster"]]
metrics.mapk(target, predictions, k=5)


train.corr()["hotel_cluster"]


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]