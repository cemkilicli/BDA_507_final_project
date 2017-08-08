import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load Data
destinations = pd.read_csv("../exp_data/1_original_data/destinations.csv")
test = pd.read_csv("../exp_data/1_original_data/test.csv")
train = pd.read_csv("../exp_data/1_original_data/train.csv")

# Check how much data there is
print train.info()
print train.shape

# Explore the first 5 rows of the train data:
print train.head(5)

# Explore the first 5 rows of the test data:
print train.head(5)


# How Much event there is?
print "plot 1 printed"
sns.countplot(x='is_booking', data=train).set_title('Booking vs. Clicks')
plt.show()


# preferred continent destinations
print "plot 2 printed"
sns.countplot(x='hotel_continent', data=train).set_title('Preferred Continent Destinations')
plt.show()



# most of people booking are from continent 3 I guess is one of the rich continent?
print "plot 3 printed"
sns.countplot(x='posa_continent', data=train).set_title('posa_continent')
plt.show()


# putting the two above together
print "plot 4 printed"
sns.countplot(x='hotel_continent', hue='posa_continent', data=train)
plt.show()


# how many people by continent are booking from mobile
print "plot 5 printed"
sns.countplot(x='posa_continent', hue='is_mobile', data = train)
plt.show()

# Difference between user and destination country
print "plot 6 printed"
sns.distplot(train['user_location_country'], label="User country")
sns.distplot(train['hotel_country'], label="Hotel country")
plt.legend()
plt.show()

print "plot 7 printed"
sns.countplot(x='srch_ci_month', hue='is_booking', data = train)
plt.legend()
plt.show()

print "plot 8 printed"
sns.countplot(x='event_month', hue='is_booking', data = train)
plt.legend()
plt.show()

print "plot 9 printed"
sns.distplot(train['srch_ci_month'], label="Check in Month")
sns.distplot(train['event_month'], label="Search Month")
plt.legend()
plt.show()


# get number of booked nights as difference between check in and check out
print "plot 10 printed"
hotel_nights = train["room_night"].astype(float) # convert to float to avoid NA problems
train['hotel_nights'] = hotel_nights
plt.figure(figsize=(11, 9))
ax = sns.boxplot(x='hotel_continent', y='hotel_nights', data=train)
lim = ax.set(ylim=(0, 15))

plt.figure(figsize=(11, 9))
sns.countplot(x="hotel_nights", data=train)
plt.show()



# plot all columns countplots
print "plot 9 printed"
rows = train.columns.size//3 - 1
fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(12,18))
fig.tight_layout()
i = 0
j = 0
for col in train.columns:
    if j >= 3:
        j = 0
        i += 1
    # avoid to plot by date
    if train[col].dtype == np.int64:
        sns.countplot(x=col, data=train, ax=axes[i][j])
        j += 1
plt.show()

