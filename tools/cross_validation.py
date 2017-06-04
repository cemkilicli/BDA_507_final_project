import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def cross_validation_sk(df, cols_attr, col_class ):
   df['cross_cal_select'] = np.random.uniform(0, 1, len(df))
   k = 5
   step = 1.0/k

   for i in range(k):
       print('************************************************\n')

       test = df[(df['cross_cal_select'] >= i*step) & (df['cross_cal_select'] < (i+1)*step)]
       print('Length of test dataframe: ' + str(len(test)))

       train_1 = df[(df['cross_cal_select'] < i*step)]
       train_2 = df[(df['cross_cal_select'] >= (i+1)*step)]
       train = train_1.append(train_2)
       print('Length of train dataframe: ' + str(len(train)))
       print

       '''TRAIN--ACCURACY---------------------------------'''
       trainArr_All = train.as_matrix(cols_attr)  # training array
       trainRes_All = train.as_matrix(col_class)  # training results
       rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)  # 100 decision trees
       trainRes_All = trainRes_All.ravel()
       y_score_train = rf.fit(trainArr_All, trainRes_All).predict(trainArr_All)
       y_score_train = y_score_train.tolist()

       print('Train Prediction array length K = ' + str(i+1) + ' is ' + str(len(y_score_train)))
       print('Train Original test array length K = ' + str(i+1) + ' is ' + str(len(trainRes_All)) + '\n')

       count = 0
       for j in range(0, len(y_score_train)):
           if y_score_train[j] == trainRes_All[j]:
               count += 1

       print 'Train Accuracy of k = ' + str(i+1) + ' ==> ' + str(round((count * 1.0 / len(y_score_train)), 2))

       TP = 0
       TN = 0
       FP = 0
       FN = 0

       for k in range(0, len(y_score_train)):
           if y_score_train[k] == 1 and trainRes_All[k] == 1:
               TP += 1
           if y_score_train[k] == 0 and trainRes_All[k] == 0:
               TN += 1
           if y_score_train[k] == 1 and trainRes_All[k] == 0:
               FP += 1
           if y_score_train[k] == 0 and trainRes_All[k] == 1:
               FN += 1

       train_recall = TP * 100.0 / (TP + FN)
       train_precision = TP * 100.0 / (TP + FP)
       train_accuracy = (TP + TN) * 100.0 / (TP + TN + FP + FN)

       print 'TRAIN RECALL    : %' + str(round(train_recall, 2))
       print 'TRAIN PRECISION : %' + str(round(train_precision, 2))
       print 'TRAIN ACCURACY  : %' + str(round(train_accuracy, 2))
       print

       '''TEST--ACCURACY---------------------------------'''

       testArr_All = test.as_matrix(cols_attr)  # training array
       testRes_All = test.as_matrix(col_class)  # training results
       y_score_test = rf.predict(testArr_All)
       y_score_test = y_score_test.tolist()

       print('Test Prediction array length K = ' + str(i + 1) + ' is ' +  str(len(y_score_test)))
       print('Test Original array length K = ' + str(i + 1) + ' is ' + str(len(testRes_All)) + '\n')

       count = 0
       for j in range(0, len(y_score_test)):
           if y_score_test[j] == testRes_All[j]:
               count += 1

       print 'Test Accuracy of k = ' + str(i+1) + ' ==> ' + str(round((count * 1.0 / len(y_score_test)), 2))

       TP = 0
       TN = 0
       FP = 0
       FN = 0

       for l in range(0, len(y_score_test)):
           if y_score_test[l] == 1 and testRes_All[l] == 1:
               TP += 1
           if y_score_test[l] == 0 and testRes_All[l] == 0:
               TN += 1
           if y_score_test[l] == 1 and testRes_All[l] == 0:
               FP += 1
           if y_score_test[l] == 0 and testRes_All[l] == 1:
               FN += 1

       train_recall = TP * 100.0 / (TP + FN)
       train_precision = TP * 100.0 / (TP + FP)
       train_accuracy = (TP + TN) * 100.0 / (TP + TN + FP + FN)

       print 'TEST RECALL    : %' + str(round(train_recall, 2))
       print 'TEST PRECISION : %' + str(round(train_precision, 2))
       print 'TEST ACCURACY  : %' + str(round(train_accuracy, 2))
       print