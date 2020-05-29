import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.graphics.gofplots import qqplot
from yellowbrick.regressor import ResidualsPlot

PATH = "/Users/sherryguo/code_school/datasets/"
CSV_DATA = "carInsurance.csv"
df = pd.read_csv(PATH + CSV_DATA, encoding="ISO-8859-1", sep=',')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(df.head(2))
# print(df.describe())

# --------------------------------------------------
#                       EDA - Heatmap
# --------------------------------------------------

corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns)

# plt.show()


# --------------------------------------------------
#                  DATA PREP
# --------------------------------------------------


# -------------- Imputing

def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if(measureType=="median"):
        imputedValue = df[colName].median()
    elif(measureType=="mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    return df


df = convertNAcellsToNum('CAR_AGE', df, 'mean')
df = convertNAcellsToNum('AGE', df, 'mean')
df = convertNAcellsToNum('YOJ', df, 'mean')
print(df.describe())

# ------ ( BINNING )
# bin CLAIM_FLAG by CAR_AGE
df["CarAgeBin"] = pd.cut(x=df["imp_CAR_AGE"], bins=4)

tempDf = df[['CarAgeBin', 'CLAIM_FLAG']]  # Isolate columns
# Get dummies
dummyDf = pd.get_dummies(tempDf, columns=['CarAgeBin', 'CLAIM_FLAG'])
df = pd.concat(([df, dummyDf]), axis=1)     # Join dummy df with original
# print("printing binned: ", df.head().transpose())


# ------- ( DUMMY VARS )
df = pd.get_dummies(df, columns=['CAR_USE', 'REVOKED', 'CAR_TYPE'])
print(df.describe().transpose())


# ----------------------------------------------------------
#                   Building Model
# ----------------------------------------------------------

df = pd.DataFrame(df, columns=[
    'imp_YOJ',
    'HOMEKIDS',
    'imp_AGE',
    'TRAVTIME',
    'CLM_FREQ',
    'MVR_PTS',
    'imp_CAR_AGE',
    'CarAgeBin_(-3.031, 4.75]',
    'CarAgeBin_(4.75, 12.5]',
    'CarAgeBin_(12.5, 20.25]',
    'CarAgeBin_(20.25, 28.0]',
    'CAR_USE_Commercial',
    'CAR_USE_Private',
    'REVOKED_No',
    'REVOKED_Yes',
    'CAR_TYPE_Minivan',
    'CAR_TYPE_Panel Truck',
    'CAR_TYPE_Pickup',
    'CAR_TYPE_Sports Car',
    'CAR_TYPE_Van',
    'CAR_TYPE_z_SUV',
    'CLAIM_FLAG'
    ])

# REFmodel -- all desired variables
X = df[[
    'HOMEKIDS',
    'imp_AGE',
    'TRAVTIME',
    'CLM_FREQ',
    'MVR_PTS',
    'imp_CAR_AGE',
    'CarAgeBin_(-3.031, 4.75]',
    'CarAgeBin_(4.75, 12.5]',
    'CarAgeBin_(12.5, 20.25]',
    'CarAgeBin_(20.25, 28.0]',
    'CAR_USE_Commercial',
    'CAR_USE_Private',
    'REVOKED_No',
    'REVOKED_Yes',
    'CAR_TYPE_Minivan',
    'CAR_TYPE_Panel Truck',
    'CAR_TYPE_Pickup',
    'CAR_TYPE_Sports Car',
    'CAR_TYPE_Van',
    'CAR_TYPE_z_SUV'
    ]]

y = df['CLAIM_FLAG']


# ---------------- MODEL 2: promising values from heatmap

# X = df[[
#     'CLM_FREQ',
#     'MVR_PTS'
#     ]]

# ---------------- MODEL 3: recursive feature selection/binned/dummy

# X = df[[
#     'HOMEKIDS',
#     'imp_YOJ',
#     'TRAVTIME',
#     'REVOKED_No',
#     'CAR_TYPE_Minivan',
#     'CarAgeBin_(-3.031, 4.75]',
#     'CarAgeBin_(4.75, 12.5]',
#     'CarAgeBin_(12.5, 20.25]',
#     'CarAgeBin_(20.25, 28.0]',
#     'CAR_USE_Commercial',
#     'CAR_USE_Private',
#     'REVOKED_No',
#     'REVOKED_Yes',
#     'CAR_TYPE_Minivan',
#     'CAR_TYPE_Panel Truck',
#     'CAR_TYPE_Pickup',
#     'CAR_TYPE_Sports Car',
#     'CAR_TYPE_Van',
#     'CAR_TYPE_z_SUV'
#     ]]

# ------------------ MODEL 4: myFeatures

# X = df[[
#     'HOMEKIDS',
#     'imp_AGE',
#     'TRAVTIME',
#     'CLM_FREQ',
#     'MVR_PTS',
#     'imp_CAR_AGE',
#     'CarAgeBin_(4.75, 12.5]',
#     'CarAgeBin_(12.5, 20.25]',
#     'CAR_USE_Commercial',
#     'CAR_USE_Private',
#     'REVOKED_No',
#     'CAR_TYPE_Minivan'
#     ]]


# ============================================================

from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import roc_curve
from sklearn.metrics           import roc_auc_score


# Adding an intercept !!
X = sm.add_constant(X)

# --------- applying MinMax scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# sc_x    = MinMaxScaler()

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()
# X_Scale = sc_x.fit_transform(X)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

# Split data.
# X_train, X_test, y_train, y_test = train_test_split(
#     X_Scale, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                   solver='liblinear')

# Fit the model.
logisticModel.fit(X_train, y_train)


# --------- ( RECURSIVE FEATUR SELECTION )
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# print("Please wait for automated feature selection...")
# logreg = LogisticRegression(max_iter=200)
# rfe = RFE(logreg, 5) # Select top 5 features.
# rfe = rfe.fit(X_Scale, y)
# print("Feature selection is complete.")
# print(rfe.support_)
# print(rfe.ranking_)
#
# def getSelectedColumns(ranking):
#     # Extract selected indices from ranking.
#     indices = []
#     for i in range(0, len(ranking)):
#         if (ranking[i] == 1):
#             indices.append(i)
#     # Build list of selected column names.
#     counter = 0
#     selectedColumns = []
#     for col in X:
#         if (counter in indices):
#             selectedColumns.append(col)
#         counter += 1
#     return selectedColumns
# selectedPredictorNames = getSelectedColumns(rfe.ranking_)
#
# # Show selected names from RFE.
# print("\n*** Selected names: ")
# for i in range(0, len(selectedPredictorNames)):
#     print(selectedPredictorNames[i])
#
# # ==========================================================================
# # >> Recursion selected: HOMEKIDS, YOJ, TRAVTIME, CAR_TYPE_Minivan, REVOKED_No
# # ==========================================================================
#
# # ---- CHI^2 -----
#
# # Show chi-square scores for each feature.
# # There is 1 degree freedom since 1 predictor during feature evaluation.
# # Generally, >=3.8 is good)
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import SelectKBest
# test      = SelectKBest(score_func=chi2, k=20)
# XScaled   = MinMaxScaler().fit_transform(X)
# chiScores = test.fit(XScaled, y)  # Summarize scores
# np.set_printoptions(precision=3)
#
# # Search here for insignificant features.
# print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

# ==========================================================================
# >> Insignificant from X^2: CAR_TYPE_z_SUV
# ==========================================================================


# =========================================
#     !!  TESTING MODEL ACCURACY !!
# =========================================
# ------ k-folds ------
from sklearn.model_selection import KFold
# data sample

# count = 0
# # prepare cross validation with three folds and 1 as a random seed.
# kfold = KFold(3, True, 1)
# for train, test in kfold.split(df):
#     X_train = X.iloc[train,:] # Gets all rows with train indexes.
#     y_train = y.iloc[train,:]
#     X_test =  X.iloc[test,:]
#     y_test =  y.iloc[test,:]
#
#     # Perform logistic regression.
#     logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
#                                        solver='liblinear')
#     # Fit the model.
#     logisticModel.fit(X_train, np.ravel(y_train))
#
#     y_pred = logisticModel.predict(X_test)
#
#     # Show confusion matrix and accuracy scores.
#     cm = pd.crosstab(np.ravel(y_test), y_pred,
#                                    rownames=['Actual'],
#                                    colnames=['Predicted'])
#     count += 1
#     print("\n>> K-fold: " + str(count))
#     print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
#     from sklearn.metrics import classification_report
#
#     print(classification_report(y_test, y_pred))
#
#     from sklearn.metrics import average_precision_score
#
#     average_precision = average_precision_score(y_test, y_pred)
#
#     print('Average precision-recall score: {0:0.2f}'.format(
#         average_precision))
#
#     print("\nConfusion Matrix")
#     print(cm)

# -------------------------------------------------------------
# Calculates accuracy and shows confusion matrix with custom
# cutoff probability for a response of 1.
# -------------------------------------------------------------
# -------------------------------------------------------------
# Calculates accuracy and shows confusion matrix with custom
# cutoff probability for a response of 1.
# -------------------------------------------------------------
# -------------------------------------------------------------
# Calculates accuracy and shows confusion matrix with custom
# cutoff probability for a response of 1.
# -------------------------------------------------------------
# def predictUsingAlternateCutoff(y_test, probIsOne, cutoff, size):
#     cm = np.zeros((size, size))  # Create empty 5x5 matrix of 0’s
#     predictions = []
#     correctCount = 0
#
#     for i in range(0, len(probIsOne)):
#         actualValue_Row = y_test.values[i]
#         predictValue_Col = 0  # Predicted value is 0 by default.
#         prediction = 0
#
#         # Check if probability of zero is high.
#         if (probIsOne[i] >= cutoff):
#             predictValue_Col = 1
#             prediction = 1
#
#         cm[actualValue_Row][predictValue_Col] += 1
#         predictions.append(prediction)
#
#         # Increase correct item count when prediction matches actual response.
#         if (actualValue_Row == predictValue_Col):
#             correctCount += 1
#
#     accuracy = correctCount / len(probIsOne)
#     print("\n*** Accuracy with CUTOFF of " + str(cutoff) + ": " + str(accuracy))
#     print("\nConfusion Matrix: actual (row) vs predicted (col)")
#     print(cm)
#     return predictions, cm
# # Extract probabilities.
# y_proba = logisticModel.predict_proba(X_test)
# SIZE    = 2  # for 2x2 matrix
# CUT_OFF = 0.48
# updatedPredict = predictUsingAlternateCutoff(y_test, y_proba[:, 1], CUT_OFF, SIZE)
#
# def getPositiveRates(cm):
#     tn  = cm[0][0]
#     fn  = cm[1][0]
#     tp  = cm[1][1]
#     fp  = cm[0][1]
#     tpr = tp / (tp + fn)  # tp over all possible positives
#     fpr = fp / (fp + tn)  # fp over all possible negatives
#     return fpr, tpr
#
# #----------------------------------------
# # Plot the ROC
# #----------------------------------------
# def graphROC(y_test, y_proba, decrement):
#     tprList = []
#     fprList = []
#     SIZE = 2
#
#     # Start with cut-off of 1 and decrement down to 0.
#     # In other words, use from most likely true positive candidates first and then
#     # to most uncertain candidates.
#     cutOff = 1
#     while cutOff >-decrement:
#         updatedPredictions, cm = predictUsingAlternateCutoff(y_test,
#                                         y_proba, cutOff, SIZE)
#         fpr, tpr = getPositiveRates(cm)
#         tprList.append(tpr)
#         fprList.append(fpr)
#         cutOff -= decrement
#         print("tpr: " + str(tpr))
#         print("fpr: " + str(fpr))
#         print("cutoff: " + str(cutOff))
#
#     # Plot the list.
#     plt.plot(fprList, tprList, marker='.', label='Logistic')
#     plt.plot([0, 1], [0, 1], '--', label='No Skill')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.show()
#
#
# DECREMENT = 0.025
# graphROC(y_test, y_proba[:, 1], DECREMENT)
#
# # calculate scores
# auc = roc_auc_score(y_test, y_proba[:, 1],)
# print('Logistic: ROC AUC=%.3f' % (auc))
#
# # calculate roc curves
# # lr_fpr, lr_tpr, _ = roc_curve(y_test, y_proba[:, 1])
# # plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# # plt.plot([0,1], [0,1], '--', label='No Skill')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.legend()
# # plt.show()
#
# clf = LogisticRegression(
#     random_state=0, multi_class='multinomial', solver='newton-cg')
#
# clf.fit(X_train, y_train)
# predicted_probas = clf.predict_proba(X_test)
# y_pred           = clf.predict(X_test);
#
# # The magic happens here
# import matplotlib.pyplot as plt
# import scikitplot as skplt
# skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)
# skplt.metrics.plot_lift_curve(y_test, predicted_probas)
# plt.show()

# the culmultive gains plot was done in Spyder3, as it did not work in Pycharm

# Build and evaluate model.
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())
print(f"**RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}")
print(predictions)






