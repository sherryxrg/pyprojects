import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
import seaborn as sns

from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression

from    sklearn.preprocessing   import StandardScaler
from sklearn.decomposition import PCA
from    sklearn.model_selection import train_test_split
from    sklearn.linear_model    import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, \
    confusion_matrix, average_precision_score

PATH = "/Users/sherryguo/code_school/datasets/"
CSV_DATA = "heart_original.xlsx"
df = pd.read_csv(PATH + CSV_DATA, encoding="ISO-8859-1", sep=',')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# =============
#  summaries:
# =============

# print("\n>> Preliminary Summary: ")
print(df.head(2))
# print()
# print("Shape:", df.shape)
# print("n = 303, characteristics = 14, 13 independent\n")
# df.info()
# print(df.describe())


# ====== EDA: Heatmap and Scatter Matrix ===========

# # scatter matrix of continous data
# dfSub = pd.DataFrame(df, columns=['age', 'trestbps', 'chol', 'thalach',
#                                   'oldpeak', 'target'])
#
# scatter_matrix(dfSub, figsize=(12, 12))
# plt.show()
#
# # heatmap of continuos data
# corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns,
#             yticklabels=corr.columns)
#
# plt.show()

# ======== EDA: Factor Analysis ===================

# bartlett's test if identity matrix
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value, p_value=calculate_bartlett_sphericity(df)

print("\nBartlett's test chi-square value: ")
print(chi_square_value)

print("\nBartlett's test p-value: ")
print(p_value)  # super significant p-value

# KMO test
# the more suited your data is to Factor Analysis. Factor analysis is suitable
# for scores of 0.6 (and sometimes 0.5) and above.
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
print("\nKaiser-Meyer-Olkin (KMO) Test: Suitability of data for factor analysis.")
print(kmo_model)

# Create factor analysis and examine loading vectors and Eigenvalues.
fa = FactorAnalyzer(rotation=None)
fa.fit(df)
print("\nFactors:")
print(fa.loadings_)

# prepping data to export to excel...
loading_arr = fa.loadings_
print(type(loading_arr))
data = loading_arr.tolist()
print(data)
dfexcel = pd.DataFrame.from_records(data, columns=['Factor 1',
                                                   'Factor 2',
                                                   'Factor 3'])
print(dfexcel)
dfexcel.to_excel("factors_results.xlsx")

ev, v = fa.get_eigenvalues()
print("\nEignenvalues:")
print(ev)

# ========================================
#       Handling Multicollinearity
# ========================================
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate and show VIF Scores for original data.
vif = pd.DataFrame()
X = df
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in
                     range(X.shape[1])]
vif["features"] = X.columns
print("\nOriginal VIF Scores")

print(vif)
# vif.to_excel("vif_scores.xlsx")


# ========================================
#               PCA
# ========================================
X = pd.DataFrame(df, columns=['age', 'trestbps', 'chol', 'thalach',
                              'oldpeak'])
y = pd.DataFrame(df, columns=['target'])
print(X.head())
print(y.head())

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y,
                                            test_size=0.20)

pca = PCA(.8)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test = pca.transform(X_test)[:,:7]

print("\nPrincipal Components")
print(pca.components_)

pca_list = pca.components_.tolist()
print(pca_list)
dfexcel = pd.DataFrame.from_records(pca_list)
print(dfexcel)
dfexcel.to_excel("pca_run.xlsx")

print("\nExplained variance: ")
print(pca.explained_variance_)

# Train regression model on training data
model = LinearRegression()
model.fit(X_reduced_train[:,:7], y_train)

# Prediction with test data
pred = model.predict(X_reduced_test)
print("predictions:")
# print(pred)

# Show stats about the regression.
mse = mean_squared_error(y_test, pred)
RMSE = np.sqrt(mse)
print("\nRMSE: " + str(RMSE))

print("\nModel Coefficients")
print(model.coef_)

print("\nModel Intercept")
print(model.intercept_)

from sklearn.metrics import r2_score
print("\nr2_score",r2_score(y_test, pred))

# For each principal component, calculate the VIF and save in dataframe
vif = pd.DataFrame()

# Show the VIF score for the principal components.
print()
vif["VIF Factor"] = [variance_inflation_factor(X_reduced_train, i)
                     for i in range(X_reduced_train.shape[1])]
print(vif)

# ======================================================================
#               Model - Logistic Regression
# ======================================================================

# Separate into x and y values.
X = df[['age', 'trestbps', 'thalach', 'slope']]
y = df['target']

print(X.head())
print(y.head())

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)  # removed randomstate to randomize

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, random_state=1,
                                   solver='liblinear')
logisticModel.fit(X_train, y_train)
y_pred = logisticModel.predict(X_test)

# Show model coefficients and intercept.
print("\nModel Coefficients: ")
print("\nIntercept: ")
print(logisticModel.intercept_)

print(logisticModel.coef_)

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

mse = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(mse)
print("\nRMSE: " + str(RMSE))

from sklearn.metrics import r2_score
print("\nr2_score",r2_score(y_test, y_pred))
print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print('\nPrecision: ', metrics.precision_score(y_test, y_pred))
print('\nRecall: ', metrics.recall_score(y_test, y_pred))
print('\nF1: ', metrics.f1_score(y_test, y_pred))


# ======================================================================
#               Model - PCA Mixed Logistic Regression
#
#      (Note: please only uncomment one model section when running,
#      or there may be an error generating the second confusion matrix)
# ======================================================================
#
# logistic_cols = ['sex', 'restecg', 'exang', 'slope', 'thal']
# pca_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
# all_cols = logistic_cols + pca_cols
# print(all_cols)
#
# # Separate into x and y values.
# X = df[all_cols]
# y = df['target']
#
# print(X.head())
# print(y.head())
# print()
#
# # Split x and y into test and training.
# X_train,X_test,y_train,y_test = train_test_split(
#         X, y, test_size=0.25)  # removed random state
#
# # Scale data.
# X_trainPCA = StandardScaler().fit_transform(X_train[pca_cols])
# X_testPCA  = StandardScaler().fit_transform(X_test[pca_cols])
#
# # Determine how many components are needed.
# from sklearn.decomposition import PCA
# pcaTest = PCA(.9)
# pcaTest.fit(X_trainPCA)
#
# # Using 2 components
# print("components:")
# print(pcaTest.components_)
# print(pcaTest.explained_variance_)
#
# # Extract 2 PCA components and fit for transformation.
# pca = PCA(n_components=1)
# pca.fit(X_trainPCA)
#
# # Use PCA components to transform 'all' data to
# # effectively generate our manufactured column.
# X_trainScaled = pca.transform(X_trainPCA)
# X_testScaled = pca.transform(X_testPCA)
#
# # Perform logistic regression with PCA output and
# # other selected variables.
# logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=1000)
# logisticRegr.fit(X_trainScaled, y_train)
# pca_predictionsTest = logisticRegr.predict(X_testScaled)
# pca_predictionsTrain = logisticRegr.predict(X_trainScaled)
#
# # cm = confusion_matrix(y_test, pca_predictionsTest)
# # print(cm)
#
# #-------------------------------------------------------
# # Now build logistic model with mix of categorical variables
# # and PCA results.
# #-------------------------------------------------------
# X_trainLogistic = X_train[logistic_cols]
#
# # Avoid 'copy' warning.
# pd.set_option('mode.chained_assignment', None)
# X_trainLogistic['PCA_result'] = pca_predictionsTrain
#
# X_testLogistic = X_test[logistic_cols]
# X_testLogistic['PCA_result'] = pca_predictionsTest
#
# logisticRegr2 = LogisticRegression(solver = 'lbfgs', max_iter=1000)
# logisticRegr2.fit(X_trainLogistic, y_train)
#
# # Now make predictions using PCA result plus other variables.
# predictions2 = logisticRegr2.predict(X_testLogistic)
# cm2 = confusion_matrix(y_test, predictions2)
#
# # Show confusion matrix and accuracy scores.
# confusion_matrix = pd.crosstab(y_test, pca_predictionsTest,
#                                rownames=['Actual'],
#                                colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)
#
# print('\nAccuracy: ', metrics.accuracy_score(y_test, pca_predictionsTest))
# print("\nConfusion Matrix")
# print(confusion_matrix)
#
# # evaluating:
# # Show stats about the regression.
# mse = mean_squared_error(y_test, pca_predictionsTest)
# RMSE = np.sqrt(mse)
# print("\nRMSE: " + str(RMSE))
#
# from sklearn.metrics import r2_score
# print("\nr2_score",r2_score(y_test, pca_predictionsTest))
#
# print('\nAccuracy: ', metrics.accuracy_score(y_test, pca_predictionsTest))
# print('\nPrecision: ', metrics.precision_score(y_test, pca_predictionsTest))
# print('\nRecall: ', metrics.recall_score(y_test, pca_predictionsTest))
# print('\nF1: ', metrics.f1_score(y_test, pca_predictionsTest))





















