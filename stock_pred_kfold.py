import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


df = pd.read_csv("./resources/2018.csv")
df = df.dropna()
df = df.reset_index(drop=True)
df.head()

nyse_previous_close = 12412.07
nyse_one_year_close = 11812.20

nasdaq_previous_close = 7712.9502
nasdaq_one_year_close = 6233.9502

nasdaq_begin_index = 977

nyse_performance = math.log(nyse_previous_close/nyse_one_year_close)
nasdaq_performance = math.log(nasdaq_previous_close/nasdaq_one_year_close)

# print(nyse_performance)

df = df.infer_objects()
df['last_price'] = df['last_price'].astype('float')

#print(df.dtypes)

df['performance'] = np.log(df['last_price'] /df['previous_year_price'])
df['status'] = 0
nyse_performance = (df.loc[range(0,nasdaq_begin_index),'performance']>nyse_performance).astype('int')
nasdaq_performance = (df.loc[range(nasdaq_begin_index,len(df)),'performance']>nasdaq_performance).astype('int')
df['status'] = nyse_performance
df.loc[nasdaq_begin_index:len(df),'status'] = nasdaq_performance
df['status'] = df['status'].astype('int')


X = df[['cash_ratio','return_to_equity','price_to_book',
        'pe','short_interest_ratio','debt_to_equity','eps']]
y = df['status']

X = np.array(X)
y = np.array(y)

def F1_score(confusion_matrix):
    v = np.array(confusion_matrix)
    TN, FP, FN, TP = v.flatten()
    prec = TP/ (TP + FP)
    recall = TP/ (TP + FN)
    F1 = 2*(prec*recall)/(prec+recall)
    return F1


def confusion_matrix_for_reg(predict, real):
    ## when running regression, we expect the predict performance would not consists only 1 or 0.
    ## We round the predicted performance to the first decimal
    predict = np.array(np.round(predict))

    ## Since the underlying model is a regression, we also limit the prediction between 0 and 1
    predict = predict.clip(min=0, max=1)

    real = np.array(real)

    ## use AND gate to check for TP
    TP = np.sum(np.logical_and(predict, real))
    ## FP = sum of all 1s in predict - TP
    FP = np.sum(predict == 1) - TP

    ## first flip predict and real 1 and 0s, and use the same AND gate to check for TN
    TN = np.sum(np.logical_and(np.logical_not(predict), np.logical_not(real)))
    FN = np.sum(predict == 0) - TN

    conf_matrix = [[TN, FP], [FN, TP]]

    return conf_matrix


def model_OLS(data_X_train, data_y_train, data_X_test):
    linreg = linear_model.LinearRegression(fit_intercept=True, normalize=True)  # can change with or without intercept
    linreg.fit(data_X_train, data_y_train)
    linreg.get_params()
    y_train_from_OLS = linreg.predict(data_X_train)
    y_test_from_OLS = linreg.predict(data_X_test)

    return y_train_from_OLS, y_test_from_OLS


def model_Lassi(data_X_train, data_y_train, a):
    Lassi = Lasso(alpha=a, tol=1e-5, normalize=True)
    Lassi.fit(data_X_train, data_y_train)

    y_train_from_LASSO = Lassi.predict(data_X_train)

    return y_train_from_LASSO

def model_Rachel(data_X_train, data_y_train, a):
    Rachel = Ridge(alpha=a, tol=1e-5, normalize=True)
    Rachel.fit(data_X_train, data_y_train)
    y_train_from_Ridge = Rachel.predict(data_X_train)

    return y_train_from_Ridge

superlist =[]

kf = KFold(n_splits=10)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=55, shuffle=True, stratify=y)

    ## TODO RUN Random/ Monkey
    np.random.seed(2018)
    y_train_from_random = np.random.uniform(1,0,len(y_train))

    y_test_from_random = np.random.uniform(1,0,len(y_test))

    ## TODO RUN OLS

    y_train_from_OLS, y_test_from_OLS = model_OLS(X_train, y_train, X_test)

    conf_matrix_train_from_OLS = confusion_matrix_for_reg(y_train_from_OLS, y_train)

    F1_score_train_from_OLS = F1_score(conf_matrix_train_from_OLS)


    conf_matrix_test_from_OLS = confusion_matrix_for_reg(y_test_from_OLS, y_test)


    F1_score_test_from_OLS = F1_score(conf_matrix_test_from_OLS)

    ## TODO Optimizing LASSO and Ridge Lambda

    ## initialising

    # range of lambda we are testing
    x_axis_interval = np.arange(0, 1, 1e-3)

    # lists of F1 scores

    F1_train_LASSO = []
    F1_train_Ridge = []
    F1_train_RidgeC =[]

    ## finding optimal lambda for the least amount of False Positive
    for a in x_axis_interval:
        y_train_from_LASSO = model_Lassi(X_train, y_train, a)

        ## Calculate the F1 score and append the score list

        F1_train_from_LASSO = F1_score(confusion_matrix_for_reg(y_train_from_LASSO, y_train))
        F1_train_LASSO.append(F1_train_from_LASSO)


        y_train_from_Ridge = model_Rachel(X_train, y_train, a)

        F1_train_from_Ridge = F1_score(confusion_matrix_for_reg(y_train_from_Ridge, y_train))
        F1_train_Ridge.append(F1_train_from_Ridge)



    ## TODO plot lambda across
    # Plotting the precisions across lambdas
    plt.scatter(x_axis_interval,F1_train_LASSO, label="LASSO train", s=12)
    plt.scatter(x_axis_interval,F1_train_Ridge, label="Ridge train", s=8)


    plt.xlabel('tuning parameter lambda')
    plt.ylabel('F1 score of train data fitting')
    plt.legend()
    # plt.show()

    ## TODO optimal model with test set

    a_range = x_axis_interval

    ## Only optimize for minimum FP, long only fund
    F1_train_from_LASSO = np.array(F1_train_from_LASSO)
    F1_train_from_Ridge = np.array(F1_train_from_Ridge)

    opt_lambda_LASSO = a_range[np.argmax(F1_train_from_LASSO)]
    opt_lambda_Ridge = a_range[np.argmax(F1_train_from_Ridge)]


    ## predicting test data set with optimal lambda/ alpha, returning confusion matrix

    Lassi_opt = Lasso(alpha=opt_lambda_LASSO, tol=1e-5)
    Lassi_opt.fit(X_train, y_train)
    y_test_from_Lassi = Lassi_opt.predict(X_test)

    F1_test_from_Lassi = F1_score(confusion_matrix_for_reg(y_test_from_Lassi,y_test))


    Rachel_opt = Ridge(alpha=opt_lambda_Ridge, tol=1e-5)
    Rachel_opt.fit(X_train, y_train)
    y_test_from_Rachel = Rachel_opt.predict(X_test)

    F1_test_from_Rachel = F1_score(confusion_matrix_for_reg(y_test_from_Rachel, y_test))


    ## TODO print test set F1 score

    ## TODO Performance metric, calculating F1 score

    ## first get F1 score of randomly picking

    F1_train_from_random = F1_score(confusion_matrix_for_reg(y_train_from_random, y_train))
    F1_test_from_random = F1_score(confusion_matrix_for_reg(y_test_from_random, y_test))

    F1_score_list = [F1_train_from_random, F1_test_from_random,
                     F1_score_train_from_OLS, F1_score_test_from_OLS,
                     F1_test_from_Lassi, F1_test_from_Rachel]

    name_label = ["train_from_random", "test_from_random",
                  "train_from_OLS", "test_from_OLS",
                  "test_LASSO", "test_Ridge"]

    test_F1 = pd.DataFrame([F1_score_list], columns=name_label)
    # test_precision.set_index("Precision")


    superlist.append(F1_score_list)

s = np.array(superlist)

for arr in s:
    for idx, val in enumerate(arr):
        if math.isnan(val):
            arr[idx] = 0

print(s.mean(axis=0))