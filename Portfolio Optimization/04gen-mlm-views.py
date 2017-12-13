from portanalytics import *
from plotsetup import *
from mlviews import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as NBC
from sklearn.svm import SVC
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import *


ertn = pd.read_excel('02exc_returns.xlsx')
ret = pd.read_excel('01returns.xlsx')
ext = pd.read_excel('06ext-features.xlsx')

# setting classifier list, enabling probability outputs
c_list = [RFC(), NBC(), SVC(probability=True)]

start = dt.datetime(2013, 12, 31)
end = dt.datetime(2016, 12, 31)
## generating pick, views and omega using (RFC) ML views and Meucci frameworks
## starting with same conditions as in the original approach

histER = ertn.mean(axis=0)*256
hist_cov_mat = (covMat(ertn)*256).round(4)
pMarket = ['IVV', 'IJH', 'IJR', 'EWC', 'EFA', 'EEM', 'SYBT', 'LQD', 'EMB', 'IYR', 'IFGL', 'GSG']
mpwts = [['IVV', 0.1200], ['IJH', 0.0451], ['IJR', 0.0316], ['EWC', 0.0119], ['EFA', 0.1239],
         ['EEM', 0.0407], ['SYBT', 0.3033], ['LQD', 0.1902], ['EMB', 0.0308], ['IYR', 0.0154],
         ['IFGL', 0.0370], ['GSG', 0.0500]]
dfMP = pd.DataFrame([r[1] for r in mpwts], columns=['Mpwts'], index=[r[0] for r in mpwts])

### calculating market variance and standard deviation
vMarket = np.dot(np.dot(dfMP.values.T, hist_cov_mat.values), dfMP.values)
sigMarket = np.sqrt(vMarket)

### setting up dataframes to hold weights are expected excess returns
df_compWts = pd.DataFrame(index=pMarket)
df_compER = pd.DataFrame(index=pMarket)
df_compWts['MP'] = dfMP.values

### market implied risk aversion Lambda
SHARPE = 0.5
L = SHARPE/(2*sigMarket)

### Annualised covariance matrix
Sigma = covMat(ertn)*256

d_er = ertn.resample('W').sum()

### BL prior expected returns
df_compER['Pi'] = blPrior(L, Sigma, dfMP).round(4)
_, preds, _, _ = classify(c_list[0], d_er, ext, todate='20150215')

P, v, Om = getMlViews(preds, df_compER['Pi'].values, hist_cov_mat, conf=.55)
df_P = pd.DataFrame(P)
df_v = pd.DataFrame(v)
df_Om = pd.DataFrame(Om)
df_P.to_excel('outB15-MeML-pick.xlsx')
df_v.to_excel('outB15-MeML-views.xlsx')
df_Om.to_excel('outB15-MeML-omega.xlsx')

start_dt = dt.datetime(2013, 12, 31)
end_dt = dt.datetime(2016, 12, 31)

### The backtest with GJR GARCH, ML and BL - full

dt_range = pd.date_range(start='20141231', end='20150331', freq='W')
L = 2.24/2
ilist = pMarket
ilist.append('Total')
df_res = pd.DataFrame(index=ilist, columns=dt_range)

res_list = []
for count, wDay in enumerate(dt_range):
    Sig = gjrCovMat(ertn, todate=wDay, horizon=5)
    _, preds, _, _ = classify(c_list[1], d_er, ext, todate=wDay)

    P, v, Om = getMlViews(preds, df_compER['Pi'], Sig, conf=0.55)
    df_P = pd.DataFrame(P)
    df_v = pd.DataFrame(v)
    df_Om = pd.DataFrame(Om)
    muBL = blBayesStd(0.01, Sig, df_P, df_v, df_compER['Pi'], Om=df_Om)
    alloc, _ = srMaxAnalytical(2.24/2, muBL, Sig)
    df_res.iloc[:-1, count] = alloc.values.flatten()
df_res.iloc[-1] = df_res.sum(axis=0)
df_res.to_excel('outB26-multi-GJRMLBL.xlsx')
