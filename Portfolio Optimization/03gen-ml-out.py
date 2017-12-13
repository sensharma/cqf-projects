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

inp = input('This may take quite a few minutes to run. '
            'Type and enter yes here to continue running \n')
if inp != 'yes':
    exit()

ertn = pd.read_excel('02exc_returns.xlsx')
modify_image(fig_width=12, fig_height=10, columns=None)

ertn = pd.read_excel('02exc_returns.xlsx')
ret = pd.read_excel('01returns.xlsx')
ext = pd.read_excel('06ext-features.xlsx')

# setting classifier list, enabling probability outputs
c_list = [RFC(), NBC(), SVC(probability=True)]

start = dt.datetime(2013, 12, 31)
end = dt.datetime(2016, 12, 31)

# backtesting, saving results and plots for RFC
bt_panel = mlBacktest(c_list[0], ertn, ext, startdate=start, todate=end)
tot_acc = bt_panel.ix[:, 'CumAccRF', :].ix[:, -1]
df_tot_acc = pd.DataFrame(tot_acc)
df_eg = pd.DataFrame(bt_panel.ix[:, :, 0].ix[:, :6])
df_tot_acc.to_excel('outB11-RFC-acc.xlsx')
df_eg.to_excel('outB14-RFC-IVV-slice.xlsx')

fig1 = plt.figure(figsize=(12, 7))
for col in ertn.columns:
    ax = bt_panel.ix[:, 'CumAccRF', :].ix[col, :].plot(label=col)
ax.set_xlabel('Dates')
ax.set_ylabel('Accuracy')
ax.legend(bbox_to_anchor=(1.12, 1.02))
fig1.savefig('B03RFC-conv.pdf')

# backtesting, saving results and plots for NBC
bt_panel = mlBacktest(c_list[1], ertn, ext, startdate=start, todate=end)
tot_acc = bt_panel.ix[:, 'CumAccRF', :].ix[:, -1]
df_tot_acc = pd.DataFrame(tot_acc)
df_tot_acc.to_excel('outB12-NBC-acc.xlsx')

fig2 = plt.figure(figsize=(12, 7))
for col in ertn.columns:
    ax = bt_panel.ix[:, 'CumAccRF', :].ix[col, :].plot(label=col)
ax.set_xlabel('Dates')
ax.set_ylabel('Accuracy')
ax.legend(bbox_to_anchor=(1.12, 1.02))
fig2.savefig('B04NBC-conv.pdf')

# backtesting, saving results and plots for SVC
bt_panel = mlBacktest(c_list[2], ertn, ext, startdate=start, todate=end)
tot_acc = bt_panel.ix[:, 'CumAccRF', :].ix[:, -1]
df_tot_acc = pd.DataFrame(tot_acc)
df_tot_acc.to_excel('outB13-SVC-acc.xlsx')

fig3 = plt.figure(figsize=(12, 7))
for col in ertn.columns:
    ax = bt_panel.ix[:, 'CumAccRF', :].ix[col, :].plot(label=col)
ax.set_xlabel('Dates')
ax.set_ylabel('Accuracy')
ax.legend(bbox_to_anchor=(1.12, 1.02))
fig3.savefig('B05SVC-conv.pdf')

