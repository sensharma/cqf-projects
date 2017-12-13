from arch import arch_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portanalytics import *


dfMD = pd.read_excel('05other_data.xlsx')
ertn = pd.read_excel('02exc_returns.xlsx')

dfSPY = pd.DataFrame(dfMD['SPY'], columns=['SPY'])
dfSPY = pd.DataFrame(np.log(dfSPY.SPY/dfSPY.SPY.shift(1))*100).dropna()
st_date = dfSPY.index[0]

# "historical" std devn
exp_std = pd.DataFrame(dfSPY.expanding().std()*np.sqrt(252)).loc[st_date:]

# rolling realized (forward) std. devn.
roll30_std_fwd = (pd.DataFrame(dfSPY[::-1].rolling(22).std().dropna().iloc[::-1]*
                              np.sqrt(252)).loc[st_date:])
roll30_std_fwd_m = roll30_std_fwd.resample('M').last()

# rolling historical standard deviation
roll30_std = pd.DataFrame(dfSPY.rolling(22).std().dropna()*np.sqrt(252)).loc[st_date:]
roll30_std_m = roll30_std.resample('M').last()

# rolling GJR GARCH forecast
fcgj = gjrForecast(dfSPY, horizon=22)
fcgj_m = pd.DataFrame(fcgj).resample('M').last()

# Plotting
fig1 = plt.figure(figsize=(10, 6))
plt.style.use('seaborn-muted')
plt.plot(exp_std, label='Historical St Dev.')
plt.plot(fcgj_m, label='GJR GARCH Forecasts (Monthly Avg.)')
plt.plot(roll30_std_m, label='Historical 30d Rolling Realised St Dev.')
plt.plot(roll30_std_fwd_m, label='Forward (Realised) 30d Rolling St Dev.')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Annualised Vol(%)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Vol Forecast vs. Realised', fontsize=14)
plt.grid()
fig1.savefig('B02vol-comp.pdf')

##### GJR GARCH Covariance matrix output #####

gcovmat = gjrCovMat(ertn, todate='20161231', horizon=22)
gcovmat.to_excel('outB10-garch-cov.xlsx')
