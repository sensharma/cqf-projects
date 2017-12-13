from portanalytics import *
from mlviews import *
import pandas as pd
from time import time

ertn = pd.read_excel('02exc_returns.xlsx')
ret = pd.read_excel('01returns.xlsx')

histER = ertn.mean(axis=0)*256
hist_cov_mat = (covMat(ertn)*256).round(4)

hist_cov_mat.to_excel('outB02_cov_mat.xlsx')

pMarket = ['IVV', 'IJH', 'IJR', 'EWC', 'EFA', 'EEM', 'SYBT', 'LQD', 'EMB', 'IYR', 'IFGL', 'GSG']
wMarket = mpwts = [['IVV', 0.1200], ['IJH', 0.0451], ['IJR', 0.0316], ['EWC', 0.0119], ['EFA', 0.1239],
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
df_compER['Hist'] = histER.round(4)

### market implied risk aversion Lambda
SHARPE = 0.5
L = SHARPE/(2*sigMarket)

### Annualised covariance matrix
Sigma = covMat(ertn)*256

### BL prior expected returns
df_compER['Pi'] = blPrior(L, Sigma, dfMP).round(4)
df_compER.to_excel('outB03_er_comp.xlsx')

#### tau as a standard error of estimate - monthly ####
tau = 1/(len(ertn)/22)

### Demo Pick/Link and views matrix
P = pd.DataFrame(np.zeros((3, 12)), columns=pMarket)
P.EMB[0] = 1
P.IYR[1] = -1
P.IFGL[1] = 1
P.IVV[2] = 1

writer = pd.ExcelWriter('outB04_views.xlsx')
P.to_excel(writer, 'pick')

vL = np.array((0.05, 0.03, 0.07))
v = pd.DataFrame(vL, columns=['v'])
v.to_excel(writer, 'views')

t1 = time()
Om, muBL = blBayesStd(tau, Sigma, P, v, df_compER.Pi)
t2 = time() - t1

Om = pd.DataFrame(Om).round(5)
Om.to_excel(writer, 'Omega')
df_compER['muBL'] = muBL.round(4)

t3 = time()
Om_M, muBL_M = blBayesMeucci(tau, Sigma, P, v, df_compER.Pi)
t4 = time() - t3

Om_M = pd.DataFrame(Om_M).round(5)
Om_M.to_excel(writer, 'OmegaM')

writer.save()

##### BL Posteriors #########

df_compER['muBLM'] = muBL_M.round(4)
df_compER.to_excel('outB05-compER.xlsx')

############## MVO ###########

ilist = pMarket.append('Total')
df_mvo = pd.DataFrame(index=pMarket, columns=['An', 'Num'])
mvo_alloc_an = mvoUnconstrAnalytical(L, df_compER.muBLM.reshape(len(df_compER), 1), Sigma)
df_mvo['An'][:-1] = mvo_alloc_an.values.flatten()

ilist2 = [r[0] for r in mpwts]
ilist2.append('Total')
dfComp = pd.DataFrame(index=ilist2, columns=['muBL', 'Pi', 'muBL - Pi', 'BL Alloc',
                                                                       'MktWts', 'BL Alloc - Mkt'])
dfComp['muBL'][:-1] = df_compER['muBL'].values.flatten()
dfComp['Pi'][:-1] = df_compER['Pi']
dfComp['muBL - Pi'][:-1] = (dfComp['muBL'][:-1] - dfComp['Pi'][:-1])
dfComp['BL Alloc'][:-1] = df_mvo['An'][:-1]
dfComp['MktWts'][:-1] = dfMP.values.flatten()
dfComp['BL Alloc - Mkt'][:-1] = dfComp['BL Alloc'][:-1] - dfComp['MktWts'][:-1]
dfComp['BL Alloc'][-1] = dfComp['BL Alloc'][:-1].sum()
dfComp['MktWts'][-1] = dfComp['MktWts'][:-1].sum()
dfComp.fillna('', inplace=True)
dfComp = dfComp.round(5)
dfComp.to_excel('outB06-comp-table.xlsx')


#### Analytical vs. Numeric ####

mvo_alloc_num = mvoUnconstNumeric(L, df_compER.muBLM, Sigma)
df_mvo['Num'][:-1] = mvo_alloc_num.values.flatten()
df_mvo['Num'][-1] = df_mvo['Num'][:-1].sum()
df_mvo['An'][-1] = df_mvo['An'][:-1].sum()
df_mvo = round(df_mvo, 4)
df_mvo.to_excel('outB07-an-num.xlsx')

#### Analytical Variation min - Various lambdas allocation and risk characteristics######
lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['L1', 'L2', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['L1', 'L2', 'L3'])
for i, L in enumerate(lam_list):
    mvo_alloc = mvoUnconstrAnalytical(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, i] = mvo_alloc.values.flatten()
    er = np.dot(mvo_alloc.T, muBL)
    stdev = calcStd(mvo_alloc, Sigma)
    varn = stdev**2
    sharpe = calcSharpe(mvo_alloc, muBL, Sigma)
    var = calcExpVar(mvo_alloc, muBL, Sigma, 0.95)
    cvar = calcExpCVar(mvo_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,i] = np.array([er, stdev, varn, sharpe, var, cvar]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams.round(4)
df_lams.to_excel('outB08-lam-comp.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB09-lam-rr.xlsx')


#### Numerical variance min - Various lambdas allocation and risk characteristics######
lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['L1', 'L2', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['L1', 'L2', 'L3'])
for i, L in enumerate(lam_list):
    mvo_alloc = mvoUnconstNumeric(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, i] = mvo_alloc.values.flatten()
    er = np.dot(mvo_alloc.T, muBL)
    stdev = calcStd(mvo_alloc, Sigma)
    varn = stdev**2
    sharpe = calcSharpe(mvo_alloc, muBL, Sigma)
    var = calcExpVar(mvo_alloc, muBL, Sigma, 0.95)
    cvar = calcExpCVar(mvo_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,i] = np.array([er, stdev, varn, sharpe, var, cvar]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams.round(4)
df_lams.to_excel('outB22-lam-comp-num.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB23-lam-rr-num.xlsx')


#### Constrained variance min - Various lambdas allocation and risk characteristics######
#### First optimization to get Long only, no bound on leverage
#### Second optimization to get long only, no leverage allowed

lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['L1', 'L2', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['L1', 'L2', 'L3'])
for i, L in enumerate(lam_list):
    mvo_alloc = mvoConstrLO(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, i] = mvo_alloc.values.flatten()
    er = np.dot(mvo_alloc.T, muBL)
    stdev = calcStd(mvo_alloc, Sigma)
    varn = stdev**2
    sharpe = calcSharpe(mvo_alloc, muBL, Sigma)
    var = calcExpVar(mvo_alloc, muBL, Sigma, 0.95)
    cvar = calcExpCVar(mvo_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,i] = np.array([er, stdev, varn, sharpe, var, cvar]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams.round(4)
df_lams.to_excel('outB27-lam-num-constr.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB28-lam-rr-num-constr.xlsx')

# Calculating all the allocations, risk and return values etc.

lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['L1', 'L2', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['L1', 'L2', 'L3'])
for i, L in enumerate(lam_list):
    mvo_alloc = mvoConstrLONL(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, i] = mvo_alloc.values.flatten()
    er = np.dot(mvo_alloc.T, muBL)
    stdev = calcStd(mvo_alloc, Sigma)
    varn = stdev**2
    sharpe = calcSharpe(mvo_alloc, muBL, Sigma)
    var = calcExpVar(mvo_alloc, muBL, Sigma, 0.95)
    cvar = calcExpCVar(mvo_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,i] = np.array([er, stdev, varn, sharpe, var, cvar]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams.round(4)
df_lams.to_excel('outB29-lam-num-constr.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB30-lam-rr-num-constr.xlsx')


#### SR Opt - CAL Allocation - analytical varying lambdas####
lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['TP', 'L1', 'TP2', 'L2', 'TP3', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['TP', 'L1', 'TP2', 'L2', 'TP3', 'L3'])

# Calculating all the allocations, risk and return values etc.
for i, L in enumerate(lam_list):
    sr_alloc, cal_alloc = srMaxAnalytical(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, 2*i] = sr_alloc.values.flatten()
    df_lams.iloc[:-1, 2*i+1] = cal_alloc.values.flatten()
    er_t = np.dot(sr_alloc.T, muBL)
    er_c = np.dot(sr_alloc.T, muBL)
    stdev_t = calcStd(sr_alloc, Sigma)
    stdev_c = calcStd(cal_alloc, Sigma)
    varn_t = stdev_t**2
    varn_c = stdev_c**2
    sharpe_t = calcSharpe(sr_alloc, muBL, Sigma)
    sharpe_c = calcSharpe(cal_alloc, muBL, Sigma)
    var_t = calcExpVar(sr_alloc, muBL, Sigma, 0.95)
    var_c = calcExpVar(cal_alloc, muBL, Sigma, 0.95)
    cvar_t = calcExpCVar(sr_alloc, muBL, Sigma, 0.95)
    cvar_c = calcExpCVar(cal_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,2*i] = np.array([er_t, stdev_t, varn_t, sharpe_t, var_t, cvar_t]).reshape(6, 1)
    df_lams_rr.iloc[:,2*i+1] = np.array([er_c, stdev_c, varn_c, sharpe_c, var_c, cvar_c]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams = df_lams[['TP', 'L1', 'L2', 'L3']]
df_lams.round(4)
df_lams_rr= df_lams_rr[['TP', 'L1', 'L2', 'L3']]
df_lams.to_excel('outB18-sran-lcomp-an.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB19-sran-rr-an.xlsx')

### SR OPT - CAL Alloc - numerical varying lambda ###
lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['TP', 'L1', 'TP2', 'L2', 'TP3', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['TP', 'L1', 'TP2', 'L2', 'TP3', 'L3'])
for i, L in enumerate(lam_list):
    sr_alloc, cal_alloc = srMaxNumerical(L, df_compER.muBL, Sigma)
    df_lams.iloc[:-1, 2*i] = sr_alloc.values.flatten()
    df_lams.iloc[:-1, 2*i+1] = cal_alloc.values.flatten()
    er_t = np.dot(sr_alloc.T, muBL)
    er_c = np.dot(sr_alloc.T, muBL)
    stdev_t = calcStd(sr_alloc, Sigma)
    stdev_c = calcStd(cal_alloc, Sigma)
    varn_t = stdev_t**2
    varn_c = stdev_c**2
    sharpe_t = calcSharpe(sr_alloc, muBL, Sigma)
    sharpe_c = calcSharpe(cal_alloc, muBL, Sigma)
    var_t = calcExpVar(sr_alloc, muBL, Sigma, 0.95)
    var_c = calcExpVar(cal_alloc, muBL, Sigma, 0.95)
    cvar_t = calcExpCVar(sr_alloc, muBL, Sigma, 0.95)
    cvar_c = calcExpCVar(cal_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,2*i] = np.array([er_t, stdev_t, varn_t, sharpe_t, var_t, cvar_t]).reshape(6, 1)
    df_lams_rr.iloc[:,2*i+1] = np.array([er_c, stdev_c, varn_c, sharpe_c, var_c, cvar_c]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams = df_lams[['TP', 'L1', 'L2', 'L3']]
df_lams.round(4)
df_lams_rr= df_lams_rr[['TP', 'L1', 'L2', 'L3']]
df_lams.to_excel('outB20-sran-lcomp-num.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB21-sran-rr-num.xlsx')


#### CVaR-Mean - Various lambdas allocation and risk characteristics######
lam_list = [0.01/2, 2.24/2, 6/2]
df_lams = pd.DataFrame(index=ilist2, columns=['L1', 'L2', 'L3'])
df_lams_rr = pd.DataFrame(index=['Exp Rtn.', 'Exp Std.', 'Exp Var', 'Sharpe', 'Exp VaR(95%)', 'Exp CVaR(95%)'],
                          columns=['L1', 'L2', 'L3'])
for i, L in enumerate(lam_list):
    cvar_alloc = cvarMean(L, df_compER.muBL, Sigma, 0.95, 0.10)
    df_lams.iloc[:-1, i] = cvar_alloc.values.flatten()
    er = np.dot(cvar_alloc.T, muBL)
    stdev = calcStd(cvar_alloc, Sigma)
    varn = stdev**2
    sharpe = calcSharpe(cvar_alloc, muBL, Sigma)
    var = calcExpVar(cvar_alloc, muBL, Sigma, 0.95)
    cvar = calcExpCVar(cvar_alloc, muBL, Sigma, 0.95)
    df_lams_rr.iloc[:,i] = np.array([er, stdev, varn, sharpe, var, cvar]).reshape(6, 1)

df_lams.loc['Total'] = df_lams[:-1].sum()
df_lams.round(4)
df_lams.to_excel('outB24-lam-comp-cvar.xlsx')
df_lams_rr.round(4)
df_lams_rr.to_excel('outB25-lam-rr-cvar.xlsx')

