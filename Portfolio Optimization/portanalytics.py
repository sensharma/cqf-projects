import numpy as np
import pandas as pd
from math import *
from pandas.tseries.offsets import *
import datetime as dt
from arch import arch_model
import scipy as sp
from scipy.stats import norm
import scipy.optimize as scopt


def dataSlice(data, fromdate=None, todate=None, numdays=None, symbols=None):
    """
       Helper function: takes a date as yyyymmdd or datetime object and returns dataslice from dataframe
       Takes previous BD if todate is not a BD. Takes following BD if fromdate is not a BD.
       Returns for entire dataset if specific dates not specified
    """

    if todate is None:
        todate = data.ix[-1].name
    if fromdate is None:
        fromdate = data.axes[0][0]
    if numdays is None:
        numdays = len(data)
    if symbols is None:
        symbols = data.columns
    if todate not in data.index:
        if not isinstance(todate, dt.date) and not isinstance(todate, dt.date):
            todate = dt.datetime.strptime(str(todate), '%Y%m%d') - BDay(1)
        else:
            todate = todate - BDay(1)
    if todate == data.ix[-1].name and numdays == len(data) and fromdate == data.axes[0][0]:
        dataset = data
    elif numdays != len(data):
        prevdate = todate - pd.DateOffset(numdays)
        if prevdate not in data.index:
            prevdate = prevdate - BDay(1)
        dataset = data.ix[prevdate:todate, :]
    else:
        if fromdate not in data.index:
            if not isinstance(fromdate, dt.date) and not isinstance(fromdate, dt.date):
                fromdate = dt.datetime.strptime(str(fromdate), '%Y%m%d') + BDay(1)
            else:
                fromdate = fromdate - BDay(1)
        dataset = data.ix[fromdate:todate, :]
    return dataset[symbols]


def meanER(data, fromdate=None, todate=None, numdays=None, symbols=None):
    """
       Takes a date as yyyymmdd and returns mean returns for no. of days before or from a start date
       Takes previous BD if todate date entered is not a BD
       Returns for entire dataset if specific dates not specified
       Takes all symbols in dataframe if not specified
    """
    df = dataSlice(data, fromdate, todate, numdays, symbols)
    return df.mean(axis=0) #*256*100


def corrMat(data, fromdate=None, todate=None, numdays=None, symbols=None):
    """
       Takes a date as yyyymmdd and returns correl matrix for no. of days before or from a start date
       Takes previous BD if todate date entered is not a BD
       Returns for entire dataset if specific dates not specified
       Takes all symbols in dataframe if not specified
    """

    df = dataSlice(data, fromdate, todate, numdays, symbols)
    return df.corr()


def stdDev(data, fromdate=None, todate=None, numdays=None, symbols=None):
    """
       Takes a date as yyyymmdd and returns std. devn for no. of days before or from a start date
       Takes previous BD if todate date entered is not a BD
       Returns for entire dataset if specific dates not specified
       Takes all symbols in dataframe if not specified
    """

    df = dataSlice(data, fromdate, todate, numdays, symbols)
    return df.std(axis=0)


def covMat(data, fromdate=None, todate=None, numdays=None, symbols=None):
    """
    Covariance matrix for the given date range. Returns for entire dataset if range unspecified.
    Takes all symbols in dataframe if not specified
    """

    df = dataSlice(data, fromdate, todate, numdays, symbols)
    return df.cov()


def gjrForecast(data, horizon=None):
    """
    GJR GARCH forecasting for given symbols. Generates an average of the std. devn. over a horizon.
    Generated with fitting freq = 5. Default horizon = 22, approx 1 calendar month
    """

    if horizon is None:
        horizon = 22
    st_date = data.index[0]
    am_gjr = arch_model(data, p=1, o=0, q=1)
    res_gjr = am_gjr.fit(update_freq=5, disp=0, cov_type='robust')
    forecasts_gjr = res_gjr.forecast(horizon=horizon, start=st_date)
    fcgj = np.sqrt(forecasts_gjr.variance.mean(axis=1).dropna())
    return fcgj


def gjrVols(data, todate=None, horizon=None, symbols=None):
    """
    Returns GJR GARCH volatility vector for given symbols as of the param todate.
    Default horizon = 22 trading days, approx 1 calendar month
    """
    if horizon is None:
        horizon = 22
    data_set = dataSlice(data, todate=todate, symbols=symbols)
    print(data_set.index[0])
    print(data_set.index[-1])
    fc_gj_dt = []
    for sym in data_set.columns:

        # Scaled by 100 for convergence to be achieved

        ret = data_set[sym] * 100
        fcgj = gjrForecast(ret, horizon=horizon)
        fc_gj_dt.append(fcgj.ix[-1])
        print(todate)
        print(fcgj.index[-1])

    # downscaling the variance for the previous upscaling of returns and annualizing:
    fc_gj_dt_norm = [np.sqrt(item/10000)*np.sqrt(256) for item in fc_gj_dt]
    return pd.DataFrame(fc_gj_dt_norm, columns=['GJR Vol'], index=data_set.columns)


def gjrCovMat(data, fromdate=None, todate=None, numdays=None, symbols=None, horizon=None):
    """
    Returns GJR GARCH volatility covariance matrix for a given date.
    Allows the flexibility of using correlation in any desired date range.
    """

    S = (np.diag(gjrVols(data, todate=todate, horizon=horizon,
                              symbols=symbols).values.flatten()))
    R = corrMat(data, fromdate, todate, numdays, symbols)

    # annualised, but no percentage scaling
    gjr_covmat = np.dot(np.dot(S, R), S)
    return pd.DataFrame(gjr_covmat, columns=R.columns, index=R.index)


def blPrior(L, V, wM):
    """
    Black Litterman Prior Calculation
    :param L: As L/2
    :param V: Covariance matrix, annualised
    :param wM: Market weights: 3% -> 0.03
    :return: BL prior expected returns vector
    """

    Pi = 2 * L * np.dot(V, wM)
    return Pi


def blBayesStd(tau, V, P, Q, Pi, Om=None):
    """
    Standard BL Bayesian method to calculate the posterior exp rtns vector
    Can accommodate external Omega as in the Meucci views scenario.
    P -> Pick matrix, Q -> the mean returns vector associated with P
    V -> 3% = 3; Q, Pi -> 3% = 0.03
    """

    V_pi = tau * V

    # If Omega not provided, then as per He/Litterman - the inner diag extracts the diag,
    # the outer one creates a diagonal covar matrix
    ## Designed also to take external given Omega, used in case of the Meucci approach

    Pi = Pi.reshape(len(Pi), 1)
    flag = 1
    if Om is None:
        flag = 0
        Om = np.diag(np.diag(np.dot(np.dot(P, V_pi), P.T)))
    pOP = np.dot(np.dot(P.T, np.linalg.inv(Om)), P)
    pOQ = np.dot(np.dot(P.T, np.linalg.inv(Om)), Q)
    p1 = np.linalg.inv(np.linalg.inv(V_pi) + pOP)
    p2 = np.dot(np.linalg.inv(V_pi), Pi) + pOQ
    erPost = np.dot(p1, p2)
    if flag is 0:
        return Om, erPost
    else:
        return erPost


def blBayesMeucci(tau, V, P, v, Pi):
    """
    Meucci formalation of the BL Bayesian prior, where c is not used as 1 by default.
    :param tau: tau
    :param V: Covariance Matrix
    :param P: Pick/ Link Matrix
    :param v: views matrix
    :param Pi: Prior expected returns vector
    :return: BL posterior expected returns vector
    """

    V_pi = tau * V

    # Omega as per He/Litterman - the inner diag extracts the diag, the outer one creates a diagonal covar matrix
    Pi = Pi.reshape(len(Pi), 1)
    Om = np.diag(np.diag(np.dot(np.dot(P, V_pi), P.T)))
    tPOP = tau*(np.dot(np.dot(P, V), P.T)) + Om
    tSPOP = tau*(np.dot(np.dot(V, P.T), np.linalg.inv(tPOP)))
    mu = Pi + np.dot(tSPOP, (v - np.dot(P, Pi)))
    return Om, mu


def mvoUnconstrAnalytical(L, erPost, V):
    """
    Inputs as: V -> 3% = 3; Q, Pi -> 3% = 0.03
    Analytical implementation, unconstrained MVO
    """

    alloc = (1/(2*L))*np.dot(np.linalg.inv(V), erPost)
    return pd.DataFrame(alloc)


def mvoUnconstNumeric(L, erPost, V):
    """
    Given a lambda, covariance matrix and vector of excess returns, returns MVO optimized portfolio.
    Numeric optimization implementation
    """

    def mvoObj(w, ER, Sig, L):
        mu = np.dot(w.T, ER)
        sig = np.sqrt(np.dot(w.T, np.dot(Sig, w)))
        # returns -(mu - penalised variance) since this function is going to be minimized
        return -(mu - L*sig**2)

    # w0 = initial guess = equal weights
    w0 = np.ones((len(erPost), 1))/len(erPost)
    wRes = scopt.minimize(mvoObj, w0, args=(erPost, V, L),
                        method='SLSQP')
    # error handling if there is no optimization convergence
    if not wRes.success:
        raise Exception(wRes.message)
    return pd.DataFrame((wRes.x))


def mvoConstrLO(L, erPost, V):
    """
    Constrained variance minimisation: Option 0: Long only with no leverage barrier,
     Option 1: Long only with no leverage allowed
    :param L: risk aversion
    :param erPost: Excess returns dataframe
    :param V: Cov Mat
    :param constrType: 0 or 1, depending on which constraints are desired
    :return: allocation dataframe
    """

    def mvoObj(w, ER, Sig, L):
        mu = np.dot(w.T, ER)
        sig = np.sqrt(np.dot(w.T, np.dot(Sig, w)))
        # returns -(mu - penalised variance) since this function is going to be minimized
        return -(mu - L*sig**2)

    # w0 = initial guess = equal weights
    bnds = ((0, None),(0, None),(0, None),(0, None),(0, None),(0, None),(0, None),(0, None),
            (0, None),(0, None),(0, None),(0, None))

    w0 = np.ones((len(erPost), 1))/len(erPost)
    wRes = scopt.minimize(mvoObj, w0, args=(erPost, V, L),
                          method='SLSQP', bounds=bnds)
    # error handling if there is no optimization convergence
    if not wRes.success:
        raise Exception(wRes.message)
    wRes = pd.DataFrame(wRes.x)
    lv_ind = wRes.values < 0.001
    wRes.values[lv_ind] = 0
    return wRes


def mvoConstrLONL(L, erPost, V):
    """
    Constrained variance minimisation: Option 0: Long only with no leverage barrier,
     Option 1: Long only with no leverage allowed
    :param L: risk aversion
    :param erPost: Excess returns dataframe
    :param V: Cov Mat
    :param constrType: 0 or 1, depending on which constraints are desired
    :return: allocation dataframe
    """

    def mvoObj(w, ER, Sig, L):
        mu = np.dot(w.T, ER)
        sig = np.sqrt(np.dot(w.T, np.dot(Sig, w)))
        # returns -(mu - penalised variance) since this function is going to be minimized
        return -(mu - L*sig**2)

    # w0 = initial guess = equal weights
    bnds = ((0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),(0, 1),
            (0, 1),(0, 1),(0, 1),(0, 1))
    constr = ({'type':'eq', 'fun': lambda W: sum(W)-1.})

    w0 = np.ones((len(erPost), 1))/len(erPost)
    wRes = scopt.minimize(mvoObj, w0, args=(erPost, V, L),
                          method='SLSQP', bounds=bnds, constraints=constr)
    # error handling if there is no optimization convergence
    if not wRes.success:
        raise Exception(wRes.message)
    wRes = pd.DataFrame(wRes.x)
    lv_ind = wRes.values < 0.001
    wRes.values[lv_ind] = 0
    return wRes


def srMaxAnalytical(L, mu, V):
    """
    Analytical function for Sharpe Ration maximization
    """

    ones = np.ones(len(mu))
    numer = np.dot(np.linalg.inv(V), mu)
    denom = np.dot(ones, np.dot(np.linalg.inv(V), mu))
    alloc = numer/denom
    r_sharpe = np.dot(mu.T, alloc)
    varn_sharpe = np.dot(np.dot(alloc.T, V), alloc)
    sharpe = r_sharpe/np.sqrt(varn_sharpe)
    sig_L = sharpe/L
    cal_alloc = sig_L/np.sqrt(varn_sharpe)*alloc
    return pd.DataFrame(alloc), pd.DataFrame(cal_alloc)


def srMaxNumerical(L, erPost, V):
    """
    Numeric function for Sharpe Ration maximization
    """

    def srObj(w, ER, Sig):
        mu = np.dot(w.T, ER)
        sig = np.sqrt(np.dot(w.T, np.dot(Sig, w)))
        # returns the inverse of the Sharpe Ratio for minimization
        return 1/(mu/sig)
    w0 = np.ones((len(erPost), 1))/len(erPost)
    # sum of weights = 100% constraint
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1.})
    wRes = scopt.minimize(srObj, w0, args=(erPost, V),
                          method='SLSQP', constraints=c_)
    if not wRes.success:
        raise Exception(wRes.message)
    alloc = wRes.x
    r_sharpe = np.dot(erPost.T, alloc)
    varn_sharpe = np.dot(np.dot(alloc.T, V), alloc)
    sharpe = r_sharpe/np.sqrt(varn_sharpe)
    sig_L = sharpe/L
    allocCAL = sig_L/np.sqrt(varn_sharpe)*alloc
    return pd.DataFrame(alloc), pd.DataFrame(allocCAL)


def cvarMean(L, erPost, V, c, tgt):
    """
    Calculates the CVaR-mean optimization based on specified conf interval and target return
    :param L: lambda
    :param erPost: excess returns
    :param V: Cov mat
    :param c: conf interval
    :param tgt: return target
    :return: allocation
    """
    F = norm.pdf(norm.ppf(1-c))/(1-c)
    # F = norm.ppf(1-c)

    def cvarObj(W, L, ER, Sig, F):
        mu = np.dot(W.T, ER)
        sig = np.sqrt(np.dot(np.dot(W.T, Sig), W))
        return np.sqrt(L)*sig*F - mu
    cons = ({'type':'ineq', 'fun': lambda W: np.dot(W.T, erPost)-tgt})
    w0 = np.ones((len(erPost), 1))/len(erPost)
    wRes = scopt.minimize(cvarObj, w0, args=(L, erPost, V, F),
                          constraints=cons,
                          method='SLSQP')
    if not wRes.success:
        raise Exception(wRes.message)
    alloc = wRes.x
    return pd.DataFrame(alloc)


def calcExpVar(w, er, V, c):
    """
    Returns VaR for a portfolio as a Loss R.V. (can be minimized in optimization)
    :param w: weights vector
    :param er: excess returns vector
    :param V: Cov mat
    :param c: Confidence level (0.95, 0.99 etc.)
    :return: VaR
    """
    mu = np.dot(w.T, er)
    sig = np.sqrt(np.dot(w.T, np.dot(V, w)))
    var = norm.ppf(c)*sig - mu
    return var


def calcExpCVar(w, er, V, c):
    """
    Returns CVaR for a portfolio as a Loss R.V. (can be minimized in optimization)
    :param w: weights vector
    :param er: excess returns vector
    :param V: Cov mat
    :param c: Confidence level (0.95, 0.99 etc.)
    :return: CVaR
    """
    mu = np.dot(w.T, er)
    sig = np.sqrt(np.dot(w.T, np.dot(V, w)))
    cvar = sig * norm.pdf(norm.ppf(1-c))/(1-c) - mu
    return cvar


def calcSharpe(w, er, V):
    """
    Calculating Sharpe Ratio
    """
    mu = np.dot(w.T, er)
    sig = np.sqrt(np.dot(w.T, np.dot(V, w)))
    return mu/sig


def calcStd(w, V):
    """Calc Std devn for a specific portfolio, given weight and cov mat"""
    return np.sqrt(np.dot(np.dot(w.T, V), w))
