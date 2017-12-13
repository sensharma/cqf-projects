import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import quandl as ql


# tickers to be downloaded from yahoo and quandl
tlist = [['IVV', 'y'], ['IJH', 'y'], ['IJR', 'y'], ['EWC', 'y'], ['EFA', 'y'],
            ['EEM', 'y'], ['WIKI/SYBT', 'q'], ['LQD', 'y'], ['EMB', 'y'], ['IYR', 'y'],
            ['IFGL', 'y'], ['GSG', 'y']]
dlist = [['YAHOO/INDEX_SPY', 'q', 'Adjusted Close'], ['CBOE/VIX', 'q', 'VIX Close'],
         ['GLD', 'y'], ['CBOE/EQUITY_PC', 'q', 'P/C Ratio'],
         ['FRED/DTWEXM', 'q', 'VALUE']]


def getreturns(tickers, process='yes'):
    '''
    Function to clean and save data, calculate daily log returns, excess returns,
    normalized return levels or just the plain data depending on given process param
    and saving to folder
    '''

    start_date = '1900-01-01'
    end_date = '2017-01-01'
    dates = pd.date_range(start_date, end_date)
    df_lev = pd.DataFrame(index=dates)
    df_ret = pd.DataFrame(index=dates)
    df_ext = pd.DataFrame(index=dates)
    tick_list = []
    for tick in tickers:
        print(tick)
        if tick[1] == 'y':
            df_data = web.DataReader(tick[0], 'yahoo', start_date, end_date)
            ticker = tick[0]
            tick_list.append(ticker)
            df_data.rename(columns={'Adj Close': ticker}, inplace=True)
            if process == 'no':
                df_ext = df_ext.join(pd.DataFrame(df_data[ticker]), how='inner')
                continue
        else:
            df_data = ql.get(tick[0], )
            ticker = tick[0].split(sep='/')[1]
            tick_list.append(ticker)
            if process == 'no':
                df_data.rename(columns={tick[2]: ticker}, inplace=True)
                df_ext = df_ext.join(pd.DataFrame(df_data[ticker]), how='inner')
                continue
            df_data.rename(columns={'Adj. Close': ticker}, inplace=True)
        df_lev = df_lev.join(pd.DataFrame(df_data[ticker]), how='inner')
        df_lev.dropna(how='any', inplace=True)

        # Calculating log returns
        df_data['Returns'] = np.log(df_data[ticker] / df_data[ticker].shift(1))
        df_ret = df_ret.join(pd.DataFrame(df_data.Returns), how='inner')
        df_ret.rename(columns={'Returns': ticker}, inplace=True)
        df_ret.dropna(inplace=True)

        # 3mTbills
        tbill3 = ql.get('FRED/DTB3') / (100 * 360)
        tbill3 = tbill3.reindex(df_ret.index).fillna(method='pad')

        # Excess Returns
        df_exc_ret = df_ret.sub(tbill3.VALUE, axis=0)
        df_exc_ret.dropna(how='any', inplace=True)
        df_start = pd.DataFrame(np.ones((1, len(df_ret.columns))), columns=tick_list)
        df_start['date'] = df_lev.index[0] + pd.DateOffset(-1)
        df_start.set_index('date', inplace=True)

        # Normalized returns
        df_norm_lev = df_start.append(df_ret)
        df_norm_lev = df_norm_lev.cumsum(axis=0)
        df_norm_lev.dropna(how='any', inplace=True)
        df_norm_lev_exc = df_start.append(df_exc_ret)
        df_norm_lev_exc = df_norm_lev_exc.cumsum(axis=0)
        df_norm_lev_exc.dropna(how='any', inplace=True)
    if process == 'yes':
        return df_lev, df_ret, df_exc_ret, df_norm_lev, df_norm_lev_exc
    else:
        return df_ext

# Getting and saving the asset data
dfL, dfR, dfER, dfNL, dfNLE = getreturns(tlist)

dfL.to_excel('00levels.xlsx', header=True)
dfR.to_excel('01returns.xlsx', header=True)
dfER.to_excel('02exc_returns.xlsx', header=True)
dfNL.to_excel('03norm-levels.xlsx', header=True)
dfNLE.to_excel('04exc_norm-levels.xlsx', header=True)

# Getting and saving the extra data
dfD = getreturns(dlist, process='no')
dfD.columns = ['SPY', 'VIX', 'GLD', 'PCPar', 'USDIndex']

# Calculating and saving extra features (to be used in the mlviews
dfFeat = pd.DataFrame()
dfFeat['SPY'] = np.log(dfD.SPY/dfD.SPY.shift(1))
dfFeat['VIX'] = dfD.VIX.diff(1)
dfFeat['GLD'] = np.log(dfD.GLD/dfD.GLD.shift(1))
dfFeat['PCR'] = dfD.PCPar.diff(1)
dfFeat['USDI'] = dfD.USDIndex.diff(1)
dfFeat = dfFeat.reindex(dfER.index).fillna(method='pad')
dfD.to_excel('05other_data.xlsx')
dfFeat.to_excel('06ext-features.xlsx')
