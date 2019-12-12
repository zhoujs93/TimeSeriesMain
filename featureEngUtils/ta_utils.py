import numpy as np
import pandas as pd


# add technical analysis feature engineering
def bollinger_hband_indicator(close, n=20, ndev=2, fillna=True):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    if fillna:
        df['hband'] = df['hband'].fillna(0)
    return pd.Series(df['hband'], name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2, fillna=True):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    if fillna:
        df['lband'] = df['lband'].fillna(0)
    return pd.Series(df['lband'], name='bbilband')


def donchian_channel_hband_indicator(close, n=20, fillna=True):
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close >= hband, 'hband'] = 1.0
    if fillna:
        df['hband'] = df['hband'].fillna(0)
    return pd.Series(df['hband'], name='dcihband')


def donchian_channel_lband_indicator(close, n=20, fillna=True):
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close <= lband, 'lband'] = 1.0
    if fillna:
        df['lband'] = df['lband'].fillna(0)
    return pd.Series(df['lband'], name='dcilband')


def macd(close, n_fast=12, n_slow=26, fillna=True):
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    if fillna:
        macd = macd.fillna(0)
    return pd.Series(macd, name='MACD_%d_%d' % (n_fast, n_slow))


def macd_signal(close, n_fast=12, n_slow=26, n_sign=9, fillna=True):
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    macd = macd.ewm(n_sign).mean()
    if fillna:
        macd = macd.fillna(0)
    return pd.Series(macd, name='MACD')


def macd_diff(close, n_fast=12, n_slow=26, n_sign=9, fillna=True):
    emafast = close.ewm(n_fast).mean()
    emaslow = close.ewm(n_slow).mean()
    macd = emafast - emaslow
    macdsign = macd.ewm(n_sign).mean()
    macd = macd - macdsign
    if fillna:
        macd = macd.fillna(0)
    return pd.Series(macd, name='MACD_diff_%d_%d' % (n_fast, n_slow))


def trix(close, n=15, fillna=True):
    ema1 = close.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    ema3 = ema2.ewm(span=n, min_periods=n-1).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix*1000
    if fillna:
        trix = trix.fillna(0)
    return pd.Series(trix, name='trix_'+str(n))


def dpo(close, n=20, fillna=True):
    dpo = close.shift(int(n/(2+1))) - close.rolling(n).mean()
    if fillna:
        dpo = dpo.fillna(0)
    return pd.Series(dpo, name='dpo_'+str(n))


def daily_return(close, fillna=True):
    dr = (close / close.shift(1)) - 1
    dr *= 100
    if fillna:
        dr = dr.fillna(0)
    return pd.Series(dr, name='d_ret')


def cumulative_return(close, fillna=True):
    cr = (close / close.iloc[0]) - 1
    cr = cr * 100
    if fillna:
        cr = cr.fillna(method='backfill')
    return pd.Series(cr, name='cum_ret')


def rsi(close, n=14, fillna=True):
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = up.ewm(n).mean()
    emadn = dn.ewm(n).mean()

    rsi = 100 * emaup/(emaup + emadn)
    if fillna:
        rsi = rsi.fillna(50)
    return pd.Series(rsi, name='rsi')


def tsi(close, r=25, s=13, fillna=True):
    m = close - close.shift(1)
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1/m2
    if fillna:
        tsi = tsi.fillna(0)
    return pd.Series(100*tsi, name='tsi')


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=True):
    rocma1 = (close / close.shift(r1) - 1).rolling(n1).mean()
    rocma2 = (close / close.shift(r2) - 1).rolling(n2).mean()
    rocma3 = (close / close.shift(r3) - 1).rolling(n3).mean()
    rocma4 = (close / close.shift(r4) - 1).rolling(n4).mean()
    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)
    sig = kst.rolling(nsig).mean()
    if fillna:
        sig = sig.fillna(0)
    return pd.Series(sig, name='sig')

def bollinger_hband(close, n=20, ndev=2):
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    return pd.Series(hband, name='hband')


def bollinger_lband(close, n=20, ndev=2):
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    return pd.Series(lband, name='lband')


def bollinger_mavg(close, n=20):
    mavg = close.rolling(n).mean()
    return pd.Series(mavg, name='mavg')


def bollinger_hband_indicator(close, n=20, ndev=2):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2):
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='bbilband')


def donchian_channel_hband(close, n=20):
    hband = close.rolling(n).max()
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(close, n=20):
    lband = close.rolling(n).min()
    return pd.Series(lband, name='dclband')


def donchian_channel_hband_indicator(close, n=20):
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close > hband, 'hband'] = 1.0
    return pd.Series(df['hband'], name='dcihband')


def donchian_channel_lband_indicator(close, n=20):
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close < lband, 'lband'] = 1.0
    return pd.Series(df['lband'], name='dcilband')

def on_balance_volume(close, volume):
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return df['OBV']


def on_balance_volume_mean(close, volume, n=10):
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    return pd.Series(df['OBV'].rolling(n).mean(), name='obv_mean')


def force_index(close, volume, n=2):
    return pd.Series(close.diff(n) * volume.diff(n), name='fi_'+str(n))


def volume_price_trend(close, volume):
    vpt = volume * ((close - close.shift(1)) / close.shift(1).astype(float))
    vpt = vpt.shift(1) + vpt
    return pd.Series(vpt, name='vpt')