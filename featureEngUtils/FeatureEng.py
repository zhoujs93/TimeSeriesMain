import time
from utils import *
from featureEngUtils.ta_utils import *


def feature_engineering_ta(df, df_grouped):
    cols = ['close']
    v = 'volume'
    for c in cols:
        # bollinger bands
        df[c + '_lb'] = df_grouped[c].apply(lambda x: bollinger_lband(x))
        df[c + '_mb'] = df_grouped[c].apply(lambda x: bollinger_mavg(x))
        df[c + '_hb'] = df_grouped[c].apply(lambda x: bollinger_hband(x))
        df[c + '_hband'] = df_grouped[c].apply(lambda x: bollinger_hband_indicator(x))
        df[c + '_lband'] = df_grouped[c].apply(lambda x: bollinger_lband_indicator(x))

        # donchian
        df[c + '_dclband'] = df_grouped[c].apply(lambda x: donchian_channel_lband(x))
        df[c + '_dchband'] = df_grouped[c].apply(lambda x: donchian_channel_hband(x))
        df[c + '_dcihband'] = df_grouped[c].apply(lambda x: donchian_channel_hband_indicator(x))
        df[c + '_dcilband'] = df_grouped[c].apply(lambda x: donchian_channel_lband_indicator(x))
    return df


def feature_engineering_ta2(df, grouped):
    cols = ['close', 'returnsClosePrevMktres10']
    for c in cols:
        df_grouped = grouped[c]
        df[c + '_hband'] = df_grouped.apply(lambda x: bollinger_hband_indicator(x, 10, 2))
        df[c + '_lband'] = df_grouped.apply(lambda x: bollinger_lband_indicator(x, 10, 2))

        df[c + '_dclband'] = df_grouped.apply(lambda x: donchian_channel_lband_indicator(x, 10))
        df[c + '_dchband'] = df_grouped.apply(lambda x: donchian_channel_hband_indicator(x, 10))

        df[c + '_macd'] = df_grouped.apply(lambda x: macd(x, 6, 13))
        df[c + '_macd_signal'] = df_grouped.apply(lambda x: macd_signal(x, 6, 13, 5))
        df[c + '_macd_diff'] = df_grouped.apply(lambda x: macd_diff(x, 6, 13, 5))

        df[c + '_trix'] = df_grouped.apply(lambda x: trix(x, 8))
        df[c + '_dpo'] = df_grouped.apply(lambda x: dpo(x, 10))

        df[c + '_dr'] = df_grouped.apply(lambda x: daily_return(x))
        df[c + '_cr'] = df_grouped.apply(lambda x: cumulative_return(x))

        df[c + '_rsi'] = df_grouped.apply(lambda x: rsi(x, 7))
        df[c + '_tsi'] = df_grouped.apply(lambda x: tsi(x, 13, 7))

    return df


def generate_ema_features(x, x_grouped):
    # CW Features
    x = x.assign(ema_om1=x_grouped.apply(lambda g: g.returnsOpenPrevMktres1.ewm(10).mean()).reset_index(0, drop=True))
    x = x.assign(ema_om10=x_grouped.apply(lambda g: g.returnsOpenPrevMktres10.ewm(10).mean()).reset_index(0, drop=True))
    x = x.assign(ema_cm1=x_grouped.apply(lambda g: g.returnsClosePrevMktres1.ewm(10).mean()).reset_index(0, drop=True))
    x = x.assign(
        ema_cm10=x_grouped.apply(lambda g: g.returnsClosePrevMktres10.ewm(10).mean()).reset_index(0, drop=True))
    x = x.assign(volume10=x_grouped['volume'].apply(lambda g: g.ewm(10).mean()).reset_index(0, drop=True))
    return x


def feature_engineering_rolling(df, df_grouped, n=5):
    df_inverse_grouped = df[::-1].groupby(['assetCode'])
    cols = ['returnsClosePrevRaw1',
            'returnsOpenPrevRaw1',
            'returnsClosePrevMktres1',
            'returnsOpenPrevMktres1',
            'returnsClosePrevRaw10',
            'returnsOpenPrevRaw10',
            'returnsClosePrevMktres10',
            'returnsOpenPrevMktres10']

    for c in cols:
        # rolling
        for h in [3, 5, 10, 15, 20]:
            df[c + '_rolling_mean_' + str(h)] = df_grouped[c].apply(lambda x: x.rolling(h, min_periods=0).mean())
            df[c + '_inverse_rolling_mean_' + str(h)] = df_inverse_grouped[c].apply(
                lambda x: x.rolling(h, min_periods=0).mean())[::-1]
        # diffs
        df[c + '_diff_1'] = df_grouped[c].apply(lambda x: x.diff().fillna(method="backfill").fillna(0))
        df[c + '_diff_2'] = df_grouped[c].apply(lambda x: x.diff(2).fillna(method="backfill").fillna(0))
        df[c + '_diff_3'] = df_grouped[c].apply(lambda x: x.diff(3).fillna(method="backfill").fillna(0))
        # cumsum
        df[c + '_cumsum'] = df_grouped[c].apply(lambda x: x.cumsum())
        # volatility
        df[c + '_volatility'] = 0
        df.loc[abs(df[c] - df[c].mean()) > (n * df[c].std()), c + '_volatility'] = 1
    return df

def fillna_bystock(df, df_grouped):
    num_cols = df.select_dtypes('number').columns
    for c in num_cols:
        df[c] = df.groupby(['assetCode']).apply(lambda x: x.fillna(x.median()))
    return df

# feature generation helper functions
def generate_momentum(x, groups, periods = [1,2,3,4,5,6,7,8,9,10, 11, 12]):
    for p in periods:
        x.loc[:, 'momentum_' + str(p * 21)] = groups['returnsClosePrevMktres1'].transform(lambda g: g.pct_change(p * 21))
    return x


def proc_df(market_train_df, symbols=None, current_t=None):
    if current_t is None:
        market_train_df = market_train_df.loc[market_train_df.time > '2007-02-15', :]
    else:
        initialization_t = current_t - pd.tseries.offsets.DateOffset(years=2)
        market_train_df = market_train_df.loc[market_train_df.time > initialization_t, :]
    try:
        x = market_train_df.loc[market_train_df.universe != 0, :].copy()
        del market_train_df
    except:
        x = market_train_df.copy()
        del market_train_df
    groups = x.groupby(['assetCode'])
    print(f'Moving onto EMA')
    x = generate_ema_features(x, groups)
    print(f'Moving onto TA')
    x = feature_engineering_ta(x, groups)
    print(f'Moving onto TA2')
    x = feature_engineering_ta2(x, groups)
    print(f'Moving onto rolling')
    x = feature_engineering_rolling(x, groups)
    print(f'Moving onto Momentum')
    x = generate_momentum(x, groups)

    if symbols is not None:
        x = x.loc[x.assetCode.isin(symbols)]
    if current_t is not None:
        x = x.loc[x.time == current_t]

    return x
