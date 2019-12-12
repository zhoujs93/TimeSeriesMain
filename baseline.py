import argparse
import os, pathlib
import pandas as pd
import numpy as np
from tqdm import trange
from scipy import stats
import feather, argparse
import matplotlib.pyplot as plt
from featureEngUtils.FeatureEng import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils
from evaluate import evaluate
from dataloader import *
from statsmodels.tsa.arima_model import ARIMA
import matplotlib
from sklearn.metrics import mean_squared_error

matplotlib.use('Agg')
from dataloader import *

import matplotlib

matplotlib.use('Agg')
import pandas as pd

logger = logging.getLogger('Baseline.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='stock', help='Name of the dataset')
parser.add_argument('--data_folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model_name', default='base_stock_model', help='Directory containing params.json')
parser.add_argument('--relative_metrics', default=0, type=int, help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', default=0, type=int, help='Whether to sample during evaluation')
parser.add_argument('--save-best', default=0, type=int, help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def gen_covariates(df, num_covariates):
    covariates = np.zeros((df.shape[0], num_covariates))  # modified
    covariates[:, 1] = df.index.month
    return covariates

def prep_data(data, covariates, data_start, window_size,
              stride_size, save_path, total_time, ts_covariates = None, train = True, num_covariates = 2):
    time_len = data.shape[0]
    num_series = data.shape[1]
    input_size = window_size - stride_size
    windows_per_series = np.full((num_series), (time_len - input_size) // stride_size)

    if ts_covariates is not None:
        add_covariates = len(ts_covariates)

    if train:
        strides = (data_start + stride_size - 1) // stride_size # modified
        windows_per_series = np.fmax(windows_per_series - strides, 1) # modified

    total_windows = np.sum(windows_per_series)

    if ts_covariates is not None:
        x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1 + add_covariates), dtype = float)
    else:
        x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype = float)

    label = np.zeros((total_windows, window_size), dtype = float)
    v_input = np.zeros((total_windows, 2), dtype = float)

    count = 0
    if not train:
        covariates = covariates[-time_len:]

    for series in trange(num_series):
        try:
            cov_age = stats.zscore(np.arange(total_time - data_start[series]))
            if train:
                #covariates[data_start[series]:time_len, 0] = cov_age[:time_len - data_start[series]]
                covariates = covariates
            else:
                covariates[:, 0] = cov_age[-time_len:]
            for i in range(windows_per_series[series]):
                try:
                    if train:
                        window_start = stride_size * i + data_start[series]
                    else:
                        window_start = stride_size * i
                    window_end = window_start + window_size
                    '''
                    print("x: ", x_input[count, 1:, 0].shape)
                    print("window start: ", window_start)
                    print("window end: ", window_end)
                    print("data: ", data.shape)
                    print("d: ", data[window_start:window_end-1, series].shape)
                    '''
                    x_input[count, 1:, 0] = data[window_start:window_end - 1, series]

                    x_input[count, :, 1:1 + num_covariates] = covariates[window_start:window_end, :]

                    label[count, :] = data[window_start:window_end, series]
                    nonzero_sum = (~np.isnan(x_input[count,1:input_size,0])).sum()
                    if nonzero_sum == 0:
                        v_input[count, 0] = 0
                    else:
                        v_input[count, 0] = np.true_divide(x_input[count,1:input_size,0].sum(), nonzero_sum) + 1
                        x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                        if train:
                            label[count, :] = label[count, :] / v_input[count, 0]
                    count += 1
                except Exception as e:
                    print(f'Exception as: {e}')
                    print(f'Skipping (series, iter): {(series, i)}')
        except Exception as e:
            print(f'Exception as {e}')
            print(f'Skipping series = {series}')

    print (x_input.shape)
    print (v_input.shape)
    print (label.shape)
    print (count)

    file = f'train' if train else f'test'
    np.save(save_path / f'{file}_data_{save_path.stem}.npy', x_input)
    np.save(save_path / f'{file}_v_{save_path.stem}.npy', v_input)
    np.save(save_path / f'{file}_label_{save_path.stem}.npy', label)
    return None


def load_data(file, default_base):
    df = feather.read_dataframe(file)
    if file.stem != 'market_data_feat_eng':
        df = df.loc[df['universe'] == 1].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
        df = df.rename({'returnsOpenNextMktres10': 'target'}, axis=1)
        df_tmp = df.pivot_table(values='target', columns='assetCode', index='time')
        df_tmp_codes = ((~df_tmp.isna()).sum().reset_index(drop=False)
                        .sort_values([0], ascending=False)['assetCode'].tolist())
        samples_codes = df_tmp_codes[:500]
        df = df.loc[df['assetCode'].isin(samples_codes)]
        print(f'Number of unique assetCode is {df.assetCode.nunique()}')
        if default_base == False:
            df = proc_df(df)
    df_ret = df.pivot_table(values='target', columns='assetCode', index='time')
    return df_ret, df


def perform_pca(df, train_end, test_start):
    numeric_cols = df.select_dtypes('number').columns.tolist()
    numeric_cols.remove('target')
    mapper = {i: 0.0 for i in numeric_cols}
    df_clean = df.replace([np.inf, -np.inf], np.NaN).fillna(mapper)
    df_train = df_clean.loc[df_clean['time'] < train_end]
    df_test = df_clean.loc[df_clean['time'] >= test_start]
    scaler = StandardScaler()
    train = scaler.fit_transform(df_train[numeric_cols])
    test = scaler.transform(df_test[numeric_cols])
    pca = PCA(n_components=8, svd_solver='auto', whiten=False)
    X_comp_tr = pca.fit_transform(train)
    X_comp_tr = pd.DataFrame(X_comp_tr, columns=[f'f{i}' for i in range(8)])
    X_comp_tr = X_comp_tr.assign(time=df_train['time'], assetCode=df_train['assetCode'])
    X_comp_te = pca.transform(test)
    X_comp_te = pd.DataFrame(X_comp_te, columns=[f'f{i}' for i in range(8)])
    X_comp_te = X_comp_te.assign(time=df_test['time'], assetCode=df_test['assetCode'])
    X_comp = pd.concat([X_comp_tr, X_comp_te], axis=0, ignore_index=True)
    cols = [f'f{i}' for i in range(8)]
    return X_comp, cols


def gen_ts_covariates(df, train_end, test_start, pca=False, cols=None):
    if cols is None:
        cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1']
    if pca:
        df_comp, cols = perform_pca(df, train_end, test_start)
    ## TODO: Add PCA to reduce dimensionality ?
    ts_covariate_train = {}
    ts_covariate_test = {}
    for c in cols:
        if pca:
            val = df_comp.pivot_table(values=f'{c}', columns='assetCode', index='time')
        else:
            val = df.pivot_table(values=f'{c}', columns='assetCode', index='time')
        val_train = val.loc[val.index < train_end].values
        val_test = val.loc[val.index >= test_start].values
        ts_covariate_train[c] = np.nan_to_num(val_train)
        ts_covariate_test[c] = np.nan_to_num(val_test)
    return ts_covariate_train, ts_covariate_test

def rmse(label, preds):
    diff = mean_squared_error(label, preds)
    return diff

def quantile_loss(label, preds, tau = 0.5):
    qval = np.abs(label - preds)
    qsum = np.abs(label)
    loss = qval.sum() / qsum.sum()
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_features', default=0, type=int, help='Generate features for stock dataset')
    parser.add_argument('--default_base', default=1, type=int, help='Use default model without covariates')
    parser.add_argument('--save_directory', default='stock', type=str, help='Directory to save processed files in')
    parser.add_argument('--run_arima', default=0, type = int, help = 'run arima model for baseline')
    args = parser.parse_args()

    assert args.default_base != args.generate_features  # both cannot be the same

    if args.generate_features or args.default_base:
        data_dir = pathlib.Path.cwd() / 'data' / 'market_data.feather'
    else:
        data_dir = pathlib.Path.cwd() / 'data' / 'stockFeatureEng' / 'market_data_feat_eng.feather'

    df_ret, df = load_data(data_dir, args.default_base)

    features = df.columns[df.columns.str.contains('target')].tolist()
    num_feats = df.select_dtypes('number').columns.tolist()
    df.loc[:, num_feats] = df.groupby(['assetCode'])[num_feats].transform(lambda x: x.fillna(method='ffill').fillna(0.0))

    save_path = pathlib.Path.cwd() / 'data' / args.save_directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    given_days = 7

    train_start = '2010-01-01'
    train_end = '2014-12-01'

    test_start = pd.to_datetime(train_end, format='%Y-%m-%d') - pd.DateOffset(n=given_days - 1)
    test_end = '2015-12-30' # save last 1 year for testing
    df_ret = df_ret.loc[(df_ret.index > train_start) & (df_ret.index < test_end)]

    # df_ret = df_ret.loc[df_ret.index > train_start]
    num_covariates = 2
    window_size = 192
    stride_size = 8
    covariates = gen_covariates(df_ret, num_covariates=num_covariates)

    train_ts_covariate, test_ts_covariate = None, None
    if args.default_base == 0:
        train_ts_covariate, test_ts_covariate = gen_ts_covariates(df, train_end, test_start, cols=num_feats, pca=True)

    train_data = df_ret.loc[df_ret.index < train_end].fillna(0.0)
    test_data = df_ret.loc[df_ret.index >= test_start].fillna(0.0)
    data_start = (~df_ret.isna()).values.argmax(axis=0)
    total_time = df_ret.shape[0]
    num_series = df_ret.shape[1]

    if args.run_arima:
        df_all = pd.concat([train_data, test_data], axis = 0)
        cols = df_ret.columns
        results = {}
        forecasts = {}
        arima_log = []
        l = [1,3,5,10,15]
        qlosses = {}
        for t in l:
            for cnt, c in enumerate(cols):
                try:
                    data = df_all[[c]].iloc[:-t]
                    arima_model = ARIMA(data, order=(5, 0, 5))
                    arima_fit = arima_model.fit(disp = False)
                    fcst = arima_fit.forecast(steps = t)
                    realized = df_all[[c]].iloc[-t:]
                    forecasts[c] = (fcst, realized)
                    preds = arima_fit.predict(8)
                    tmp = data.loc[preds.index[0]:, [c]]
                    if cnt < 3:
                        print(-arima_fit.llf / data.shape[0])
                    arima_log.append(-arima_fit.llf / data.shape[0])
                    results[c] = pd.concat([tmp, preds.to_frame('preds')], axis = 1)
                except Exception as e:
                    print(f'Exception as {e}, skipping {c}')
            qloss = {}
            for k, v in forecasts.items():
                qloss[k] = quantile_loss(v[0][0], v[1].values.ravel())

            qlossdf = pd.DataFrame(list(qloss.values()), index=list(qloss.keys()))
            qlosses[t] = qlossdf

            print(f'MAPE for {t} is {qlossdf.mean()}')

        rootmse = {}
        for k, v in forecasts.items():
            rootmse[k] = rmse(v[0][0], v[1].values.ravel())

        qlosses = pd.DataFrame(list(qloss.values()), index = list(qloss.keys()))
        print(f'Weighted Quantile Loss is {qlosses.mean()}')
        rootmse = pd.DataFrame(list(rootmse.values()), index = list(rootmse.keys()))
        print(f'RMSE Loss is {np.sqrt(rootmse.mean())}')
        print(f'NLL loss is {np.mean(arima_log)}')

    # train_data_clean = np.nan_to_num(train_data)
    # test_data_clean = np.nan_to_num(test_data)
    #
    # prep_data(train_data_clean, covariates, data_start, window_size, total_time = total_time,
    #                    stride_size=stride_size, train = True)
    #
    # prep_data(test_data_clean, covariates, data_start, ts_covariates=test_ts_covariate, total_time = total_time,
    #           train=False, window_size=window_size, stride_size=stride_size)
    #
    # print(total_time)
    # print(num_series)
    #
    # print(train_data.shape)
    # print(df_ret.shape)
