from preprocess_stock import *
import json, utils, logging
import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from evaluate import evaluate
from dataloader import *
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_features', default=0, type=int, help='Generate features for stock dataset')
    parser.add_argument('--default_base', default=1, type=int, help='Use default model without covariates')
    parser.add_argument('--save_directory', default='stock', type=str, help='Directory to save processed files in')
    parser.add_argument('--stride_size', default=8, type=int, help='Stride size')
    args = parser.parse_args()

    log_paths = pathlib.Path.cwd() / 'logs'
    if not os.path.exists(log_paths):
        os.makedirs(log_paths)

    utils.set_logger(log_paths / 'data_processing.log')
    logging.info(f'Params are: {args}')

    assert args.default_base != args.generate_features  # both cannot be the same

    if args.generate_features or args.default_base:
        data_dir = pathlib.Path.cwd() / 'data' / 'market_data.feather'
    else:
        data_dir = pathlib.Path.cwd() / 'data' / 'stockFeatureEng' / 'market_data_feat_eng.feather'

    df_ret, df = load_data(data_dir, args.default_base)

    features = df.columns[df.columns.str.contains('target')].tolist()
    num_feats = df.select_dtypes('number').columns.tolist()
    df.loc[:, num_feats] = df.groupby(['assetCode'])[num_feats].transform(
        lambda x: x.fillna(method='ffill').fillna(0.0))

    save_path = pathlib.Path.cwd() / 'data' / args.save_directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    given_days = 7

    train_start = '2010-01-01'
    train_end = '2014-12-01'

    test_start = pd.to_datetime(train_end, format='%Y-%m-%d') - pd.DateOffset(n=max(given_days - 1, 1))
    test_end = '2015-12-30'  # save last 1 year for testing
    df_ret = df_ret.loc[(df_ret.index > train_start) & (df_ret.index < test_end)]

    num_covariates = 2
    window_size = 192
    stride_size = args.stride_size
    covariates = gen_covariates(df_ret, num_covariates=num_covariates)

    train_ts_covariate, test_ts_covariate = None, None
    if args.default_base == 0:
        train_ts_covariate, test_ts_covariate = gen_ts_covariates(df, train_end, test_start, cols=num_feats,
                                                                  pca=True)

    train_data = df_ret.loc[df_ret.index < train_end]
    test_data = df_ret.loc[df_ret.index >= test_start]
    data_start = (~df_ret.isna()).values.argmax(axis=0)
    total_time = df_ret.shape[0]
    num_series = df_ret.shape[1]

    train_data_clean = np.nan_to_num(train_data)
    test_data_clean = np.nan_to_num(test_data)

    prep_data(train_data_clean, covariates, data_start,
              total_time=total_time,
              ts_covariates=train_ts_covariate,
              window_size=window_size, stride_size=stride_size,
              save_path=save_path)

    prep_data(test_data_clean, covariates, data_start,
              total_time=total_time, ts_covariates=test_ts_covariate,
              train=False, window_size=window_size,
              stride_size=stride_size, save_path=save_path)

    print(total_time)
    print(num_series)

    print(train_data.shape)
    print(df_ret.shape)

    params = {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "lstm_layers": 3,
        "num_epochs": 20,
        "train_window": 192,
        "test_window": 192,
        "predict_start": 192 - args.stride_size,
        "test_predict_start": 192 - args.stride_size,
        "predict_steps": args.stride_size,
        "num_class": 500,
        "cov_dim": 2,
        "lstm_hidden_dim": 40,
        "embedding_dim": 20,
        "sample_times": 200,
        "lstm_dropout": 0.1,
        "predict_batch": 256
    }

    check_point_dir = pathlib.Path.cwd() / 'experiments' / f'base_stock_stride={args.stride_size}'
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    with open(check_point_dir / 'params.json', 'w') as file:
        json.dump(params, file)
    file.close()

    logging.info(f'Completed processing data for stride_size = {args.stride_size}')