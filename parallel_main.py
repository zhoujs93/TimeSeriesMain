import ray
ray.init()
from preprocess_stock import prep_data, load_data, gen_covariates, gen_ts_covariates
import pandas as pd
import json, utils, logging
import argparse, logging, os
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

import matplotlib, pathlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prepare_data_main(strides, arg):

    log_paths = pathlib.Path.cwd() / 'logs'
    if not os.path.exists(log_paths):
        os.makedirs(log_paths)

    # utils.set_logger(log_paths / 'data_processing.log')
    # logging.info(f'Params are: {args}')


    # assert args.default_base != args.generate_features  # both cannot be the same

    if arg['generate_features'] or arg['default_base']:
        data_dir = pathlib.Path.cwd() / 'data' / 'market_data.feather'
    else:
        data_dir = pathlib.Path.cwd() / 'data' / 'stockFeatureEng' / 'market_data_feat_eng.feather'

    df_ret, df = load_data(data_dir, arg['default_base'])

    features = df.columns[df.columns.str.contains('target')].tolist()
    num_feats = df.select_dtypes('number').columns.tolist()
    df.loc[:, num_feats] = df.groupby(['assetCode'])[num_feats].transform(lambda x: x.fillna(method='ffill').fillna(0.0))

    save_path = pathlib.Path.cwd() / 'data' / arg['save_directory']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    given_days = 7

    train_start = '2010-01-01'
    train_end = '2014-12-01'

    test_start = pd.to_datetime(train_end, format='%Y-%m-%d') - pd.DateOffset(n=max(given_days - 1, 1))
    test_end = '2015-12-30' # save last 1 year for testing
    df_ret = df_ret.loc[(df_ret.index > train_start) & (df_ret.index < test_end)]

    num_covariates = 2
    window_size = 192
    stride_size = strides
    covariates = gen_covariates(df_ret, num_covariates=num_covariates)

    train_ts_covariate, test_ts_covariate = None, None
    if arg['default_base'] == 0:
        train_ts_covariate, test_ts_covariate = gen_ts_covariates(df, train_end, test_start, cols=num_feats,
                                                                  pca=True)

    train_data = df_ret.loc[df_ret.index < train_end].values
    test_data = df_ret.loc[df_ret.index >= test_start].values
    data_start = (~df_ret.isna()).values.argmax(axis=0)
    total_time = df_ret.shape[0]
    num_series = df_ret.shape[1]

    train_data_clean = np.nan_to_num(train_data)
    test_data_clean = np.nan_to_num(test_data)

    data_train, v_train, label_train = prep_data(train_data_clean, covariates, data_start,
                                                 total_time = total_time,
                                                 ts_covariates=train_ts_covariate,
                                                 window_size=window_size, stride_size=stride_size,
                                                 save_path=save_path, save = False)

    data_test, v_test, label_test = prep_data(test_data_clean, covariates, data_start,
                                              total_time = total_time, ts_covariates=test_ts_covariate,
                                              train=False, window_size=window_size,
                                              stride_size=stride_size, save_path=save_path, save = False)
    train_all = (data_train, v_train, label_train)
    test_all = (data_test, v_test, label_test)

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
        "predict_start": 192 - stride_size,
        "test_predict_start": 192 - stride_size,
        "predict_steps": stride_size,
        "num_class": 500,
        "cov_dim": 2,
        "lstm_hidden_dim": 40,
        "embedding_dim": 20,
        "sample_times": 200,
        "lstm_dropout": 0.1,
        "predict_batch": 256
    }

    check_point_dir = pathlib.Path.cwd() / 'experiments' / f'base_stock_stride={stride_size}'
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    with open(check_point_dir / 'params.json', 'w') as file:
        json.dump(params, file)
    file.close()

    print(f'Completed processing data for stride_size = {stride_size}')

    return train_all, test_all


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int,
          arg) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        if i % 1000 == 0:
            test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=arg['sampling'])
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       restore_file: str = None,
                       arg = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    best_test_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, epoch, arg)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=arg['sampling'])
        ND_summary[epoch] = test_metrics['ND']
        is_best = ND_summary[epoch] <= best_test_ND

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best ND is: %.5f' % best_test_ND)

        utils.plot_all_epoch(ND_summary[:epoch + 1], arg['dataset'] + '_ND', params.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], arg['dataset'] + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

    if arg['save_best']:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = arg['search_params'].split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()
        utils.plot_all_epoch(ND_summary, print_params + '_ND', location=params.plot_dir)
        utils.plot_all_epoch(loss_summary, print_params + '_loss', location=params.plot_dir)

@ray.remote(num_gpus=0.2)
def main(stride):
    logger = logging.getLogger('DeepAR.Train')

    arg = {'model_name' : f'base_stock_stride={stride}',
           'data_folder' : 'data',
           'dataset': 'stock',
           'relative_metrics' : 0,
           'sampling' : 0,
           'restore_file' : None,
           'save_best' : 0,
           'generate_features' : 0,
           'default_base' : 1,
           'save_directory' : 'stock',
           'stride_size' : 8
           }

    train_files, test_files = prepare_data_main(stride, arg)

    model_dir = os.path.join('experiments', arg['model_name'])
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(arg['data_folder'], arg['dataset'])
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = arg['relative_metrics']
    params.sampling =  arg['sampling']
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary

    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    np.random.seed(777)

    logger.info('Loading the datasets...')

    train_set = TrainDataset(data_dir, arg['dataset'], params.num_class, data = train_files[0], label = train_files[-1])
    test_set = TestDataset(data_dir, arg['dataset'], params.num_class, data = test_files[0],
                           v = test_files[1], label = test_files[-1])
    sampler = WeightedSampler(data_dir, arg['dataset'], v = train_files[1]) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = net.loss_fn

    # Train the model
    logger.info('Starting training for {} epoch(s) with stride_size {}'.format(params.num_epochs,
                                                                               stride))
    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       params,
                       arg['restore_file'],
                       arg)

    logger.info(f'Finished processing {stride}')
    return True

if __name__ == '__main__':
    strides = [1, 3, 5, 10, 15]

    result = ray.get([main.remote(s) for s in strides])