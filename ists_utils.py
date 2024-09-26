import pickle
from typing import Dict

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

import utils
from utils import generate_batches


def get_X_M_T(X, feat_mask):
    def f(X):
        feat_mask_ = np.array(feat_mask)
        x_arg = feat_mask_ == 0
        m_arg = feat_mask_ == 1
        # T_arg = feat_mask_ == 2
        # X, M, T = X[:, :, x_arg], X[:, :, m_arg], X[:, :, T_arg]
        X, M = X[:, :, x_arg], X[:, :, m_arg]
        X = np.transpose(X, (0, 2, 1))
        M = np.transpose(M, (0, 2, 1))
        # T = np.transpose(T, (0, 2, 1))
        T = []
        return X, M, T

    if len(np.shape(X)) == 3:
        return f(X)
    elif len(np.shape(X)) == 4:
        X_list = X
        X, M, T = [], [], []
        for X_ in X_list:
            x, m, t = f(X_)
            X.append(x)
            M.append(m)
            # T.append(t)
        X = np.concatenate(X, axis=1)
        M = np.concatenate(M, axis=1)
        # T = np.concatenate(T, axis=1)
        return X, M, T


def adapter(X, X_spt, X_exg, feat_mask, E: bool, S: bool):
    x, m, T = get_X_M_T(X, feat_mask)
    x_spt, m_spt, T_spt = get_X_M_T(X_spt, feat_mask)
    x_exg, m_exg, T_exg = get_X_M_T(X_exg, feat_mask)
    # x = np.concatenate([x, x_spt, x_exg], axis=1)
    # m = np.concatenate([m, m_spt, m_exg], axis=1)
    x, m = [x], [m]
    if S:
        x.append(x_spt)
        m.append(m_spt)
    if E:
        x.append(x_exg)
        m.append(m_exg)
    x, m = np.concatenate(x, axis=1), np.concatenate(m, axis=1)
    T = np.zeros_like(x[:, 0:1])
    T = np.apply_along_axis(lambda x: (np.arange(np.shape(x)[-1]) / np.shape(x)[-1]), -1, T)
    return np.transpose(x, (0, 2, 1)), 1 - np.transpose(m, (0, 2, 1)), np.transpose(T, (0, 2, 1))


def load_adapt_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code) -> (Dict[str, np.ndarray], Dict):
    base_path = "../ists/output/pickle"
    conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"
    with open(f'{base_path}/{conf_name}.pickle', 'rb') as f:
        train_test_dict = pickle.load(f)

    D = train_test_dict
    feat_mask = D['x_feat_mask']
    E, S = 'E' in abl_code, 'S' in abl_code
    x_train, m_train, T_train = adapter(D['x_train'], D['spt_train'], D['exg_train'], feat_mask, E, S)
    x_valid, m_valid, T_valid = adapter(D['x_valid'], D['spt_valid'], D['exg_valid'], feat_mask, E, S)
    x_test, m_test, T_test = adapter(D['x_test'], D['spt_test'], D['exg_test'], feat_mask, E, S)
    y_train, y_valid, y_test = D['y_train'], D['y_valid'], D['y_test']
    input_dim = x_train.shape[-1]

    # masked out values are set to 0
    x_train[m_train == 0] = 0
    x_valid[m_valid == 0] = 0
    x_test[m_test == 0] = 0

    x_train = np.concatenate((x_train, m_train, T_train), axis=-1)
    x_valid = np.concatenate((x_valid, m_valid, T_valid), axis=-1)
    x_test = np.concatenate((x_test, m_test, T_test), axis=-1)
    x_train, x_valid, x_test = torch.tensor(x_train).float(), torch.tensor(x_valid).float(), torch.tensor(x_test).float()
    y_train, y_valid, y_test = torch.tensor(y_train).float(), torch.tensor(y_valid).float(), torch.tensor(y_test).float()
    
    X_y_dict = {
        "X_train": x_train, "X_valid": x_valid, "X_test": x_test,
        "y_train": y_train, "y_valid": y_valid, "y_test": y_test,
        # "input_dim": input_dim
    }
    params = {
        "input_dim": input_dim,
        "scalers": D['scalers'],
        "id_array_train": D['id_train'],
        "id_array_valid": D['id_valid'],
        "id_array_test": D['id_test']
    }
    return X_y_dict, params


def get_pretrain_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code, args):
    X_y_dict, params = load_adapt_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code)
    x_train, x_valid = X_y_dict['X_train'], X_y_dict['X_valid']
    
    data_objects = generate_batches(x_train, x_valid, args)

    return data_objects


def get_finetune_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code, args):
    X_y_dict, params = load_adapt_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code)
    D = X_y_dict
    X_train, y_train = D['X_train'], D['y_train']
    X_val, y_val = D['X_valid'], D['y_valid']
    X_test, y_test = D['X_test'], D['y_test']
    input_dim = params['input_dim']

    print('X_train: ' + str(X_train.shape) + ' y_train: ' + str(y_train.shape))
    print('X_val: ' + str(X_val.shape) + ' y_val: ' + str(y_val.shape))
    print('X_test: ' + str(X_test.shape) + ' y_test: ' + str(y_test.shape))

    if args.task == 'classification':
        train_data_combined = TensorDataset(X_train, y_train.long().squeeze())
        val_data_combined = TensorDataset(X_val, y_val.long().squeeze())
        test_data_combined = TensorDataset(X_test, y_test.long().squeeze())
    elif args.task == 'regression' or args.task == 'interpolation':
        train_data_combined = TensorDataset(X_train, y_train.float())
        val_data_combined = TensorDataset(X_val, y_val.float())
        test_data_combined = TensorDataset(X_test, y_test.float())

    train_dataloader = DataLoader(train_data_combined, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data_combined, batch_size=args.batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}

    return data_objects, params


def evaluate_raw(loader, id_array, device, model, dim, scalers):
    y_pred, y_true = utils.predict_regressor(model, loader, device, dim)
    y_pred, y_true = np.reshape(y_pred, (-1, 1)), np.reshape(y_true, (-1, 1))
    y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_true, id_array)])
    y_pred = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_pred, id_array)])
    mse, mae = metrics.mean_squared_error(y_true, y_pred), metrics.mean_absolute_error(y_true, y_pred)
    return mse, mae
