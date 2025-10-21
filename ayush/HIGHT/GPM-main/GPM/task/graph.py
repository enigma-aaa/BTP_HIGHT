import yaml
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.linkproppred import *
from ogb.nodeproppred import *

from model.random_walk import get_patterns_for_graph

from utils.eval import evaluate
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def multitask_regression(y_pred, y, metric='rmse'):
    if metric == 'rmse':
        criterion = nn.MSELoss(reduction="none")
    elif metric == 'mae':
        criterion = nn.L1Loss(reduction="none")

    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred, exist_y)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def preprocess_graph(datasets, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    pattern_dir = osp.join(params['pattern_path'], params['dataset'])
    pattern_dict = {}

    if isinstance(datasets, dict):
        for key, subset in datasets.items():
            cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}", key)
            check_path(cur_dir)

            pattern_path = osp.join(cur_dir, f"ptn.pt")
            nid_path = osp.join(cur_dir, f"nid.pt")
            eid_path = osp.join(cur_dir, f"eid.pt")

            if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
                patterns = torch.load(pattern_path)
                nids = torch.load(nid_path)
                eids = torch.load(eid_path)
            else:
                patterns, nids, eids = get_patterns_for_graph(subset, params)
                torch.save(patterns, pattern_path)
                torch.save(nids, nid_path)
                torch.save(eids, eid_path)

            pattern_dict[key] = {'pattern': patterns, 'nid': nids, 'eid': eids}
    else:
        cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
        check_path(cur_dir)

        pattern_path = osp.join(cur_dir, f"ptn.pt")
        nid_path = osp.join(cur_dir, f"nid.pt")
        eid_path = osp.join(cur_dir, f"eid.pt")

        if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, weights_only=True)
            nids = torch.load(nid_path, weights_only=True)
            eids = torch.load(eid_path, weights_only=True)

        else:
            patterns, nids, eids = get_patterns_for_graph(datasets, params)
            torch.save(patterns, pattern_path)
            torch.save(nids, nid_path)
            torch.save(eids, eid_path)

        pattern_dict = {'pattern': patterns, 'nid': nids, 'eid': eids}

    return pattern_dict


def train_graph(dataset, model, optimizer, split=None, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0

    if isinstance(split, int):
        dataset = dataset['train'] if params['split'] != 'pretrain' else dataset['full']
        num_graphs = len(dataset)
        graphs = torch.arange(num_graphs)
    else:
        dataset = dataset
        graphs = mask2idx(split['train'])
        num_graphs = len(graphs)

    y = dataset.y

    # We do batch training by default
    num_batches = (num_graphs + bs - 1) // bs
    train_perm = torch.randperm(num_graphs)

    for i in range(num_batches):
        cur_graphs = graphs[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_graphs].to(device)

        pred, instance_emb, pattern_emb, commit_loss = model(dataset, cur_graphs, params, mode='train')
        if y.ndim == 1:
            if params['metric'] == 'rmse':
                loss = F.mse_loss(pred.squeeze(), cur_y.float())
            elif params['metric'] == 'mae':
                loss = F.l1_loss(pred.squeeze(), cur_y.float())
            else:
                loss = F.cross_entropy(pred, cur_y, label_smoothing=params['label_smoothing'])
        else:
            if params['metric'] in ['rmse', 'mae']:
                loss = multitask_regression(pred, cur_y.float(), metric=params['metric'])
            else:
                loss = multitask_cross_entropy(pred, cur_y)

        loss = loss + commit_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= num_batches
    total_val_loss /= num_batches
    total_test_loss /= num_batches

    return {'train': total_loss, 'val': total_val_loss, 'test': total_test_loss}


def eval_graph(graph, model, split=None, params=None):
    model.eval()
    bs = params['batch_size']

    results = {}
    results['metric'] = params['metric']
    results['train'] = 0

    with torch.no_grad():
        for key in ['val', 'test']:
            if isinstance(split, int):
                dataset = graph[key]
                num_graphs = len(dataset)
                graphs = torch.arange(num_graphs)
            else:
                dataset = graph
                graphs = mask2idx(split[key])
                num_graphs = len(graphs)

            y = dataset.y[graphs]

            num_batches = (num_graphs + bs - 1) // bs

            pred_list = []
            for i in range(num_batches):
                cur_graphs = graphs[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(dataset, cur_graphs, params, mode=key)
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)

            value = evaluate(pred, y, params=params)
            results[key] = value

    return results
