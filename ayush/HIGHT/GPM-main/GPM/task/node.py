import yaml
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.linkproppred import *
from ogb.nodeproppred import *

from model.random_walk import get_patterns

from utils.eval import evaluate
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    loss = 0.0
    for idx in range(y.shape[1]):
        cur_y = y[:, idx]
        cur_pred = y_pred[:, idx]
        task_loss = criterion(cur_pred.double(), cur_y)
        loss += torch.mean(task_loss)

    return loss / y.shape[1]


def preprocess_node(graph, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    if isinstance(graph, dict):
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        pattern_dict = {}
        for key, g_set in graph.items():
            cur_dir = osp.join(pattern_dir, key, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
            check_path(cur_dir)
            pattern_dict[key] = {}

            for i, g in enumerate(g_set):
                pattern_path = osp.join(cur_dir, f"ptn_{i}.pt")
                eid_path = osp.join(cur_dir, f"eid_{i}.pt")
                if osp.exists(pattern_path) and osp.exists(eid_path):
                    patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
                    eids = torch.load(eid_path, map_location=torch.device('cpu'))
                else:
                    patterns, eids = get_patterns(g, params)
                    torch.save(patterns, pattern_path)
                    torch.save(eids, eid_path)
                pattern_dict[key][i] = {'pattern': patterns, 'eid': eids}
        return pattern_dict
    else:
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        check_path(pattern_dir)

        pattern_path = osp.join(pattern_dir, f"ptn_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        eid_path = osp.join(pattern_dir, f"eid_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        if osp.exists(pattern_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
            eids = torch.load(eid_path, map_location=torch.device('cpu'))
            print('Done loading patterns from cache.')
        else:
            patterns, eids = get_patterns(graph, params)
            torch.save(patterns, pattern_path)
            torch.save(eids, eid_path)

        return {'pattern': patterns, 'eid': eids}


def train_node(graph, model, optimizer, split, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0
    nodes = torch.arange(graph.num_nodes)
    y = graph.y
    if y.ndim == 2:
        y = y.squeeze()

    if split is not None:
        train_mask = split["train"]
    else:
        train_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    train_nodes = nodes[train_mask]
    train_num_nodes = train_nodes.size(0)
    train_num_batches = (train_num_nodes + bs - 1) // bs
    train_perm = torch.randperm(train_num_nodes)

    for i in range(train_num_batches):
        cur_nodes = train_nodes[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_nodes].to(device)

        pred, instance_emb, pattern_emb, commit_loss = model(graph, cur_nodes, params, mode='train')
        if params.get('num_tasks') is not None:
            num_tasks = params['num_tasks']
            if num_tasks == 1:
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(), cur_y.float())
            else:
                # loss = multitask_cross_entropy(pred, cur_y)
                loss = F.binary_cross_entropy_with_logits(pred, cur_y.float())
        else:
            loss = F.cross_entropy(pred, cur_y, label_smoothing=params['label_smoothing'])
        loss = loss + commit_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= train_num_batches
    total_val_loss /= train_num_batches
    total_test_loss /= train_num_batches

    return {'train': total_loss, 'val': total_val_loss, 'test': total_test_loss}


def eval_node(graph, model, split, params):
    model.eval()
    device = get_device_from_model(model)

    bs = params['batch_size']
    results = {'train': 0, 'metric': params['metric']}

    with torch.no_grad():
        for key in ['val', 'test']:
            mask = split[key]
            idx = mask2idx(mask)
            y = graph.y[idx].to(device)
            if y.ndim == 2:
                y = y.squeeze()

            num_batches = (len(idx) + bs - 1) // bs
            pred_list = []

            for i in range(num_batches):
                cur_nodes = idx[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(graph, cur_nodes, params, mode='eval')
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)

            results[key] = evaluate(pred, y, params=params)
    return results
