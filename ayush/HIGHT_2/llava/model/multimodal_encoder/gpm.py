import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import time
from datetime import datetime

# Ensure GPM package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GPM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '../../../GPM-main'))
_GPM_PKG = os.path.join(_GPM_ROOT, 'GPM')
for p in (_GPM_ROOT, _GPM_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from GPM.model.model import Model as GPMModel  # type: ignore
from GPM.model.random_walk import get_patterns  # type: ignore


class GPMTower(nn.Module):
    def __init__(self, config, use_graph_rep=False, name="gpm"):
        super().__init__()
        self.use_graph_rep = use_graph_rep
        self.name = name
        
        # Timing stats
        self.pattern_gen_times = []
        self.forward_times = []
        self.total_nodes_processed = 0
        self.total_patterns_generated = 0

        # Minimal params to run GPM in node task mode
        hidden_dim = int(os.getenv('GPM_HIDDEN_DIM', getattr(config, 'gin_hidden_dim', 300)))
        heads = int(os.getenv('GPM_HEADS', getattr(config, 'gpm_heads', 4)))
        num_layers = int(os.getenv('GPM_LAYERS', getattr(config, 'gpm_layers', 2)))
        dropout = float(os.getenv('GPM_DROPOUT', getattr(config, 'gpm_dropout', 0.1)))
        pattern_size = int(os.getenv('GPM_PATTERN_SIZE', getattr(config, 'gpm_pattern_size', 5)))
        num_patterns = int(os.getenv('GPM_NUM_PATTERNS', getattr(config, 'gpm_num_patterns', 8)))
        pre_sample_bs = int(os.getenv('GPM_PRE_SAMPLE_BS', getattr(config, 'gpm_pre_sample_batch_size', 2048)))
        p = float(os.getenv('GPM_RW_P', getattr(config, 'gpm_p', 1.0)))
        q = float(os.getenv('GPM_RW_Q', getattr(config, 'gpm_q', 1.0)))
        codebook_size = getattr(config, 'gpm_codebook_size', 512)

        # Template for params; input/edge dims filled per batch
        self.params_template = {
            'dataset': 'custom',
            'task': 'node',
            'input_dim': None,
            'edge_dim': None,
            'output_dim': hidden_dim,  # we only need representations, not classification
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'heads': heads,
            'dropout': dropout,
            'norm_first': True,
            'use_attn_fusion': False,
            'use_cls_token': False,
            'use_vq': False,
            'codebook_size': codebook_size,
            'pattern_encoder_heads': heads,
            'pattern_encoder_layers': 2,
            'pe_encoder': 'none',
            'pattern_encoder': 'mean',
            'pattern_size': pattern_size,
            'num_patterns': num_patterns,
            'pre_sample_pattern_num': num_patterns,
            'pre_sample_batch_size': pre_sample_bs,
            'p': p,
            'q': q,
            'node_pe_dim': 0,
            'multiscale': [pattern_size],
            'split': 'inference'
        }

        # Lazily created GPM model when first called
        self.model = None

        if hasattr(config, 'projection_dim'):
            self.projector = nn.Linear(hidden_dim, config.projection_dim)
            self.output_dim = config.projection_dim
        else:
            self.projector = None
            self.output_dim = hidden_dim

    @torch.no_grad()
    def forward(self, mol):
        t_total_start = time.time()
        
        device = mol.x.device
        num_nodes = mol.x.size(0)
        num_edges = mol.edge_index.size(1)
        
        # Prepare params for this batch
        params = dict(self.params_template)
        params['device'] = device
        params['input_dim'] = mol.x.size(1)
        params['edge_dim'] = 0 if mol.edge_attr is None else mol.edge_attr.size(1)

        # Instantiate internal GPM model if needed or if dims changed
        if (self.model is None or
            getattr(self.model, 'input_dim', None) != params['input_dim'] + params['edge_dim'] + params['node_pe_dim'] or
            getattr(self.model, 'hidden_dim', None) != params['hidden_dim']):
            self.model = GPMModel(params=params)

        # Generate random-walk patterns for all nodes
        t_pattern_start = time.time()
        patterns, eids = get_patterns(mol, params)
        pattern_gen_time = time.time() - t_pattern_start
        self.pattern_gen_times.append(pattern_gen_time)
        self.total_patterns_generated += patterns.numel()
        self.total_nodes_processed += num_nodes
        params['pattern_set'] = {
            'pattern': patterns.to(device),
            'eid': eids.to(device),
        }

        # Select all nodes
        num_nodes = mol.x.size(0)
        nodes = torch.arange(num_nodes, device=device)

        # Encode node representations via GPM node task
        _pred, node_emb, _pattern_emb, _commit = self.model.encode_node(mol, nodes, params, mode='eval')

        # Pool to graph representation using batch vector
        graph_emb = global_mean_pool(node_emb, mol.batch)

        if self.projector is not None:
            graph_emb = self.projector(graph_emb)
            node_emb = self.projector(node_emb)

        # Log timing
        total_time = time.time() - t_total_start
        self.forward_times.append(total_time)
        
        # Print timing stats periodically (every 100 calls)
        if len(self.forward_times) % 100 == 0:
            avg_pattern = sum(self.pattern_gen_times[-100:]) / 100
            avg_total = sum(self.forward_times[-100:]) / 100
            avg_nodes = self.total_nodes_processed / len(self.forward_times)
            print(f"[GPM TIMING] Last 100 graphs: Pattern gen={avg_pattern*1000:.2f}ms, Total={avg_total*1000:.2f}ms, Avg nodes={avg_nodes:.1f}")

        return graph_emb, node_emb

    def encode_mol(self, mol, proj=False, return_node_feats=False):
        # Ensure model is initialized via forward if needed
        if self.model is None:
            graph_emb, node_emb = self.forward(mol)
        else:
            self.model.eval()
            graph_emb, node_emb = self.forward(mol)
        if proj and self.projector is not None:
            graph_emb = self.projector(graph_emb)
            node_emb = self.projector(node_emb)
        if return_node_feats:
            return graph_emb, node_emb
        else:
            return graph_emb

    @property
    def hidden_size(self):
        return self.output_dim

    # Optional compatibility for eval scripts expecting this method
    def _load_state_dict(self, model_file, strict=False):
        if model_file in [None, "none", "None", "", "NONE"]:
            return
        try:
            state = torch.load(model_file, map_location="cpu")
            if self.model is not None:
                try:
                    self.model.load_state_dict(state, strict=strict)
                    return
                except Exception:
                    pass
        except Exception:
            pass
