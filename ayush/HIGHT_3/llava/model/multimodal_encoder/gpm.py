import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

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

        # Minimal params to run GPM in node task mode
        hidden_dim = getattr(config, 'gin_hidden_dim', 300)
        heads = getattr(config, 'gpm_heads', 4)
        num_layers = getattr(config, 'gpm_layers', 2)
        dropout = getattr(config, 'gpm_dropout', 0.1)
        pattern_size = getattr(config, 'gpm_pattern_size', 5)
        num_patterns = getattr(config, 'gpm_num_patterns', 8)
        pre_sample_bs = getattr(config, 'gpm_pre_sample_batch_size', 2048)
        p = getattr(config, 'gpm_p', 1.0)
        q = getattr(config, 'gpm_q', 1.0)
        codebook_size = getattr(config, 'gpm_codebook_size', 512)

        # Template for params; input/edge dims filled per batch
        self.params_template = {
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
            'node_pe_dim': 8,  # lap_pe has 8 dimensions
            'multiscale': [pattern_size],
            'split': 'inference',
            'dataset': 'mol_graphs'  # Add dataset parameter for molecular graphs
        }

        # Build a dummy model; proper input/edge dims will be set on first forward
        
        self.model = None

        if hasattr(config, 'projection_dim'):
            self.projector = nn.Linear(hidden_dim, config.projection_dim)
            self.output_dim = config.projection_dim
        else:
            self.projector = None
            self.output_dim = hidden_dim
            
        # Add dummy_feature for compatibility
        # Use 308 dimensions (300 from GPM + 8 from lap_pe) to match mm_projector input
        self.dummy_feature = torch.randn(1, 308)
        
        # Initialize GPM model once with correct parameters
        initial_params = dict(self.params_template)
        initial_params['input_dim'] = 16 + initial_params['node_pe_dim']  # atom encoder + node_pe
        initial_params['edge_dim'] = 0  # Disable edge features
        initial_params['device'] = torch.device('cpu')
        self.model = GPMModel(params=initial_params)

    @torch.no_grad()
    def forward(self, mol):
        device = mol.x.device
        # Prepare params for this batch
        params = dict(self.params_template)
        params['device'] = device
        # Set input_dim to match what the pattern encoder expects (16 + node_pe_dim)
        params['input_dim'] = 16 + params['node_pe_dim']  # atom encoder + node_pe
        # Temporarily disable edge features as they're causing issues
        params['edge_dim'] = 0  # 0 if mol.edge_attr is None else mol.edge_attr.size(1)
        
        # Model is already initialized in __init__, no need to recreate it
            
        # Generate random-walk patterns for all nodes
        patterns, eids = get_patterns(mol, params)
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
        # Handle missing batch attribute during training
        if hasattr(mol, 'batch') and mol.batch is not None:
            batch_tensor = mol.batch.to(node_emb.device)
        else:
            # Create batch tensor for single graph (training case)
            batch_tensor = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        graph_emb = global_mean_pool(node_emb, batch_tensor)

        if self.projector is not None:
            graph_emb = self.projector(graph_emb)
            node_emb = self.projector(node_emb)

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
    
    def _load_state_dict(self, checkpoint_path, strict=False):
        """Load state dict method for compatibility with evaluation scripts.
        GPM doesn't use pre-trained checkpoints, so this is a no-op."""
        if checkpoint_path and checkpoint_path != "none":
            print(f"Warning: GPM doesn't use checkpoint loading. Ignoring {checkpoint_path}")
        return {}