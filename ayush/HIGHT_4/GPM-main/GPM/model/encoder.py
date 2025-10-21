import copy
import os.path as osp
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

# Custom encoders for continuous/discrete features
class CustomAtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        # Use linear layers instead of embedding for continuous features
        self.linear = torch.nn.Linear(2, emb_dim)  # 2 input features
        
    def forward(self, x):
        return self.linear(x.float())

class CustomBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        # Use linear layers instead of embedding for continuous features
        self.linear = torch.nn.Linear(2, emb_dim)  # 2 input features
        
    def forward(self, x):
        return self.linear(x.float())

try:
    from data.pyg_data_loader import mol_graphs
except Exception:
    # Fallback to molecular graph datasets when data loaders are unavailable in embedded usage
    mol_graphs = ['esol', 'freesolv', 'lipo', 'bace', 'bbbp', 'clintox', 'hiv', 'tox21', 'toxcast', 'muv', 'pcba', 'sider',
                  'qm7', 'qm8', 'qm9', 'zinc', 'pcqm4m', 'pcqm4m-v2', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-molbace',
                  'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molmuv', 'ogbg-molpcba', 'ogbg-molsider', 'ogbg-moltox21',
                  'ogbg-moltoxcast', 'ogbg-molfreesolv', 'ogbg-mollipo', 'ogbg-molesol', 'mol_graphs']
from utils.eval import *
from utils.utils import get_device_from_model, check_path


class PatternEncoder(nn.Module):
    def __init__(self, params):
        super(PatternEncoder, self).__init__()
        self.hidden_dim = params['hidden_dim']
        self.num_heads = params['pattern_encoder_heads']
        self.num_layers = params['pattern_encoder_layers']
        self.pe_encoder = params['pe_encoder']
        self.pattern_encoder = params['pattern_encoder']

        # Calculate input dimension accounting for encoders
        if params['dataset'] in mol_graphs:
            # Use custom encoders for our specific features
            self.atom_encoder = CustomAtomEncoder(emb_dim=16)
            self.bond_encoder = CustomBondEncoder(emb_dim=16)
            # Calculate input_dim based on whether edge features are available
            if params.get('edge_dim', 0) > 0:
                self.input_dim = 16 + 16 + params['node_pe_dim']  # atom + bond + node_pe
            else:
                self.input_dim = 16 + params['node_pe_dim']  # atom + node_pe only
        else:
            self.input_dim = params['input_dim'] + params['edge_dim'] + params['node_pe_dim']

        self.pre_projection = nn.Linear(self.input_dim, self.hidden_dim)

        if params['pe_encoder'] in ['mean', 'gru']:
            self.pe_dim = params['pattern_size'] + 1
        else:
            self.pe_dim = 0

        self._init_encoder()
        self._init_pe_encoder()

    def _init_encoder(self):
        # Initialize pattern encoder
        if self.pattern_encoder == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.projection = nn.Identity()  # No projection needed for transformer

        elif self.pattern_encoder == "gru":
            self.rnn = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
            self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.pattern_encoder == "mean":
            self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        else:
            raise ValueError(f"Unsupported pattern encoder: {self.pattern_encoder}")

    def _init_pe_encoder(self):
        if self.pe_encoder == "mean":
            self.pe_projection = nn.Linear(self.pe_dim, self.hidden_dim)
        elif self.pe_encoder == "gru":
            self.pe_rnn = nn.GRU(
                input_size=self.pe_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.pe_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.pe_encoder == "none":
            pass
        else:
            raise ValueError(f"Unsupported PE encoder: {self.pe_encoder}")

    def _encode_features(self, feat_gathered, mask=None):
        if self.pattern_encoder == "transformer":
            if mask is not None:
                mask = ~mask
            else:
                mask = None
            embeddings = self.encoder(feat_gathered, src_key_padding_mask=mask)
            embeddings = embeddings.mean(dim=1)  # Aggregate along the sequence

        elif self.pattern_encoder == "gru":
            # feat_gathered = feat_gathered.flip(1)  # Reverse sequence for RNN processing
            embeddings, _ = self.rnn(feat_gathered)  # Shape: [batch_size, k, hidden_dim]
            if mask is not None:
                embeddings = embeddings * mask.unsqueeze(-1)  # Shape: [batch_size, k, hidden_dim]
                mask_sum = mask.sum(dim=1, keepdim=True)
                # Avoid division by zero
                mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
                embeddings = self.projection(embeddings.sum(dim=1) / mask_sum)
            else:
                embeddings = self.projection(embeddings.mean(dim=1))  # Project to hidden_dim

        elif self.pattern_encoder == "mean":
            if mask is not None:
                mask = mask.unsqueeze(-1)
                masked_feat = feat_gathered * mask  # Shape: [batch_size, k, input_dim]
                mask_sum = mask.sum(dim=1)
                # Avoid division by zero
                mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
                embeddings = self.projection(masked_feat.sum(dim=1) / mask_sum)
            else:
                embeddings = self.projection(feat_gathered.mean(dim=1))  # Aggregate to hidden_dim

        return embeddings

    def _encode_pe(self, patterns):
        adj = patterns.unsqueeze(-1) == patterns.unsqueeze(-2)
        adj = adj.view(-1, self.pe_dim, self.pe_dim).float()

        if self.pe_encoder == "mean":
            pe = self.pe_projection(adj.mean(dim=1))
        elif self.pe_encoder == "gru":
            pe, _ = self.pe_rnn(adj)
            pe = self.pe_projection(pe.mean(dim=1))
        else:
            raise ValueError(f"Unsupported PE encoder: {self.pe_encoder}")

        return pe

    def encode_node(self, patterns, feat, node_pe, e_feat=None, eids=None, params=None):
        device = get_device_from_model(self)
        h, n, k = patterns.shape

        # Flatten patterns and gather corresponding features
        patterns_flat = patterns.view(-1)  # Shape: [h * n * k]
        feat_gathered = feat[patterns_flat].to(device)  # Shape: [h * n * k, d]
        
        # Apply atom encoder if available
        if hasattr(self, 'atom_encoder'):
            feat_gathered = self.atom_encoder(feat_gathered)
        
        feat_gathered = feat_gathered.view(h * n, k, -1)  # Reshape to [h * n, k, d]

        if node_pe is not None:
            node_pe_gathered = node_pe[patterns_flat].to(device)  # Shape: [h * n * k, d]
            node_pe_gathered = node_pe_gathered.view(h * n, k, -1)  # Reshape to [h * n, k, d]
            feat_gathered = torch.cat([feat_gathered, node_pe_gathered], dim=-1)

        # Temporarily disable edge features to test if the model works
        # if eids is not None and e_feat is not None:
        #     ed = e_feat.shape[-1]
        #     eids_flat = eids.view(-1).to(device)
        #     e_feat = e_feat.to(device)
        #     
        #     # Simplified edge feature processing like HVQAE2
        #     # Create edge embeddings directly without complex gathering
        #     if hasattr(self, 'bond_encoder'):
        #         # Apply bond encoder to all edge features first
        #         e_feat_encoded = self.bond_encoder(e_feat)
        #         ed = e_feat_encoded.shape[-1]
        #         
        #         # Create edge embeddings for patterns
        #         h, n, k = patterns.shape
        #         e_feat_gathered = torch.zeros(h * n, k, ed, device=device, dtype=e_feat_encoded.dtype)
        #         
        #         # Fill in valid edge features
        #         valid_mask = (eids_flat >= 0) & (eids_flat < e_feat.shape[0])
        #         if valid_mask.any():
        #             valid_eids = eids_flat[valid_mask]
        #             valid_positions = torch.where(valid_mask)[0]
        #             
        #             # Map valid positions to their corresponding pattern positions
        #             for i, pos in enumerate(valid_positions):
        #                 pattern_idx = pos // (k - 1)
        #                 edge_idx = pos % (k - 1)
        #                 if pattern_idx < h * n and edge_idx < k - 1:
        #                     e_feat_gathered[pattern_idx, edge_idx + 1, :] = e_feat_encoded[valid_eids[i]]
        #     else:
        #         # No bond encoder, use original edge features
        #         h, n, k = patterns.shape
        #         e_feat_gathered = torch.zeros(h * n, k, ed, device=device, dtype=e_feat.dtype)
        #         
        #         # Fill in valid edge features
        #         valid_mask = (eids_flat >= 0) & (eids_flat < e_feat.shape[0])
        #         if valid_mask.any():
        #             valid_eids = eids_flat[valid_mask]
        #             valid_positions = torch.where(valid_mask)[0]
        #             
        #             # Map valid positions to their corresponding pattern positions
        #             for i, pos in enumerate(valid_positions):
        #                 pattern_idx = pos // (k - 1)
        #                 edge_idx = pos % (k - 1)
        #                 if pattern_idx < h * n and edge_idx < k - 1:
        #                     e_feat_gathered[pattern_idx, edge_idx + 1, :] = e_feat[valid_eids[i]]
        # 
        #     feat_gathered = torch.cat([feat_gathered, e_feat_gathered], dim=-1)

        feat_gathered = self.pre_projection(feat_gathered)

        multiscale = params['multiscale']
        if len(multiscale) > 1:
            mask = torch.zeros(h, n, k, device=device).bool()
            for i, scale in enumerate(multiscale):
                start = int(i * h / len(multiscale))
                mask[start:, :, :scale + 1] = True
            mask = mask.view(h * n, k)
        else:
            mask = None

        pattern_feat = self._encode_features(feat_gathered, mask)
        pattern_feat = pattern_feat.view(h, n, self.hidden_dim)

        if self.pe_encoder != 'none':
            pe = self._encode_pe(patterns.to(device))
            pe = pe.view(h, n, -1)
            pattern_feat = pattern_feat + pe * params['pe_weight']

        return pattern_feat

    def encode_graph(self, nids, feat, patterns, eids=None, e_feat=None, node_pe=None, params=None):
        device = get_device_from_model(self)

        if params['dataset'] in mol_graphs:
            feat = self.atom_encoder(feat)
            if e_feat is not None:
                e_feat = self.bond_encoder(e_feat)

        h, n, k = patterns.shape

        # Flatten patterns and gather corresponding features
        nids_flat = nids.view(-1)  # Shape: [h * n * k]
        feat_gathered = feat[nids_flat]  # Shape: [h * n * k, d]
        feat_gathered = feat_gathered.view(h * n, k, -1)  # Reshape to [h * n, k, d]

        if node_pe is not None:
            _, max_nodes, d = node_pe.shape

            flat_patterns = patterns.permute(1, 0, 2).reshape(n, -1)  # Shape: [n, h * k]
            node_pe_gathered = node_pe[torch.arange(n)[:, None], flat_patterns]  # Shape: [n, h * k, d]
            node_pe_gathered = node_pe_gathered.view(n, h, k, d).permute(1, 0, 2, 3)  # Shape: [h, n, k, d]
            node_pe_gathered = node_pe_gathered.reshape(n * h, k, d).to(device)
            feat_gathered = torch.cat([feat_gathered, node_pe_gathered], dim=-1)

        if eids is not None:
            em, ed = e_feat.shape
            eids_flat = eids.view(-1)
            
            # Filter out invalid edge indices (-1) to prevent CUDA indexing errors
            valid_mask = eids_flat >= 0
            if valid_mask.any():
                eids_clean = torch.where(valid_mask, eids_flat, torch.zeros_like(eids_flat))
                e_feat_gathered = e_feat[eids_clean]
                e_feat_gathered = e_feat_gathered.view(h * n, k - 1, ed)
                # Zero out features for invalid edges
                valid_mask_reshaped = valid_mask.view(h * n, k - 1, 1).float()
                e_feat_gathered = e_feat_gathered * valid_mask_reshaped
                e_feat_gathered = torch.cat([torch.zeros(h * n, 1, ed, device=device), e_feat_gathered], dim=1)
            else:
                # All edges are invalid, create zero features
                e_feat_gathered = torch.zeros(h * n, k, ed, device=device, dtype=e_feat.dtype)

            feat_gathered = torch.cat([feat_gathered, e_feat_gathered], dim=-1)

        feat_gathered = self.pre_projection(feat_gathered)

        multiscale = params['multiscale']
        if len(multiscale) > 1:
            mask = torch.zeros(h, n, k, device=device).bool()
            for i, scale in enumerate(multiscale):
                start = int(i * h / len(multiscale))
                mask[start:, :, :scale + 1] = True
            mask = mask.view(h * n, k)
        else:
            mask = None

        pattern_feat = self._encode_features(feat_gathered, mask)
        pattern_feat = pattern_feat.view(h, n, self.hidden_dim)

        if self.pe_encoder != 'none':
            pe = self._encode_pe(patterns)
            pe = pe.view(h, n, -1)
            pattern_feat = pattern_feat + pe * params['pe_weight']

        return pattern_feat
