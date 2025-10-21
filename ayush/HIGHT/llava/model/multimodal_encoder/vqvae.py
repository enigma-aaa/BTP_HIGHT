# refer: https://github.com/chao1224/GraphMVP/blob/main/src_classification/models/molecule_gnn_model.py

import logging
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add, scatter_mean
num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)
        
        h_list = [x]
        for layer in range(self.num_layer):
            # print(torch.isnan(edge_index).sum(),torch.isnan(edge_attr).sum())
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = torch.nan_to_num(h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
            if torch.isnan(h).sum():
                print(torch.isnan(h).sum())
                exit()
        # exit()
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(
        self, 
        emb_dim,  
        graph_pooling, 
        projection_dim:int=None,
        molecule_node_model=None,
        init_checkpoint=None,
    ):
        super(GNN_graphpred, self).__init__()

        self.gnns = molecule_node_model
        self.emb_dim = emb_dim

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        if projection_dim is not None:
            self.projector = nn.Linear(emb_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projector = None
            self.output_dim = emb_dim
        
        if init_checkpoint is not None:
            self._load_state_dict(init_checkpoint, strict=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnns(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        return graph_representation, node_representation
    
    def encode_mol(self, mol, proj=False, return_node_feats=False, eval=True):
        if eval:
            self.gnns.eval() # hard code: set to eval mode
            with torch.no_grad():
                h_graph, h_node = self.forward(mol)
        else:
            self.gnns.train() # set to train mode
            h_graph, h_node = self.forward(mol)
        if proj and self.projector is not None:
            h_graph = self.projector(h_graph)
            h_node = self.projector(h_node)
        if return_node_feats:
            return h_graph, h_node
        else:
            return h_graph
    
    def _load_state_dict(self, model_file, strict=False):
        print("Loading from {} ...".format(model_file))
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        print(self.load_state_dict(state_dict, strict=strict))
        print("Loaded vqvae done")
        return
    
    @property
    def dummy_feature(self):
        return self.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def hidden_size(self):
        return self.output_dim

class DiscreteGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_tokens, JK = "last", 
        temperature = 0.9, drop_ratio = 0, gnn_type = "gin", use_graph_agg=True):
        super(DiscreteGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.num_tokens = num_tokens
        # self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.out_dim=emb_dim
        # self.loss_weight = loss_weight
        # self.codebook = nn.Embedding(num_tokens, emb_dim)
        self.use_graph_agg = use_graph_agg
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim,aggr = "add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))
        self.gnns.append(GINConv(emb_dim, emb_dim,aggr = "add"))    

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))        

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        if self.use_graph_agg:
            # graph_agg = scatter_mean(node_representation, data.batch, dim=0, dim_size=data.num_graphs)
            # graph_agg = graph_agg.unsqueeze(1)  # [bz, 1, feat_size]
            graph_agg = global_mean_pool(node_representation,data.batch)
            # print("hahahaha")
            graph_agg = torch.zeros(graph_agg.size(),dtype=graph_agg.dtype).to(graph_agg.device)
            return graph_agg, node_representation

        return node_representation

    @torch.no_grad()
    def get_codebook_indices(self, *argv):
        logits = self(*argv)[-1]
        codebook_indices = logits.argmax(dim = -1)
        print(logits.size(),codebook_indices.size())
        return codebook_indices

    def from_pretrained(self, model_file,device):
        print("Loading from {} ...".format(model_file))
        print(self.load_state_dict(torch.load(model_file,map_location=device)))
        print("Loaded DiscGNN done")
    def _load_state_dict(self, model_file, strict=False):
        print("Loading from {} ...".format(model_file))
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        # print(state_dict)
        print(self.load_state_dict(state_dict))
        print("Loaded DiscGNN done")
        return
class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x, e):    
        encoding_indices = self.get_code_indices(x, e) # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss
    
    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c + index_n + index_o)
        # compute L2 distance
        encoding_indices = torch.ones(x.size(0)).long().to(x.device)
        # C:
        distances = (
            torch.sum(e[index_c] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[0: 377] ** 2, dim=1) -
            2. * torch.matmul(e[index_c], self.embeddings.weight[0: 377].t())
        )
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N:
        distances = (
            torch.sum(e[index_n] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[378: 433] ** 2, dim=1) -
            2. * torch.matmul(e[index_n], self.embeddings.weight[378: 433].t())
        ) 
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O:
        distances = (
            torch.sum(e[index_o] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[434: 488] ** 2, dim=1) -
            2. * torch.matmul(e[index_o], self.embeddings.weight[434: 488].t())
        )   
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others:
        distances = (
            torch.sum(e[index_others] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[489: 511] ** 2, dim=1) -
            2. * torch.matmul(e[index_others], self.embeddings.weight[489: 511].t())
        ) 
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489

        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 

    def from_pretrained(self, model_file,device):
        print("Loading from {} ...".format(model_file))
        print(self.load_state_dict(torch.load(model_file,map_location=device)))
        print("Loaded VectorQuantizer done")
    def _load_state_dict(self, model_file, strict=False):
        print("Loading from {} ...".format(model_file))
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        print(state_dict)
        print(self.load_state_dict(state_dict, strict=strict))
        print("Loaded vqvae done")
        return
class VQVAE(nn.Module):
    def __init__(self, config,discGNN=False,use_graph_rep=False,name=""):
        super(VQVAE, self).__init__()
        config.gin_num_layers=5
        config.gin_hidden_dim=300
        config.num_tokens=512
        self.use_graph_rep=use_graph_rep
        self.name=name

        self.main_model = DiscreteGNN(
            num_layer=5,
            emb_dim=300,
            num_tokens=512,
            JK='last',
            gnn_type='gin'
        )
        if not discGNN:
            self.codebook = VectorQuantizer(300, 512, commitment_cost = 0.25)
            self.codebook.from_pretrained(f"/apdcephfs/share_1364275/yandrewchen/graph-llm/checkpoints/vqquantizer_zinc_standard_agent_epoch_60.pth",device=torch.device("cpu"))
        else:
            self.codebook=None
        if hasattr(config, "projection_dim"):
            self.projector = nn.Linear(config.gin_hidden_dim, config.projection_dim)
            self.output_dim = config.projection_dim
        else:
            self.projector = None
            self.output_dim = config.gin_hidden_dim
        if hasattr(config, "init_checkpoint"):
            logger.info("VQVAE: load checkpoint from %s" % (config.init_checkpoint))
            self.main_model.load_state_dict(torch.load(config.init_checkpoint, map_location="cpu"),strict=False)


    # if not self.use_graph_agg:
        # graph_feat = self.pad_node(graph, graph_feat)
    def pad_node(self, data, node_representation):
        # pad the repr so that each graph has some number of node repr
        ptr = data.ptr.tolist()
        nodes = [node_representation[ptr[i]: ptr[i+1]] for i in range(data.num_graphs)]
        nnodes = [ptr[i+1] - ptr[i] for i in range(data.num_graphs)]
        max_len = max(nnodes)
        pad_size = [max_len - x_ for x_ in nnodes]
        pad = self.pad_token.to(device=node_representation.device)
        node_repr = torch.stack([torch.cat([node, pad.expand(pz, -1)]) for pz, node in zip(pad_size, nodes)])
        return node_repr
    @torch.no_grad()
    def forward(self, mol):
        if self.codebook is not None:
            h_graph, h_node = self.main_model(mol)
            h_node, loss = self.codebook(mol.x,h_node)
            # h_node = self.codebook.quantize(encoding_indices)
            # TODO the original InstructMol does not do padding
            # h_node = self.pad_node(mol, h_node)
            # TODO, rigorous take if necessary
            h_graph = h_node
        else:
            h_graph, h_node = self.main_model(mol)
        if self.projector is not None:
            h_graph = self.projector(h_graph)
            h_node = self.projector(h_node)
        if "233" in self.name:
            h_graph*=0
            h_node*=0
        return h_graph, h_node

    def encode_mol(self, mol, proj=False, return_node_feats=False):
        self.main_model.eval() # hard code: set to eval mode
        h_graph, h_node = self.forward(mol)
        if proj and self.projector is not None:
            h_graph = self.projector(h_graph)
            h_node = self.projector(h_node)
        if return_node_feats:
            return h_graph, h_node
        else:
            return h_graph

    def _load_state_dict(self, state_dict, strict=True):
        return self.main_model._load_state_dict(state_dict, strict)
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def hidden_size(self):
        return self.output_dim
