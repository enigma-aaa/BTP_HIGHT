import logging
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)

num_atom_type = 121 #including the extra motif tokens and graph token
num_chirality_tag = 11  #degree

num_bond_type = 7 
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    
    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)



class GATConv(MessagePassing):
    def __init__(self,
                 emb_dim,
                 heads=2,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.

        self.in_channels = emb_dim
        self.out_channels = emb_dim
        self.edge_dim = emb_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.weight = torch.nn.Parameter(torch.Tensor(emb_dim, heads * emb_dim))    # emb(in) x [H*emb(out)]
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim + self.edge_dim))   # 1 x H x [2*emb(out)+edge_dim]    # new
        self.edge_update = torch.nn.Parameter(torch.Tensor(emb_dim + self.edge_dim, emb_dim))   # [emb(out)+edge_dim] x emb(out)  # new

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        edge_attr = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)  

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)  
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) 

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)  
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)  
                          
    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr): 
       
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim  # new
        x_j = torch.cat([x_j, edge_attr], dim=-1)  # (E+N) x H x (emb(out)+edge_dim)   # new
        x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)   # Computes a sparsely evaluated softmax

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)   # (E+N) x H x (emb(out)+edge_dim)

    def update(self, aggr_out): 
        aggr_out = torch.mm(aggr_out, self.edge_update)# N x emb(out)  # new

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out



class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


class GNN(torch.nn.Module):
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
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
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
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.elu(h), self.drop_ratio, training = self.training)  #relu->elu
            
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
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        gnn_type: gin, gcn, graphsage, gat
        
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
            torch.nn.ELU(),
            torch.nn.Linear((self.emb_dim)//2, self.num_tasks))

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def super_node_rep(self, node_rep, batch):
        super_group = []
        for i in range(len(batch)):
            if i != (len(batch)-1) and batch[i] != batch[i+1]:
                super_group.append(node_rep[i,:])
            elif i == (len(batch) -1):
                super_group.append(node_rep[i,:])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep


    def node_rep_(self, node_rep, batch):
        super_group = []
        batch_new = []
        for i in range(len(batch)):
            if i != (len(batch)-1) and batch[i] == batch[i+1]:
                super_group.append(node_rep[i,:])
                batch_new.append(batch[i].item())
        super_rep = torch.stack(super_group, dim=0)
        batch_new = torch.tensor(np.array(batch_new)).to(batch.device).to(batch.dtype)
        return super_rep, batch_new

    def mean_pool_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(len(batch)):
            super_group[batch[i]].append(node_rep[i,:])
        node_rep = [torch.stack(list, dim=0).mean(dim=0) for list in super_group]
        super_rep = torch.stack(node_rep, dim=0)
        return super_rep

    def node_group_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(len(batch)):
            super_group[batch[i]].append(node_rep[i,:])
        node_rep = [torch.stack(list, dim=0) for list in super_group]
        return node_rep 

    def graph_emb(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)
        return super_rep

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)

        return self.graph_pred_linear(super_rep)


class DiscreteGNN2(torch.nn.Module):
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
        temperature = 0.9, drop_ratio = 0, gnn_type = "gin", graph_pooling="mean",use_graph_agg=True):
        super(DiscreteGNN2, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.num_tokens = num_tokens
        # self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.out_dim=emb_dim
        self.graph_pooling= graph_pooling
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
                self.gnns.append(GINConv(emb_dim,aggr = "add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))
        self.gnns.append(GINConv(emb_dim,aggr = "add"))    

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.att_pool = GlobalAttention(nn.Linear(emb_dim, 1))        

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
        # node_representation[node_representation.isnan()]=0
        # if self.use_graph_agg:
        #     # graph_agg = scatter_mean(node_representation, data.batch, dim=0, dim_size=data.num_graphs)
        #     # graph_agg = graph_agg.unsqueeze(1)  # [bz, 1, feat_size]
        #     graph_agg = global_mean_pool(node_representation,data.batch)
        #     return graph_agg, node_representation
        if self.graph_pooling =="mean":
            graph_agg = global_mean_pool(node_representation,data.batch)
        elif self.graph_pooling =="sum":
            graph_agg = global_add_pool(node_representation,data.batch)
        elif self.graph_pooling =="max":
            graph_agg = global_max_pool(node_representation,data.batch)
        elif self.graph_pooling =="att":
            graph_agg = self.att_pool(node_representation,data.batch)
        return graph_agg, node_representation

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
        print(self.load_state_dict(state_dict,strict=strict))
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
        print("Loaded HVQVAE done")
        return
class HVQVAE2(nn.Module):
    def __init__(self, config,discGNN=False,use_graph_rep=False,name=""):
        super(HVQVAE2, self).__init__()
        config.gin_num_layers=5
        config.gin_hidden_dim=300
        config.num_tokens=512
        self.use_graph_rep=use_graph_rep
        self.name=name

        self.main_model = DiscreteGNN2(
            num_layer=5,
            emb_dim=300,
            num_tokens=300,
            JK='last',
            gnn_type='gin',
            graph_pooling=config.graph_pooling,
            drop_ratio=0.5
        )
        if not discGNN and False:
            self.codebook = VectorQuantizer(300, 512, commitment_cost = 0.25)
            self.codebook.from_pretrained(f"../drugchat/Mole-BERT/vqquantizer_zinc_standard_agent_epoch_60.pth",device=torch.device("cpu"))
        else:
            self.codebook=None
        if hasattr(config, "projection_dim"):
            self.projector = nn.Linear(config.gin_hidden_dim, config.projection_dim)
            self.output_dim = config.projection_dim
        else:
            self.projector = None
            self.output_dim = config.gin_hidden_dim
        if hasattr(config, "init_checkpoint"):
            logger.info("HVQVAE2: load checkpoint from %s" % (config.init_checkpoint))
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
