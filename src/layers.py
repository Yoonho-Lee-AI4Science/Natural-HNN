#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import math
import torch
import torch.nn.functional as F
import os
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch_geometric
#from torch_geometric.utils import scatter
from torch_geometric.typing import Adj, Size, OptTensor
import torch_scatter
import pdb


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
        

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class PMA_batch(torch.nn.Module):  # batch_confirmed
    def __init__(self, data_stat, in_channels, hid_dim, out_channels, 
                 num_layers, heads=1, concat=True, negative_slope=0.2, dropout=0.0, **kwargs):
        super(PMA_batch, self).__init__(**kwargs)

        self.data_stat = data_stat
        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'
        self.lin_K = torch.nn.Linear(in_channels, self.heads*self.hidden)
        self.lin_V = torch.nn.Linear(in_channels, self.heads*self.hidden)
        self.att_r = torch.nn.Parameter(torch.Tensor( 1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden, hidden_channels=self.heads*self.hidden, out_channels=out_channels,
                       num_layers=num_layers, dropout=.0, Normalization='None')
        self.ln0 = torch.nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = torch.nn.LayerNorm(self.heads*self.hidden)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index: Adj, size: Size = None):
        H, C = self.heads, self.hidden
        num_graph = x.size(0)
        x_K = self.lin_K(x).view(num_graph, -1, H, C)
        x_V = self.lin_V(x).view(num_graph, -1, H, C)
        alpha_r = (x_K * self.att_r.unsqueeze(0)).sum(dim=-1)[:,edge_index[0]]
        alpha_r = torch_geometric.utils.softmax(F.leaky_relu(alpha_r, self.negative_slope), index=edge_index[1], dim = -2)
        out = x_V[:,edge_index[0]] * alpha_r.unsqueeze(-1)
        out = torch_scatter.scatter(out, index = edge_index[1], dim = 1, reduce = self.aggr) 
        out += self.att_r.unsqueeze(0)
        out = self.ln0(out.view(num_graph, -1, self.heads * self.hidden)) # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out))) # rFF and skip connection. Lhs of eq(7) in GMT paper.
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class PMA(MessagePassing):  # batch_confirmed
    def __init__(self, data_stat, in_channels, hid_dim, out_channels, 
                 num_layers, heads=1, concat=True, negative_slope=0.2, dropout=0.0, **kwargs):
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.data_stat = data_stat
        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'
        self.lin_K = torch.nn.Linear(in_channels, self.heads*self.hidden)
        self.lin_V = torch.nn.Linear(in_channels, self.heads*self.hidden)
        self.att_r = torch.nn.Parameter(torch.Tensor( 1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden, hidden_channels=self.heads*self.hidden, out_channels=out_channels,
                       num_layers=num_layers, dropout=.0, Normalization='None')
        self.ln0 = torch.nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = torch.nn.LayerNorm(self.heads*self.hidden)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index: Adj, size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.hidden
        alpha_r: OptTensor = None
        assert isinstance(x, Tensor), 'data.x must be tensor'
        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)
        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr)
        out += self.att_r  # This is Seed + Multihead
        out = self.ln0(out.view(-1, self.heads * self.hidden)) # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out))) # rFF and skip connection. Lhs of eq(7) in GMT paper.
        return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geometric.utils.softmax(alpha, index, ptr, index.max()+1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        assert aggr != None,"aggr was not passed!"
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)



class MLP(torch.nn.Module): # batch_confirmed
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='ln', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.normalizations = torch.nn.ModuleList()
        self.InputNorm = InputNorm
        
        assert Normalization in ['bn', 'ln', 'None']
        norms = {'bn':torch.nn.BatchNorm1d, 'ln':torch.nn.LayerNorm, 'None':torch.nn.Identity}
        input_Normalization = Normalization if InputNorm == True else 'None'
        
        self.normalizations.append(norms[input_Normalization](in_channels))
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.normalizations.append(norms[Normalization](hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.normalizations.append(norms[Normalization](hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(self, data_stat, in_dim, hid_dim, out_dim, num_layers,
                 dropout, Normalization='bn', InputNorm=False, heads=1, attention=True ):
        super(HalfNLHconv, self).__init__()
        self.data_stat = data_stat
        self.attention = attention
        self.dropout = dropout
        if self.attention:
            if self.data_stat.batch_size == 0:
                self.prop = PMA(self.data_stat, in_dim, hid_dim, out_dim, num_layers, heads=heads)
            else: 
                self.prop = PMA_batch(self.data_stat, in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
            self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            self.f_enc.reset_parameters()
            self.f_dec.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add'):
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.data_stat.batch_size > 0:
                x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            else: 
                x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
            
        return x

    def message(self, x_j, norm):
        if self.data_stat.batch_size > 0:
            return (norm.view(-1, 1, 1) * x_j.permute(1,0,2)).permute(1,0,2)
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, aggr='add'):
        assert aggr != None, "HalfNLHconv aggregate function : aggr was not passed!"
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce=aggr)


class HNN_general_conv(MessagePassing):
    def __init__(self, data_stat, in_channels, hidden_channels, out_channels, nonlinear_activation_he = None, 
                 weight_v2e = True, weight_e2v = False, bias_v2e = False, bias_e2v = True, 
                 alpha = None, beta = None, **kwargs):
        
        
        # alpha and beta for unigcn2 
        kwargs.setdefault('aggr', 'add')
        super(HNN_general_conv, self).__init__(node_dim=0, **kwargs)
        self.data_stat = data_stat
        self.activation_function_dict = {'relu':F.relu, 'elu':F.elu, 'sigmoid':torch.sigmoid}
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_activation_he = nonlinear_activation_he # would be true for HNHN
        self.alpha = alpha
        self.beta = beta
        if weight_v2e:
            self.weight_v2e = torch.nn.Parameter(torch.Tensor(in_channels, hidden_channels))
        else: 
            self.register_parameter('weight_v2e', None)
        if weight_e2v:
            self.weight_e2v = torch.nn.Parameter(torch.Tensor(hidden_channels, out_channels))
        else: 
            self.register_parameter('weight_e2v', None)
        if bias_v2e:
            self.bias_v2e = torch.nn.Parameter(torch.Tensor(hidden_channels))
        else:
            self.register_parameter('bias_v2e', None)
        if bias_e2v:
            self.bias_e2v = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_e2v', None)
        if self.alpha != None and self.beta != None:
            self.W = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        else: 
            self.register_parameter('W', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.weight_v2e)
        glorot(self.weight_e2v)
        glorot(self.W)
        zeros(self.bias_v2e)
        zeros(self.bias_e2v)
        
    def forward(self, x, edge_index, x0 = None):
        #pdb.set_trace()
        if self.weight_v2e != None:
            x = torch.matmul(x, self.weight_v2e)
        if self.data_stat.D_v_right != 'None' and self.data_stat.D_v_right != None:
            x = self.data_stat.D_v_right.unsqueeze(-1) * x

        #self.flow = 'source_to_target'
        if self.data_stat.batch_size > 0:
            x = self.propagate(edge_index, x=x.permute(1,0,2), norm=self.data_stat.D_e_right, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges)).permute(1,0,2)
        else: 
            x = self.propagate(edge_index, x=x, norm=self.data_stat.D_e_right, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        if self.bias_v2e != None:
            x += self.bias_v2e
        if self.nonlinear_activation_he != None and self.nonlinear_activation_he != 'None':
            x = self.activation_function_dict[self.nonlinear_activation_he](x) 
            
        output_xe = x
             
        if self.weight_e2v != None:
            x = torch.matmul(x, self.weight_e2v)
        if self.data_stat.D_e_left != 'None' and self.data_stat.D_e_left != None:
            x = self.data_stat.D_e_left.unsqueeze(-1) * x
        #self.flow = 'target_to_source'     
        
        reversed_edge_index = torch.stack( [edge_index[1], edge_index[0]], dim=0)
        if self.data_stat.batch_size > 0:
            x = self.propagate(reversed_edge_index, x=x.permute(1,0,2), norm=self.data_stat.D_v_left, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes)).permute(1,0,2)
        else: 
            x = self.propagate(reversed_edge_index, x=x, norm=self.data_stat.D_v_left, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
        
        if self.bias_e2v != None:
            x += self.bias_e2v
        if self.alpha != None and self.beta != None and self.W != None:
            x_temp = (1-self.alpha) * x + self.alpha * x0
            x = (1-self.beta) * x_temp + self.beta * torch.matmul(x_temp,self.W)
            
        return x, output_xe

    def message(self, x_j, norm_i):
        if self.data_stat.batch_size > 0:
            return (norm_i.view(-1, 1, 1) * x_j)
        return norm_i.view(-1, 1) * x_j

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels, self.hidden_channels, self.out_channels)


class EquivSetConv(torch.nn.Module):
    def __init__(self, spec, data_stat, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='mean', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()
        self.spec = spec
        self.data_stat = data_stat
        self.get_xe_mean = get_xe_by_mean_agg(self.data_stat)
        if mlp1_layers > 0:
            if self.spec == 0:
                self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                    dropout=dropout, Normalization=normalization, InputNorm=input_norm)
            elif self.spec == 1:
                self.W1 = MLP(in_features+out_features, out_features, out_features, mlp1_layers,
                    dropout=dropout, Normalization=normalization, InputNorm=input_norm)
            else : 
                raise NotImplementedError
        else:
            self.W1 = torch.nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features+out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = torch.nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, edge_index, edge_attr = None, X0 = None):
        if self.spec == 0: # EDHNN
            x = self.W1(X)
            if self.data_stat.batch_size == 0:
                xe = self.get_xe_mean(x, edge_index)
            else:
                xe = self.get_xe_mean(x.permute(1,0,2), edge_index)
        elif self.spec == 1: # EDHNN2
            if self.data_stat.batch_size == 0:
                x = torch.cat([X[edge_index[0]], edge_attr], -1)
            else:
                x = torch.cat([X[:,edge_index[0]], edge_attr], -1)
            edge_attr = self.W1(x)
            xe = torch_scatter.scatter(edge_attr, edge_index[1], dim = -2, reduce = self.aggr, dim_size = self.data_stat.exact_num_hyperedges)
        else: 
            raise NotImplementedError
        
        if self.data_stat.batch_size == 0:
            e2v_attr = self.W2(torch.cat([X[edge_index[0]], xe[edge_index[1]]], -1))
        else:
            e2v_attr = self.W2(torch.cat([X[:,edge_index[0]], xe[:,edge_index[1]]], -1))
        x = torch_scatter.scatter(e2v_attr, edge_index[0], dim = -2, reduce = self.aggr, dim_size = self.data_stat.exact_num_nodes)

        X = (1-self.alpha) * x + self.alpha * X0
        X = self.W(X)

        return X, edge_attr, xe



class shine_dual_attention(MessagePassing):
    def __init__(self, data_stat, in_channels, out_channels, dropout = None, heads=1, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(shine_dual_attention, self).__init__(node_dim=-3, **kwargs)
        assert dropout != None, 'dropout must not be None : layers.shine_dual_attention'
        self.data_stat = data_stat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.heads = heads
        self.factor_size = int(out_channels // heads)
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.attention_weight = torch.nn.Parameter(torch.zeros(out_channels,1))
                    
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.attention_weight)
        self.linear.reset_parameters()
        
    def forward(self, x = None, xe = None, edge_index = None):
        x = self.linear(x)
        xe = self.linear(xe)      
        if self.data_stat.batch_size > 0:
            num_graph = x.size(0)
            x = x.reshape(num_graph, -1, self.heads, self.factor_size)
            xe = xe.reshape(num_graph, -1, self.heads, self.factor_size)
            ss = F.leaky_relu(x[:,edge_index[0]] * xe[:,edge_index[1]], negative_slope=0.2) # size of s == edge_index.size(1) * self.out_channels
            alpha = (ss * self.attention_weight.reshape(1,1,self.heads, self.factor_size)).sum(-1) # edge_index.size(1) * 1
        else: 
            x = x.reshape(-1, self.heads, self.factor_size)
            xe = xe.reshape(-1, self.heads, self.factor_size)
            ss = F.leaky_relu(x[edge_index[0]] * xe[edge_index[1]], negative_slope=0.2) # size of s == edge_index.size(1) * self.out_channels  
            alpha = (ss * self.attention_weight.reshape(1,self.heads, self.factor_size)).sum(-1)
            
        alpha_v2e = torch_geometric.utils.softmax(alpha, edge_index[1], dim = -2)#-2?
        alpha_e2v = torch_geometric.utils.softmax(alpha, edge_index[0], dim = -2)#-2?
        alpha_v2e = F.dropout(alpha_v2e, p=self.dropout, training=self.training) # edge_index.size(1)
        alpha_e2v = F.dropout(alpha_e2v, p=self.dropout, training=self.training) # edge_index.size(1)

        # propagate
        xe_out = self.propagate(edge_index, x=x, alpha=alpha_v2e, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        x_out = self.propagate(edge_index.flip([0]), x=xe, alpha=alpha_e2v, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
        
        if self.data_stat.batch_size > 0:
            x_out = x_out.reshape(num_graph, -1, self.out_channels)
            xe_out = xe_out.reshape(num_graph, -1, self.out_channels)
        else: 
            x_out = x_out.reshape(-1, self.out_channels)
            xe_out = xe_out.reshape(-1, self.out_channels)
        return F.elu(x_out), F.elu(xe_out)

    def message(self, x_j: Tensor, alpha: Tensor):
        return alpha.unsqueeze(-1) * x_j


    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class get_xe_by_mean_agg(MessagePassing): # Node to hyperedge MPNN via mean aggregation
    def __init__(self, data_stat, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(get_xe_by_mean_agg, self).__init__(node_dim=0, **kwargs)
        self.data_stat = data_stat
        
    def forward(self, x, edge_index):
        xe = self.propagate(edge_index, x=x, norm=self.data_stat.D_e_right, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        if self.data_stat.batch_size == 0:
            return xe
        else: 
            return xe.permute(1,0,2)

    def message(self, x_j, norm_i):
        if self.data_stat.batch_size == 0:
            return norm_i.view(-1, 1) * x_j
        else: 
            return norm_i.view(-1, 1, 1) * x_j


class get_x_by_mean_agg(MessagePassing): # Hyperedge to node MPNN via mean aggregation
    def __init__(self, data_stat, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(get_x_by_mean_agg, self).__init__(node_dim=0, **kwargs)
        self.data_stat = data_stat
        
    def forward(self, x, edge_index):
        xe = self.propagate(edge_index.flip([0]), x=x, norm=self.data_stat.D_v_left, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
        if self.data_stat.batch_size == 0:
            return xe
        else: 
            #return xe.permute(1,0,2)
            return xe.permute(1,0,2,3)
        

    def message(self, x_j, norm_i):
        if self.data_stat.batch_size == 0:
            return norm_i.view(-1, 1) * x_j
        else: 
            return norm_i.view(-1, 1, 1, 1) * x_j
            #return norm_i.view(-1, 1, 1) * x_j



class disen_base(MessagePassing): 
    def __init__(self, data_stat, in_channels, out_channels, factor_size, heads, spec, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(disen_base, self).__init__(node_dim=0, **kwargs)
        self.data_stat = data_stat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.spec = spec
        self.factor_size = factor_size
        self.factor_classifier = torch.nn.Linear(factor_size, heads)
        
    def reset_base_parameters(self):
        self.factor_classifier.reset_parameters()
    
    
    
    # calculating distance correlation adopted from : https://github.com/AnsongLi/Disen-GNN/blob/main/Disen-GNN/code/model.py
    def create_centered_distance(self,x,zero):
        sq_sum = torch.sum(torch.square(x),1,keepdim=True)
        D = torch.sqrt(torch.maximum(sq_sum + sq_sum.T -2*torch.matmul(x,x.T),zero)+1e-8)
        D = D - torch.mean(D,dim=0,keepdim=True) - torch.mean(D,dim=1,keepdim=True) + torch.mean(D)
        return D
    
    def create_distance_covariance(self, x,y,zero):
        num_samples = x.size(0)
        summed = torch.sum(x*y)/(num_samples * num_samples)
        dcov = torch.sqrt(torch.maximum(summed,zero)+1e-8)
        return dcov
        
    def cov_distance_loss(self, x, y,zero):
        D1 = self.create_centered_distance(x,zero)
        D2 = self.create_centered_distance(y,zero)

        dcov_12 = self.create_distance_covariance(D1, D2,zero)
        dcov_11 = self.create_distance_covariance(D1, D1,zero)
        dcov_22 = self.create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)

        return dcor
        
 
    def factor_cov_distance_loss(self, node_emb):
        factor_emb = node_emb.chunk(self.heads, dim = 1)
        factor_loss = 0
        for i in range(self.heads):
            for j in range(i+1, self.heads):
                factor_loss += self.cov_distance_loss(factor_emb[i], factor_emb[j], torch.tensor(0, dtype= float).to(node_emb.device))
        return factor_loss
   
    def factor_class_discrimination_loss(self, node_emb):
        loss_fn = torch.nn.CrossEntropyLoss()
        factor_loss = 0
        if self.data_stat.batch_size == 0:
            factor_emb = node_emb.chunk(self.heads, dim = 1)
            for factor_iter in range(self.heads):
                factor_labels = (torch.ones(node_emb.size(0)) * factor_iter).long().to(node_emb.device)
                factor_prediction = torch.nn.Softmax(dim=1)(self.factor_classifier(factor_emb[factor_iter]))
                factor_loss += loss_fn(factor_prediction, factor_labels)
        else: 
            factor_emb = node_emb.chunk(self.heads, dim = 2)
            for factor_iter in range(self.heads):
                factor_labels = (torch.ones(node_emb.size(1)* node_emb.size(0)) * factor_iter ).long().to(node_emb.device)
                factor_prediction = torch.nn.Softmax(dim=-1)(self.factor_classifier(factor_emb[factor_iter])).reshape(-1, self.heads)
                factor_loss += loss_fn(factor_prediction, factor_labels)
            
            
        return factor_loss
    
    




class disen_hgnn_layer(disen_base): # inherit disen_base class
    def __init__(self, data_stat, in_channels, out_channels, heads, spec, drop, interpol_ratio, **kwargs):
        assert out_channels % heads == 0, "factor size * num_heads == feature/representation dimension must hold"
        factor_size = out_channels // heads
        self.data_stat = data_stat
        super(disen_hgnn_layer, self).__init__(data_stat, in_channels, out_channels, factor_size, heads, spec, **kwargs)
        self.mean_agg_func = get_xe_by_mean_agg(data_stat)
        self.mean_agg_func_2 = get_x_by_mean_agg(data_stat)
        self.encoder = torch.nn.Linear(in_channels, out_channels, bias = True) 
        self.interpol_ratio = interpol_ratio
        self.sim_scorer = torch.nn.Parameter(torch.zeros(self.heads, self.factor_size, self.factor_size))
        self.layer_norm_1 = torch.nn.LayerNorm(out_channels)
        
        
    def reset_parameters(self):
        self.reset_base_parameters()
        self.encoder.reset_parameters()
        glorot(self.sim_scorer)
        self.layer_norm_1.reset_parameters()
        
        
    def forward(self, x, edge_index):
        
        if self.data_stat.batch_size > 0 : # if batch training for hypergraph classification
            return self.batch_forward(x,edge_index)
        node_factor_emb = torch.nn.Tanh()(self.encoder(x))
        aggregated_factor_emb = self.mean_agg_func(node_factor_emb, edge_index)
        hyperedge_factor_emb = self.encoder(self.mean_agg_func(x, edge_index))
        hyperedge_factor_emb = torch.nn.Tanh()(hyperedge_factor_emb)
        hyperedge_factor_emb = hyperedge_factor_emb.reshape(-1,self.heads, self.factor_size)
        aggregated_factor_emb = aggregated_factor_emb.reshape(-1,self.heads, self.factor_size)
        

        att_score_calc = (F.normalize(hyperedge_factor_emb, p=2, dim = -1).permute(1,0,2) @ self.sim_scorer).permute(1,0,2).reshape(-1, self.heads, self.factor_size)        
        att_score = torch.sum(att_score_calc*F.normalize(aggregated_factor_emb, p=2, dim = -1), dim = -1)
        att_score = torch.nn.Sigmoid()(att_score) # shape : num_hyperedges, num_heads

        hyperedge_emb = (att_score.reshape(-1,self.heads, 1) * aggregated_factor_emb).reshape(-1,self.heads * self.factor_size)

        x_out = self.propagate(edge_index.flip([0]), x=hyperedge_emb, norm=self.data_stat.D_v_left, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
        att_score_aggregated = self.mean_agg_func_2(att_score, edge_index)
        x_out = x_out.reshape(-1, self.heads, self.factor_size) / att_score_aggregated.reshape(-1, self.heads, 1)

        x_out = x_out.reshape(-1, self.heads * self.factor_size)
        x_out = self.layer_norm_1(self.interpol_ratio * x_out + (1-self.interpol_ratio) * torch.nn.Tanh()(self.encoder(x)))
        return x_out, hyperedge_emb
    
    
    def batch_forward(self, x, edge_index):
        num_graph = x.size(0)
        node_factor_emb = torch.nn.Tanh()(self.encoder(x))
        aggregated_factor_emb = self.mean_agg_func(node_factor_emb.permute(1,0,2), edge_index).reshape(num_graph,-1,self.heads, self.factor_size)
        hyperedge_factor_emb = self.encoder(self.mean_agg_func(x.permute(1,0,2), edge_index)).reshape(num_graph, -1,self.heads, self.factor_size)
        hyperedge_factor_emb = torch.nn.Tanh()(hyperedge_factor_emb)

        att_score_calc = (F.normalize(hyperedge_factor_emb, p=2, dim = -1).permute(0,2,1,3) @ self.sim_scorer).permute(0,2,1,3)
        att_score = torch.sum(att_score_calc*F.normalize(aggregated_factor_emb, p=2, dim = -1), dim = -1, keepdim=True)
        att_score = torch.nn.Sigmoid()(att_score) # shape : num_hyperedges, num_heads

        hyperedge_emb = att_score * aggregated_factor_emb

        x_out = self.mean_agg_func_2(hyperedge_emb.permute(1,0,2,3), edge_index)
        att_score_aggregated = self.mean_agg_func_2(att_score.permute(1,0,2,3), edge_index)
        x_out /= att_score_aggregated

        x_out = x_out.reshape(num_graph, -1, self.heads * self.factor_size)        
        x_out = self.layer_norm_1(self.interpol_ratio * x_out + (1-self.interpol_ratio) * torch.nn.Tanh()(self.encoder(x)))
        return x_out, hyperedge_emb.reshape(num_graph,-1,self.heads * self.factor_size)
    

    

    def message(self, x_j: Tensor, norm_i: Tensor):
        if self.data_stat.batch_size == 0:
            return norm_i.view(-1,1) * x_j
        else: 
            return norm_i.view(-1,1,1,1) * x_j
    
    def factor_loss(self, x, edge_index):
        node_factor_emb = self.encoder(x)
        node_factor_emb = torch.nn.Tanh()(node_factor_emb)
        if self.data_stat.batch_size == 0:
            aggregated_factor_emb = self.mean_agg_func(node_factor_emb, edge_index)
        else:
            aggregated_factor_emb = self.mean_agg_func(node_factor_emb.permute(1,0,2), edge_index)
        return self.factor_class_discrimination_loss(aggregated_factor_emb)
    


    def get_hyperedge_attention_score(self, x, edge_index):
        if self.data_stat.batch_size == 0 :
            node_factor_emb = torch.nn.Tanh()(self.encoder(x))
            aggregated_factor_emb = self.mean_agg_func(node_factor_emb, edge_index)
            hyperedge_factor_emb = self.encoder(self.mean_agg_func(x, edge_index))
            hyperedge_factor_emb = torch.nn.Tanh()(hyperedge_factor_emb)
            hyperedge_factor_emb = hyperedge_factor_emb.reshape(-1,self.heads, self.factor_size)
            aggregated_factor_emb = aggregated_factor_emb.reshape(-1,self.heads, self.factor_size)
            
            att_score_calc = (F.normalize(hyperedge_factor_emb, p=2, dim = -1).permute(1,0,2) @ self.sim_scorer).permute(1,0,2).reshape(-1, self.heads, self.factor_size)        
            att_score = torch.sum(att_score_calc*F.normalize(aggregated_factor_emb, p=2, dim = -1), dim = -1)
            att_score = torch.nn.Sigmoid()(att_score) # shape : num_hyperedges, num_heads
            return att_score
            
        else: # if batch training
            node_factor_emb = torch.nn.Tanh()(self.encoder(x))
            num_graph = x.size(0)
            aggregated_factor_emb = self.mean_agg_func(node_factor_emb.permute(1,0,2), edge_index)
            hyperedge_factor_emb = self.encoder(self.mean_agg_func(x.permute(1,0,2), edge_index))
            hyperedge_factor_emb = torch.nn.Tanh()(hyperedge_factor_emb)
            hyperedge_factor_emb = hyperedge_factor_emb.reshape(num_graph, -1,self.heads, self.factor_size)
            node_factor_emb = node_factor_emb.reshape(num_graph,-1,self.heads, self.factor_size)
            aggregated_factor_emb = aggregated_factor_emb.reshape(num_graph,-1,self.heads, self.factor_size)
            
            att_score_calc = (F.normalize(hyperedge_factor_emb, p=2, dim = -1).permute(0,2,1,3) @ self.sim_scorer).permute(0,2,1,3)#.reshape(num_graph, -1, self.heads, self.factor_size)
            att_score = torch.sum(att_score_calc*F.normalize(aggregated_factor_emb, p=2, dim = -1), dim = -1)
            att_score = torch.nn.Sigmoid()(att_score) # shape : num_hyperedges, num_heads
            return att_score.reshape(-1,self.heads)[:self.data_stat.original_num_hyperedges,:]

    
    
    
class hsdn_prop_layer(MessagePassing):
    def __init__(self, data_stat, heads, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(hsdn_prop_layer, self).__init__(node_dim=0, **kwargs)
        self.data_stat = data_stat
        self.heads = heads
        self.agg = get_xe_by_mean_agg(data_stat)
        
    def reset_parameters(self):
        
        return
            
        
    def forward(self, x, edge_index, att_score):
        if self.data_stat.D_v_right != 'None' and self.data_stat.D_v_right != None:
            if self.data_stat.batch_size > 0 :
                x = self.data_stat.D_v_right.unsqueeze(-1).unsqueeze(0) * x
            else: 
                x = self.data_stat.D_v_right.unsqueeze(-1) * x

        if self.data_stat.batch_size > 0:
            num_graph = x.size(0)
            x = self.agg(x.permute(1,0,2), edge_index)
        else: 
            x = self.agg(x, edge_index)
            
        x = torch.cat([x.unsqueeze(-2) for i in range(self.heads)], dim = -2)* att_score.unsqueeze(-1)
        
        if self.data_stat.batch_size > 0:
            output_xe = x.reshape(num_graph, -1, x.size(-1)*x.size(-2))
            out = self.propagate(edge_index.flip([0]), x=output_xe.permute(1,0,2), size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes)).permute(1,0,2)
        else: 
            output_xe = x.reshape(-1, x.size(-1)*x.size(-2))
            out = self.propagate(edge_index.flip([0]), x=output_xe, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
        return out, output_xe

    def message(self, x_j):
        return x_j
    
    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels, self.hidden_channels, self.out_channels)
    
    
    

class hypergat_layer(MessagePassing):
    def __init__(self, data_stat, in_channels, out_channels, heads, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(hypergat_layer, self).__init__(node_dim=-3, **kwargs)
        self.data_stat = data_stat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.factor_size = int(out_channels // heads)
        self.w1 = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.w2 = torch.nn.Parameter(torch.Tensor(heads, self.factor_size, self.factor_size))
        self.a1 = torch.nn.Parameter(torch.Tensor(1, heads, self.factor_size))
        self.a2 = torch.nn.Parameter(torch.Tensor(1, heads, self.factor_size*2))
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        torch.nn.init.uniform_(self.a1.data, -stdv, stdv)
        torch.nn.init.uniform_(self.a2.data, -stdv, stdv)
        return
        
    def forward(self, x, edge_index, x0 = None):
        x = x @ self.w1
        if self.data_stat.batch_size > 0: # batch training for graph classification
            num_graph = x.size(0)
            x = x.view(num_graph, -1, self.heads, self.factor_size)
            u_k = self.leaky_relu(x)
            att_v2e = torch_geometric.utils.softmax((u_k * self.a1.unsqueeze(0)).sum(-1)[:,edge_index[0]], edge_index[1], dim = -2)
            xe = self.propagate(edge_index, x=x, alpha=att_v2e, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
            output_xe = xe.reshape(num_graph, -1, self.out_channels)
            xe = (xe.permute(0,2,1,3) @ self.w2).permute(0,2,1,3)
            v_j = self.leaky_relu(torch.cat([xe[:,edge_index[1]], x[:,edge_index[0]]], -1))
            att_e2v = torch_geometric.utils.softmax((v_j * self.a2.unsqueeze(0)).sum(-1), edge_index[0], dim = -2)
            x_out = self.propagate(edge_index.flip([0]), x=xe, alpha=att_e2v, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
            x_out = x_out.reshape(num_graph, -1, self.out_channels)
        else:
            x = x.view(-1,self.heads, self.factor_size)
            u_k = self.leaky_relu(x)
            att_v2e = torch_geometric.utils.softmax((u_k * self.a1).sum(-1)[edge_index[0]], edge_index[1], dim = -2)
            xe = self.propagate(edge_index, x=x, alpha=att_v2e, size=(self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
            output_xe = xe.reshape(-1, self.out_channels)
            xe = (xe.permute(1,0,2) @ self.w2).permute(1,0,2)
            v_j = self.leaky_relu(torch.cat([xe[edge_index[1]], x[edge_index[0]]], -1))
            att_e2v = torch_geometric.utils.softmax((v_j * self.a2).sum(-1), edge_index[0], dim = -2)
            x_out = self.propagate(edge_index.flip([0]), x=xe, alpha=att_e2v, size=(self.data_stat.exact_num_hyperedges, self.data_stat.exact_num_nodes))
            x_out = x_out.reshape(-1, self.out_channels)
        return x_out, output_xe
    
    def message(self, x_j: Tensor, alpha: Tensor):
        return alpha.unsqueeze(-1) * x_j
    
    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels, self.hidden_channels, self.out_channels)

