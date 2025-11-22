import torch
from layers import *
import torch.nn.functional as F
import math
import copy
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
import pdb
import os
import torch_geometric.utils
import torch_scatter
import numpy as np


# current new one
class HNHN(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(HNHN, self).__init__()
        self.num_layers = args.num_layers
        self.args = args
        self.dropout = args.dropout
        self.data_stat = data_stat
        self.convs = torch.nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNN_general_conv(data_stat, data_stat.exact_num_feature_dim, args.hidden, args.out_dim , nonlinear_activation_he=args.he_activation,
                                               weight_v2e = True, weight_e2v= True, bias_v2e=True, bias_e2v=True, alpha = None, beta = None))
        else:
            self.convs.append(HNN_general_conv(data_stat, data_stat.exact_num_feature_dim, args.hidden, args.hidden, nonlinear_activation_he=args.he_activation,
                                               weight_v2e = True, weight_e2v= True, bias_v2e=True, bias_e2v=True, alpha = None, beta = None))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNN_general_conv(data_stat, args.hidden, args.hidden, args.hidden, nonlinear_activation_he=args.he_activation,
                                               weight_v2e = True, weight_e2v= True, bias_v2e=True, bias_e2v=True, alpha = None, beta = None))
            self.convs.append(HNN_general_conv(data_stat, args.hidden, args.hidden, args.out_dim , nonlinear_activation_he=args.he_activation,
                                               weight_v2e = True, weight_e2v= True, bias_v2e=True, bias_e2v=True, alpha = None, beta = None))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        if self.num_layers == 1:
            x, conv_xe_out = self.convs[0](x, edge_index)
            xe_emb_list.append(conv_xe_out)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x, conv_xe_out = conv(x, edge_index)
                xe_emb_list.append(conv_xe_out)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x, conv_xe_out = self.convs[-1](x, edge_index)
            xe_emb_list.append(conv_xe_out)
        if self.args.task == 'bio':
            return x, None, xe_emb_list
        return x, None
        

class HCHA(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(HCHA, self).__init__()
        self.args = args

        self.num_layers = args.num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.data_stat = data_stat
        self.convs = torch.nn.ModuleList()
        self.convs.append(HNN_general_conv(data_stat, data_stat.exact_num_feature_dim, args.hidden, args.hidden, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))
        for _ in range(self.num_layers-2):
            self.convs.append(HNN_general_conv(data_stat, args.hidden, args.hidden, args.hidden, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))
        self.convs.append(HNN_general_conv(data_stat, args.hidden, args.out_dim, args.out_dim, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        for i, conv in enumerate(self.convs[:-1]):
            x, conv_xe_out = conv(x, edge_index)
            xe_emb_list.append(conv_xe_out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, conv_xe_out = self.convs[-1](x, edge_index)
        xe_emb_list.append(conv_xe_out)
        if self.args.task == 'bio':
            return x, None, xe_emb_list
        return x, None

    

class HGNN(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(HGNN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.data_stat = data_stat
        self.convs = torch.nn.ModuleList()
        self.convs.append(HNN_general_conv(data_stat, data_stat.exact_num_feature_dim, args.hidden, args.hidden, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))
        for _ in range(self.num_layers-2):
            self.convs.append(HNN_general_conv(data_stat, args.hidden, args.hidden, args.hidden, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))
        self.convs.append(HNN_general_conv(data_stat, args.hidden, args.out_dim, args.out_dim, nonlinear_activation_he=None,
                                               weight_v2e = True, weight_e2v= False, bias_v2e=False, bias_e2v=True, alpha = None, beta = None))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        for i, conv in enumerate(self.convs[:-1]):
            x, conv_xe_out = conv(x, edge_index)
            xe_emb_list.append(conv_xe_out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, conv_xe_out = self.convs[-1](x, edge_index)
        xe_emb_list.append(conv_xe_out)
        if self.args.task == 'bio':
            return x, None, xe_emb_list
        return x, None

class UniGCNII(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(UniGCNII, self).__init__()
        self.args = args
        self.input_drop = torch.nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = torch.nn.Dropout(0.2) # 0.2 is chosen for GCNII
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(data_stat.exact_num_feature_dim, args.hidden))
        lamda, alpha = 0.5, 0.1 
        for i in range(args.num_layers):
            beta = math.log(lamda/(i+1)+1)
            self.convs.append(HNN_general_conv(data_stat, args.hidden, args.hidden, args.hidden, nonlinear_activation_he=None,
                                               weight_v2e = False, weight_e2v= False, bias_v2e=False, bias_e2v=False, alpha = alpha, beta = beta))
        self.convs.append(torch.nn.Linear(args.hidden, args.out_dim))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x
        for conv in self.convs[1:-1]:
            x = self.dropout(x)
            x, conv_xe_out = conv(x, edge_index, x0)
            xe_emb_list.append(conv_xe_out)
            x = F.relu(x)
        if self.args.task == 'bio':
            return x, None, xe_emb_list         
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x, None
    

class SetGNN(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(SetGNN, self).__init__()
        self.args = args
        self.data_stat = data_stat
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        
        self.V2EConvs = torch.nn.ModuleList()
        self.E2VConvs = torch.nn.ModuleList()
        self.V2EConvs.append(HalfNLHconv(self.data_stat, in_dim=self.data_stat.exact_num_feature_dim, hid_dim=args.hidden, out_dim=args.hidden, 
                                         num_layers=args.MLP_num_layers, dropout=self.dropout,
                                         Normalization=self.NormLayer, InputNorm=self.InputNorm, heads=args.heads, attention=args.PMA))
        self.E2VConvs.append(HalfNLHconv(self.data_stat, in_dim=args.hidden, hid_dim=args.hidden, out_dim=args.hidden, 
                                         num_layers=args.MLP_num_layers, dropout=self.dropout, 
                                         Normalization=self.NormLayer, InputNorm=self.InputNorm, heads=args.heads, attention=args.PMA))
        for _ in range(self.num_layers-1):
            self.V2EConvs.append(HalfNLHconv(self.data_stat, in_dim=args.hidden, hid_dim=args.hidden, out_dim=args.hidden, 
                                             num_layers=args.MLP_num_layers, dropout=self.dropout, 
                                             Normalization=self.NormLayer, InputNorm=self.InputNorm, heads=args.heads, attention=args.PMA))
            self.E2VConvs.append(HalfNLHconv(self.data_stat, in_dim=args.hidden, hid_dim=args.hidden, out_dim=args.hidden, 
                                             num_layers=args.MLP_num_layers, dropout=self.dropout, 
                                             Normalization=self.NormLayer, InputNorm=self.InputNorm, heads=args.heads, attention=args.PMA))
        self.classifier = MLP(in_channels=args.hidden, hidden_channels=args.Classifier_hidden, out_channels=args.out_dim, 
                              num_layers=args.Classifier_num_layers, dropout=self.dropout, Normalization=self.NormLayer, InputNorm=False)


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, edge_weight, xe = None, m = None):

        xe_emb_list = []
        reversed_edge_index = torch.stack( [edge_index[1], edge_index[0]], dim=0)
        
        x = F.dropout(x, p=0.2, training=self.training) # Input dropout
        for i, _ in enumerate(self.V2EConvs):
            x = F.relu(self.V2EConvs[i](x, edge_index, edge_weight, self.aggr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xe_emb_list.append(x)
            x = F.relu(self.E2VConvs[i]( x, reversed_edge_index, edge_weight, self.aggr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.task == 'bio':
            return x, None, xe_emb_list
        x = self.classifier(x)

        return x, None
    



class EquivSetGNN(torch.nn.Module):
    def __init__(self, args, data_stat):
        super().__init__()
        self.args = args
        self.data_stat = data_stat
        self.input_drop = torch.nn.Dropout(0.2) # 0.6 is chosen as default
        self.dropout = torch.nn.Dropout(args.dropout) # 0.2 is chosen for GCNII

        self.lin_in = torch.nn.Linear(self.data_stat.exact_num_feature_dim, args.hidden)
        self.conv = EquivSetConv(self.args.disen_spec, self.data_stat, args.hidden, args.hidden, mlp1_layers=self.args.edhnn_mlp_layer_1, mlp2_layers=self.args.edhnn_mlp_layer_2,
            mlp3_layers=self.args.edhnn_mlp_layer_3, alpha=args.interpol_ratio, aggr=args.aggregate,
            dropout=args.dropout, normalization=args.normalization, input_norm=True)

        self.classifier = MLP(in_channels=args.hidden,
            hidden_channels=args.Classifier_hidden,
            out_channels=args.out_dim,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization, # default layernorm
            InputNorm=False)
        if self.args.disen_spec == 0: # EDHNN
            self.edge_attr = None
        elif self.args.disen_spec  == 1: # EDHNN 2
            self.edge_attr = torch.nn.Parameter(torch.Tensor(self.data_stat.exact_num_hyperedges,self.args.hidden))
        else : 
            raise NotImplementedError

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        
        self.classifier.reset_parameters()
        if self.args.disen_spec == 1:
            glorot(self.edge_attr)
            

    def forward(self, x, edge_index, edge_weight = None, xe = None, m = None):
        xe_emb_list = []
        x = self.dropout(x)
        x = F.relu(self.lin_in(x))
        x0 = x
        if self.args.disen_spec == 1:
            if self.data_stat.batch_size == 0:
                edge_attr = self.edge_attr[edge_index[1]]
            else: 
                edge_attr = self.edge_attr[edge_index[1]].unsqueeze(0)
                edge_attr = torch.vstack([edge_attr for i in range(x.size(0))])
        elif self.args.disen_spec == 0: 
            edge_attr = None
        else: 
            raise NotImplementedError
        for i in range(self.args.num_layers):
            x = self.dropout(x)
            x, edge_attr, conv_xe_out = self.conv(x, edge_index, edge_attr, x0)
            xe_emb_list.append(conv_xe_out)
            x = F.relu(x)
        if self.args.task == 'bio':
            return x, None, xe_emb_list
        x = self.dropout(x)
        x = self.classifier(x)
        return x, None




class disen_HGNN(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(disen_HGNN, self).__init__()
        self.args = args
        self.data_stat = data_stat 
        self.spec = args.disen_spec
        self.dropout = args.dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(disen_hgnn_layer(data_stat = self.data_stat, in_channels = data_stat.exact_num_feature_dim, out_channels = args.hidden, heads = args.heads, spec = self.spec, drop = self.dropout, interpol_ratio= args.interpol_ratio))
        for _ in range(args.num_layers-1):
            self.convs.append(disen_hgnn_layer(data_stat = self.data_stat, in_channels = args.hidden, out_channels = args.hidden, heads = args.heads, spec = self.spec, drop = self.dropout, interpol_ratio= args.interpol_ratio))
        self.convs.append(torch.nn.Linear(args.hidden, args.out_dim))
        self.disen_loss_ratio = self.args.disen_loss_ratio

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
            
    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        for i, conv in enumerate(self.convs[:-1]):
            if self.disen_loss_ratio > 0:
                if i == 0:
                    factor_loss = conv.factor_loss(x, edge_index)
                else: 
                    factor_loss += conv.factor_loss(x, edge_index)
            else: 
                factor_loss = None
            x, conv_xe_out = conv(x, edge_index)
            xe_emb_list.append(conv_xe_out)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.args.task == 'bio':
            return x, factor_loss, xe_emb_list
        if self.args.he_emb:
            return x, factor_loss, xe_emb_list
        x = self.convs[-1](x)
        return x, factor_loss
    
    def get_hyperedge_attention_score(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x_old = x
            x, conv_xe_out = conv(x, edge_index)
            if i == 0:
                att_list = conv.get_hyperedge_attention_score(x_old, edge_index).unsqueeze(0)
            else: 
                att_list = torch.vstack([att_list, conv.get_hyperedge_attention_score(x_old, edge_index).unsqueeze(0)])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return att_list

    
    
class cancer_task_wrapper(torch.nn.Module): # batch_confirmed
    def __init__(self, args, data_stat, model):
        super(cancer_task_wrapper, self).__init__()
        self.args = args
        self.data_stat = data_stat
        self.in_channels = None
        self.out_channels = data_stat.exact_num_labels
        self.dropout = torch.nn.Dropout(args.dropout)
            
        self.model = model(self.args, self.data_stat)
        if self.args.hcl_spec in [3]:
            self.hidden_channels = args.out_dim
            self.fc1 = torch.nn.Linear(self.hidden_channels*self.data_stat.original_num_hyperedges, self.out_channels)           
        else:
            raise NotImplementedError
        
        if self.args.model == 'unigcn2':
            self.reg_params = self.model.reg_params
            self.non_reg_params = self.model.non_reg_params + list(self.fc1.parameters())
        
        self.reset_parameters()
            
    def reset_parameters(self):
        self.model.reset_parameters()
        self.fc1.reset_parameters()
        
        
    def forward(self, x, edge_index, edge_weight = None, xe = None, m = None): 
        x_origin, additional_loss, xe_out_emb  = self.model(x, edge_index, edge_weight, xe)
        xe = torch.mean(xe_out_emb[-1][:,:self.data_stat.original_num_hyperedges],1)       
        x = torch.mean(x_origin, 1)
        in_x = torch.flatten(xe_out_emb[-1][:,:self.data_stat.original_num_hyperedges],1)
        return self.fc1(self.dropout(in_x)), additional_loss

    def get_hyperedge_attention_score(self, x, edge_index):
        return self.model.get_hyperedge_attention_score(x, edge_index)

    def get_hyperedge_emb(self, x, edge_index, edge_weight, xe):
        x, additional_loss, xe_out_emb  = self.model(x, edge_index, edge_weight, xe)
        return xe_out_emb


class hsdn(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(hsdn, self).__init__()
        self.args = args
        self.data_stat = data_stat
        self.heads = args.heads
        self.spec = args.disen_spec
        self.dropout = args.dropout
        self.encoder = torch.nn.Linear(data_stat.exact_num_feature_dim, args.hidden, bias = False)
        self.classifier = torch.nn.Linear(data_stat.exact_num_feature_dim * self.heads, args.out_dim)
        self.factor_size = int(args.hidden // self.heads)
        self.get_xe_mean = get_xe_by_mean_agg(self.data_stat)
        self.conv = hsdn_prop_layer(data_stat, self.args.heads)################
        self.agg = get_xe_by_mean_agg(data_stat)
        
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
            
    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        x0=x
        if self.args.batch_size > 0:
            num_graph = x.size(0)
            x = self.encoder(x).reshape(num_graph, -1, self.heads, self.factor_size)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.softmax(x, dim = -1).reshape(num_graph, -1, self.heads * self.factor_size)
            temp_edge = self.get_xe_mean(x.permute(1,0,2), edge_index)
            temp_edge = temp_edge.reshape(num_graph, -1, self.heads, self.factor_size)
            temp_x = x[:, edge_index[0]].reshape(num_graph, -1,self.heads, self.factor_size)
            att_calc = (temp_edge[:, edge_index[1]] * temp_x).sum(-1)
            att_score = torch_scatter.scatter(att_calc, index = edge_index[1], dim = -2, reduce = 'min') 
        else: 
            x = self.encoder(x).reshape(-1, self.heads, self.factor_size)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.softmax(x, dim = -1).reshape(-1, self.heads * self.factor_size)
            temp_edge = self.get_xe_mean(x, edge_index)
            temp_edge = temp_edge.reshape(-1, self.heads, self.factor_size)
            temp_x = x[edge_index[0]].reshape(-1,self.heads, self.factor_size)
            att_calc = (temp_edge[edge_index[1]] * temp_x).sum(-1)
            att_score = torch_scatter.scatter(att_calc, index = edge_index[1], dim = -2, reduce = 'min') 
            
        
        factor_loss = -(temp_edge *temp_edge).sum(-1) + torch.logsumexp((temp_edge[...,None,:,:] * temp_edge[...,:,None,:]).sum(-1), -1)
        if self.data_stat.batch_size == 0:
            factor_loss = factor_loss.mean()
        else: 
            factor_loss = factor_loss.mean() * num_graph
        x, xe_emb = self.conv(x0, edge_index, att_score)
        if self.args.task == 'bio' or self.args.he_emb:
            return x, factor_loss, [xe_emb]
        x = self.classifier(x)
        return x, factor_loss
    
    def get_hyperedge_attention_score(self, x, edge_index):
        num_graph = x.size(0)
        x = self.encoder(x).reshape(num_graph, -1, self.heads, self.factor_size)
        x = F.softmax(x, dim = -1).reshape(num_graph, -1, self.heads * self.factor_size)
        temp_edge = self.get_xe_mean(x.permute(1,0,2), edge_index)
        temp_edge = temp_edge.reshape(num_graph, -1, self.heads, self.factor_size)
        temp_x = x[:, edge_index[0]].reshape(num_graph, -1,self.heads, self.factor_size)
        att_calc = (temp_edge[:, edge_index[1]] * temp_x).sum(-1)
        att_score = torch_scatter.scatter(att_calc, index = edge_index[1], dim = -2, reduce = 'min') 
        return att_score



class hypergat(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(hypergat, self).__init__()
        self.args = args
        self.data_stat = data_stat
        self.convs = torch.nn.ModuleList()
        self.convs.append(hypergat_layer(data_stat, data_stat.exact_num_feature_dim, args.hidden, self.args.heads))
        for i in range(self.args.num_layers-1):
            self.convs.append(hypergat_layer(data_stat, args.hidden, args.hidden, self.args.heads))
        self.classifier = torch.nn.Linear(args.hidden, args.out_dim)
        self.dropout = self.args.dropout
        
    def reset_parameters(self):
        self.classifier.reset_parameters()
        for conv in self.convs: 
            conv.reset_parameters()
        return 
    
    def forward(self, x, edge_index, edge_weight, xe = None, m = None):
        xe_emb_list = []
        for i,conv in enumerate(self.convs):
            x,xe = conv(x,edge_index)
            xe_emb_list.append(xe)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.task == 'bio':
            return x, None, xe_emb_list
        x = self.classifier(x)
        return x, None


class SHINE(torch.nn.Module):
    def __init__(self, args, data_stat):
        super(SHINE, self).__init__()
        assert data_stat.shine_G != None, "shine_G must be provided"
        self.args = args
        self.num_layers = args.num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.data_stat = data_stat
        self.convs = torch.nn.ModuleList()
        self.spec = self.args.disen_spec # 0 : no reg, 1 : with reg
        
        self.convs.append(shine_dual_attention(data_stat, data_stat.exact_num_feature_dim, args.hidden, self.dropout, args.heads))
        for _ in range(self.num_layers-1):
            self.convs.append(shine_dual_attention(data_stat, args.hidden, args.hidden, self.dropout, args.heads))
        self.classifier = torch.nn.Linear(args.hidden, args.out_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, edge_weight = None, xe = None, m = None):
        if self.data_stat.batch_size > 0:
            return self.batch_forward(x, edge_index, edge_weight, xe, m)
        xe_emb_list=[]
        for i, conv in enumerate(self.convs):
            x, xe = conv(x, xe, edge_index)
            xe_emb_list.append(xe)
        
        node_loss = None
        if self.spec == 1 :
            x2 = F.normalize(x, p=2, dim=-1)
            node_loss  = torch.sum((2 -2*torch.mm(x2, x2.T))*(self.data_stat.shine_G.to_dense()))
        if self.args.task == 'bio':
            return x, node_loss, xe_emb_list
        x = self.classifier(x)
        return x, node_loss 
    
    def batch_forward(self, x, edge_index, edge_weight = None, xe = None, m = None):
        num_graph = x.size(0)
        xe_emb_list=[]
        for i, conv in enumerate(self.convs):
            x, xe = conv(x, xe, edge_index)
            xe_emb_list.append(xe)
        node_loss = None
        if self.spec == 1:
            x2 = F.normalize(x, p=2, dim=-1)
            node_loss = 0
            loss_batch_size = 20
            batch_num = int(num_graph // loss_batch_size)
            if num_graph % loss_batch_size > 0:
                batch_num += 1
            for curr_batch in range(batch_num):
                batch = torch.arange(num_graph)[curr_batch*loss_batch_size:(curr_batch+1)*loss_batch_size]
                node_loss += torch.sum((2 -2*torch.bmm(x2[batch],x2[batch].permute(0,2,1)))*(self.data_stat.shine_G.to_dense().unsqueeze(0)))
        #for i in range(num_graph):
        #    node_loss += torch.sum((2 -2*torch.mm(x2[i], x2[i].T))*(self.data_stat.shine_G.to_dense()))
        #node_loss  = torch.sum((2 -2*torch.bmm(x2,x2.permute(0,2,1)))*(self.data_stat.shine_G.to_dense().unsqueeze(0)))
        if self.args.task == 'bio':
            return x, node_loss, xe_emb_list
        x = self.classifier(x)
        return x, node_loss 

