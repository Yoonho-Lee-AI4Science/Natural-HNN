import torch
import os
import os.path as osp
import numpy as np
from torch_geometric.utils import remove_self_loops, is_undirected
import torch.nn.functional as F
import copy
#from utils import set_seed
import random
import utils
from data_loader import dataset_hypergraph
import models
from tqdm import tqdm
import pdb

############# Description for classes defined in this file #####################
# data_manager      : Called by model_manager() class. Load and prepare datasets. 
#                     Prepare things that are required for each model (ex: calculate node degrees, add self-loop). 
# model_manager     : Called by Model_Trainer() class located at trainer.py file. 
#                     Load dataset (by calling data_manager() ), prepare things that are required by the model training. (ex: optimizer, scheduler)
# data_stat_storage : Used to store information about data such as degree of node/hyperedge, dimension of embeddings, number of nodes/hyperedges, etc.
#                     Some models use this information during message passing.
################################################################################

class data_manager():  
    def __init__(self, args = None):
        assert args != None
        assert (args.add_self_loop and args.clique_expansion) == False, "we don't allow adding self loop before clique expansion"
        assert osp.isdir(args.data_dir)
        self.args = args
        self.data = dataset_hypergraph(data_dir = args.data_dir, dataset_name = args.dataset, train_percent = args.train_percent, feature_noise = args.feature_noise)[0]
        self.H = None  # incidence matrix
        self.A = None
        self.D = None
        self.I = None
        self.norms = None
        self.seed_list = None
        self.edge_index_type = 'hypergraph'
        self.requirements = [args.add_self_loop,        # add_self_loop
                             args.clique_expansion,     # clique expansion
                             args.use_incidence_matrix, # create_incidence_matrix
                             args.get_norm,             # get normalization
                             True,                      # split data
                             args.get_shine_G,          # convert hypergraph to graph as described in the shine paper.
                             args.set_mean_xe,          # initialize hyperedge feature as average of nodes within a hyperedge.
                             args.get_min_angle_xe,     # Never used this in our experiment
                             args.convert2graph         # Never used this in our experiment
                             ]
        self.requirements_satisfied = [False, False, False, False, False, False, False, False, False]
        self.manage_data() 
        self.outputs = [self.data, self.seed_list, self.H, self.norms, self.A, self.D, self.I]
        torch.set_num_threads(10)
        
    def manage_data(self):
        self.data.shine_G = 'None'
        self.data.original_num_hyperedges = self.data.exact_num_hyperedges
        if type(self.data.keys) == type([]):  # we added this part as grammar differs by python or pytorch version.
            if (self.data.keys.__contains__('xe') ) == False:
                self.data.xe = 'None'
            if (self.data.keys.__contains__('m') ) == False:
                self.data.m = 'None'
        else: 
            if (self.data.keys().__contains__('xe') ) == False:
                self.data.xe = 'None'
            if (self.data.keys().__contains__('m') ) == False:
                self.data.m = 'None'   
        self.data.edge_weight = None
        self.data.D_e_right = None
        self.data.D_e_left = None
        self.data.D_v_right = None
        self.data.D_v_left = None
        #pdb.set_trace()
        
        if self.requirements[0] and not self.requirements_satisfied[0]:
            self.requirements_satisfied[0] = self.add_self_loop(self.data.edge_index.numpy())
            
        if self.requirements[1] and not self.requirements_satisfied[1]:
            self.requirements_satisfied[1] = self.clique_expansion(self.data.edge_index.numpy())
            
        if self.requirements[2] and not self.requirements_satisfied[2]:
            self.incidence_matrix(self.data.edge_index.numpy())
            
        if self.requirements[3] and not self.requirements_satisfied[3]:
            self.requirements_satisfied[3] = self.norm(self.data.edge_index.numpy(), alpha = self.args.hnhn_alpha , beta = self.args.hnhn_beta)
            
        if self.requirements[4] and not self.requirements_satisfied[4]:
            self.requirements_satisfied[4] = self.split_data()
            
        self.data.edge_index[1] -= self.data.exact_num_nodes   
        
        if self.requirements[5] and not self.requirements_satisfied[5]:    
            self.requirements_satisfied[5] = self.get_shine_G(self.data.edge_index)
            
        if self.requirements[6] and not self.requirements_satisfied[6]:    
            self.requirements_satisfied[6] = self.set_mean_xe()
            
        if self.requirements[7] and not self.requirements_satisfied[7]:    
            self.requirements_satisfied[7] = self.set_min_angle_xe()
            
        if self.requirements[8] and not self.requirements_satisfied[8]:    
            self.requirements_satisfied[8] = self.convert_2_graph()
        
        assert self.requirements == self.requirements_satisfied, "requirements not satisifed"
        
        return 
    
    
    def add_self_loop(self, edge_index): # adding self-hyperedges(hyperedge connecting node to itself only)
        num_nodes = self.data.exact_num_nodes
        num_hyperedges = self.data.exact_num_hyperedges
        assert (num_nodes + num_hyperedges -1) == edge_index[1].max(), "num_hyperedges mismatch @ report from add_self_loop function "
        assert edge_index[0].max() <= (num_nodes - 1), "num_node mismatch @ report from add_self_loop function"
        
        hyperedge_sorted = edge_index[:, np.argsort(edge_index[1])]             # sorted edge_index w.r.t hyperedge number(index)
        _ , counts = np.unique(edge_index[1], return_counts=True)
        index = np.asarray([0]+np.cumsum(counts).tolist()[:-1])[counts == 1]    # The indices of self-loops(hyperedge) in hyperedge_sorted
        already_exist = hyperedge_sorted[0,index ]                              # the nodes that already has self loop
        
        new_edge_mask = np.ones(num_nodes).astype(bool)                         # The mask for selecting nodes that need self loop
        new_edge_mask[already_exist] = 0
        new_edge_list_temp = np.arange(num_nodes)[new_edge_mask]
        new_edge_list = np.vstack((new_edge_list_temp, np.arange(new_edge_list_temp.shape[0]) + num_nodes + num_hyperedges))
        edge_index = np.hstack((edge_index, new_edge_list))
        assert edge_index.shape[0] == 2
        
        num_hyperedges += new_edge_list.shape[1]
        self.data.edge_index = torch.LongTensor(np.unique(edge_index.T, axis = 0).T)  # This removes redundancy and also sorts w.r.t nodes
        self.data.exact_num_hyperedges = num_hyperedges

        return True
    
    
    def clique_expansion(self, edge_index):
        assert self.requirements[1] == True, "clique expansion already done"
        assert self.requirements[0] == False, "we don't allow adding self loop before clique expansion"
        num_nodes = self.data.exact_num_nodes
        num_hyperedges = self.data.exact_num_hyperedges
        
        temp_edge_index = edge_index.copy()
        temp_edge_index[1] -= num_nodes
        sparse_H = torch.sparse_coo_tensor(temp_edge_index, np.ones(temp_edge_index.shape[1]), (num_nodes, num_hyperedges))
        sparse_G = torch.sparse.mm(sparse_H, torch.transpose(sparse_H,0,1))
        new_edge_index = sparse_G.coalesce().indices()
        new_edge_weight = sparse_G.coalesce().values()
        new_edge_index, new_edge_weight = remove_self_loops(new_edge_index, new_edge_weight)
        assert is_undirected(new_edge_index, new_edge_weight)
        
        self.data.edge_index = torch.LongTensor(new_edge_index)
        self.data.edge_weight = torch.FloatTensor(new_edge_weight)
        self.requirements_satisfied[1] = True
        self.edge_index_type = 'graph'
        
        return True
    

    def split_data(self): # label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False            
        train_percent = self.args.train_percent
        valid_percent = self.args.valid_percent
        test_percent = self.args.test_percent
        num_repeat = self.args.num_repeat
        num_answers = self.data.y.size(0)
        assert self.seed_list == None, "seed list must be None @ split_data"
        
        random.seed(0)
        self.seed_list = random.sample(list(set([random.randint(0,200) for i in range(self.args.num_repeat*2)])),self.args.num_repeat)
        train_mask = torch.zeros(num_repeat, num_answers)
        val_mask = torch.zeros(num_repeat, num_answers)
        test_mask = torch.zeros(num_repeat, num_answers)
        
        if self.args.use_default_split: # Use splits provided by each dataset.
            assert self.data.default_train_mask != None, "default mask must not be None, check for data_loader functions"
            assert self.data.default_val_mask != None, "default mask must not be None, check for data_loader functions"
            assert self.data.default_test_mask != None, "default mask must not be None, check for data_loader functions"
            for i in range(num_repeat): 
                train_mask[i, self.data.default_train_mask] = 1
                val_mask[i, self.data.default_val_mask] = 1
                test_mask[i, self.data.default_test_mask] = 1   
        elif self.args.use_balanced_split: # considering label distribution.
            class_counts = [a.item() for a in torch.bincount(self.data.y)]
            for i in range(num_repeat):
                utils.set_seed(self.seed_list[i])
                for j in range(self.data.exact_num_labels):
                    class_places = (self.data.y == j).nonzero(as_tuple=True)[0]
                    perm = torch.as_tensor(np.random.permutation(class_counts[j]))
                    train_mask[i, class_places[perm[:int(class_counts[j]*train_percent)]]] = 1
                    val_mask[i, class_places[perm[(-int(class_counts[j]*test_percent) - int(class_counts[j]*valid_percent)):-int(class_counts[j]*test_percent)]]] = 1
                    test_mask[i, class_places[perm[-int(class_counts[j]*test_percent):]]] = 1    
                    
        else: # Just random split.
            for i in range(num_repeat):
                utils.set_seed(self.seed_list[i])
                perm = torch.as_tensor(np.random.permutation(num_answers))
                train_mask[i, perm[:int(num_answers*train_percent)]] = 1
                val_mask[i, perm[int(num_answers*train_percent):(int(num_answers*train_percent)+int(num_answers*valid_percent))]] = 1
                test_mask[i, perm[int(num_answers*train_percent)+int(num_answers*valid_percent):(int(num_answers*train_percent)+int(num_answers*valid_percent)+int(num_answers*test_percent))]] = 1                
            
        self.data.train_mask = train_mask.bool()
        self.data.val_mask = val_mask.bool()
        self.data.test_mask = test_mask.bool()
        
        return True
    
    
    def norm(self, edge_index, alpha = None, beta = None): # get degree-normalized edge weights. Differ by model.
        num_nodes = self.data.exact_num_nodes
        num_hyperedges = self.data.exact_num_hyperedges
        
        if self.edge_index_type == 'hypergraph':
            self.data.edge_weight = torch.FloatTensor(np.ones(edge_index.shape[1]))
            temp_edge_index = edge_index.copy()
            temp_edge_index[1] -= num_nodes
            sparse_H = torch.sparse_coo_tensor(temp_edge_index, np.ones(temp_edge_index.shape[1]), (num_nodes, num_hyperedges))
            D_E = torch.sparse.sum(sparse_H, 0).to_dense().float()
            D_V = torch.sparse.sum(sparse_H, 1).to_dense().float()
            # D_e_left
            if self.args.model in ['hnhn']:
                self.data.D_e_left = D_E ** alpha   # D_e_alpha
            else: 
                self.data.D_e_left = 'None'
            
            # D_e_right
            if self.args.model in ['hcha', 'hgnn', 'shine', 'disen_hgnn', "edhnn", "hsdn", "hypergat"]:
                self.data.D_e_right = D_E ** (-1)
                self.data.D_e_right[self.data.D_e_right == float("inf")] = 0
            elif self.args.model in ['unigcn2']:
                self.data.D_e_right = D_E ** (-3/2)
                self.data.D_e_right[self.data.D_e_right == float("inf")] = 0
            elif self.args.model in ['hnhn']: # D_e_beta_inverse
                D_e_right = np.zeros(num_hyperedges)
                for i in range(num_hyperedges):
                    D_e_right[i] = torch.sum(D_V[temp_edge_index[0, np.where(temp_edge_index[1] == i)[0]]] ** beta)
                self.data.D_e_right = torch.from_numpy(D_e_right).float()
                self.data.D_e_right = self.data.D_e_right ** (-1)
                self.data.D_e_right[self.data.D_e_right == float("inf")] = 0
            else: 
                self.data.D_e_right = 'None'
                
            # D_v_left
            if self.args.model in ['hcha', 'disen_hgnn', "edhnn"]:
                self.data.D_v_left = D_V ** (-1)
                self.data.D_v_left[self.data.D_v_left == float("inf")] = 0
            elif self.args.model in ['hgnn', 'unigcn2', 'shine', "hypergat"]:
                self.data.D_v_left = D_V ** (-1/2)
                self.data.D_v_left[self.data.D_v_left == float("inf")] = 0
            elif self.args.model in ['hnhn']: # D_v_alpha_inverse
                D_v_left = np.zeros(num_nodes)
                for i in range(num_nodes):
                    D_v_left[i] = torch.sum(D_E[temp_edge_index[1, np.where(temp_edge_index[0] == i)[0]]] ** alpha)
                self.data.D_v_left = torch.from_numpy(D_v_left).float()
                self.data.D_v_left = self.data.D_v_left ** (-1)
                self.data.D_v_left[self.data.D_v_left == float("inf")] = 0
            else: 
                self.data.D_v_left = 'None'
            
            # D_v_right
            if self.args.model in ['hgnn', 'shine', "hypergat"]:
                self.data.D_v_right = self.data.D_v_left
            elif self.args.model in ['hnhn']: # D_v_beta
                self.data.D_v_right = D_V ** beta
            elif self.args.model in ['hsdn']: # D_v_beta
                self.data.D_v_right = D_V ** (-1)
                self.data.D_v_right[self.data.D_v_right == float("inf")] = 0
            else: 
                self.data.D_v_right = 'None'
        else: 
            raise ValueError("Error in edge_index type. check for data manager class")
    
        # we are going to set 4 norms
        return True
    
    def incidence_matrix(self, edge_index):
        if self.H is None or (self.requirements_satisfied[2] == False):
            self.H = np.zeros((self.data.exact_num_nodes, self.data.exact_num_hyperedges))
            self.H[edge_index[0], edge_index[1]-self.data.exact_num_nodes] = 1
        self.requirements_satisfied[2] = True
        return self.H
    
    def get_shine_G(self, edge_index):
        sparse_H = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (self.data.exact_num_nodes, self.data.exact_num_hyperedges)) 
        dv_matrix = torch.sparse_coo_tensor(torch.vstack((torch.arange(self.data.exact_num_nodes),torch.arange(self.data.exact_num_nodes))), self.data.D_v_left,(self.data.exact_num_nodes, self.data.exact_num_nodes))
        de_matrix = torch.sparse_coo_tensor(torch.vstack((torch.arange(self.data.exact_num_hyperedges),torch.arange(self.data.exact_num_hyperedges))), self.data.D_e_right,(self.data.exact_num_hyperedges, self.data.exact_num_hyperedges))
        self.data.shine_G = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(dv_matrix, sparse_H),de_matrix),torch.transpose(sparse_H,0,1)), dv_matrix)
        return True
    
    
    def set_mean_xe(self):
        sparse_HT = torch.sparse_coo_tensor(self.data.edge_index.flip([0]), torch.ones(self.data.edge_index.shape[1]), (self.data.exact_num_hyperedges, self.data.exact_num_nodes))
        row_sum_inv = 1/torch.sparse.sum(sparse_HT, dim = 1).to_dense().unsqueeze(-1)
        if len(self.data.x.size()) == 3:
            xe = []
            for i in tqdm(range(self.data.x.size(0))):
                xe.append((torch.mm(sparse_HT,self.data.x[i])*row_sum_inv).unsqueeze(0))
            xe = torch.vstack(xe)
        else:
            xe = torch.mm(sparse_HT,self.data.x)*row_sum_inv
        self.data.xe = xe
        return True
    
    
    def set_min_angle_xe(self):
        
        self.get_xe = models.get_min_angle_xe_rep(self.args, self.data)
        xe, loss, angles = self.get_xe.train()
        self.data.xe = xe
        pdb.set_trace()
        return True
    
    def convert_2_graph(self):
        assert self.requirements_satisfied[2] == True
        assert not (self.H  is None)
        H = torch.from_numpy(self.H).float().to_sparse()
                
        A_beta = torch.sparse.mm(H, torch.transpose(H,0,1))
        I = torch.eye(self.H.shape[0]).to_sparse()
        A_beta += I
        D_beta=torch.diag(torch.sparse.sum(A_beta,1).to_dense()).to_sparse()**(-1/2)
        A_beta=torch.sparse.mm(torch.sparse.mm(D_beta, A_beta), D_beta)
        D_beta=torch.diag(torch.sparse.sum(A_beta,1).to_dense()).to_sparse()
        

        DE=torch.diag((torch.sparse.sum(H,0)**(-1)).to_dense()).to_sparse()
        A_gamma=torch.sparse.mm(torch.sparse.mm(H, DE), torch.transpose(H,0,1))
        A_gamma += I
        D_gamma=torch.diag(torch.sparse.sum(A_gamma,1).to_dense()).to_sparse()**(-1/2)
        A_gamma=torch.sparse.mm(torch.sparse.mm(D_gamma, A_gamma), D_gamma)
        D_gamma=torch.diag(torch.sparse.sum(A_gamma,1).to_dense()).to_sparse()
        
        self.A = [A_beta, A_gamma]
        self.D = [D_beta, D_gamma]
        self.I = I    
        self.data.D_v_right = 'None'
        self.data.D_v_left = 'None'
        self.data.D_e_right = 'None'
        self.data.D_e_left = 'None'
        
                
        return True
    
    
    def __getitem__(self, index):
        assert index < 7
        return self.outputs[index]
    
    


class model_manager():
    def __init__(self, args):
        self.args = args
        MODEL = {'hgnn':models.HGNN, 'hcha':models.HCHA, 'unigcn2':models.UniGCNII, 'hnhn':models.HNHN,'alldeepset':models.SetGNN, 'allsettransformer':models.SetGNN, 'disen_hgnn':models.disen_HGNN, "edhnn":models.EquivSetGNN, "hsdn":models.hsdn, "hypergat":models.hypergat, "shine":models.SHINE } # alldeepset, allsettransformer
        
        # data processing args
        self.args.add_self_loop = False # add_self_loop
        self.args.clique_expansion = False # clique expansion
        self.args.use_incidence_matrix = False # create_incidence_matrix
        self.args.get_norm = False # get normalization 
        self.args.get_shine_G = False # only for shine model. Convert hypergraph to graph as described in the shine paper.
        self.args.set_mean_xe = False # initialize hyperedge embedding by mean aggregating nodes within hyperedge
        self.args.get_min_angle_xe = False # Never used this in our experiment
        self.args.convert2graph = False # Never used this in our experiment
        
        # get data
        self.set_data_args()
        data_managed = data_manager(self.args)
        self.data = data_managed[0]
        self.seed_list = data_managed[1]
        self.norms = data_managed[3]
        self.A = data_managed[4]
        self.D = data_managed[5]
        self.I = data_managed[6]
        
        # self.data_stat stores degree of each node/hyperedge, number of labels/nodes/hyperedges/dimensions
        # Some models use this information during message passing.
        self.set_data_stat()
        
        # set output representation dimension. Differ by task or model
        self.args.out_dim = self.data_stat.exact_num_labels
        # bio task : cancer subtyp classification
        # basic task : node classification. maybe standard benchmark dataset.
        if self.args.task == 'bio':
            self.args.out_dim = self.args.hidden
            if self.args.model in ['hsdn']:
                self.args.out_dim = self.data.exact_num_feature_dim * self.args.heads
        if self.args.task == 'basic':    
            self.model = MODEL[args.model](self.args, self.data_stat)
        elif self.args.task == 'bio': # cancer subtype classification task uses additional MLP for hypergraph classification. Thus we need a model(wrapper) that contains MPNN model and Classification MLP.
            self.model = models.cancer_task_wrapper(self.args, self.data_stat, MODEL[args.model])
        else: 
            raise NotImplementedError
        optimizer = self.set_optimizer()
        self.delete_data_attributes() # save memory
        self.manager_outputs = [self.model, self.data, self.data_stat, self.args, optimizer, self.seed_list]

    def set_data_stat(self):
        # self.data_stat stores degree of each node/hyperedge, number of labels/nodes/hyperedges/dimensions
        # Some models use this information during message passing.
        self.data_stat = data_stat_storage()
        self.device = f'cuda:{self.args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        self.data_stat.exact_num_feature_dim = self.data.exact_num_feature_dim
        self.data_stat.exact_num_hyperedges = self.data.exact_num_hyperedges
        self.data_stat.exact_num_labels = self.data.exact_num_labels
        self.data_stat.exact_num_nodes = self.data.exact_num_nodes
        self.data_stat.train_percent =  self.data.train_percent
        self.data_stat.D_v_right = self.data.D_v_right if self.data.D_v_right == 'None' else self.data.D_v_right.to(self.device)
        self.data_stat.D_v_left = self.data.D_v_left if self.data.D_v_left == 'None' else self.data.D_v_left.to(self.device)
        self.data_stat.D_e_right = self.data.D_e_right if self.data.D_e_right == 'None' else self.data.D_e_right.to(self.device)
        self.data_stat.D_e_left = self.data.D_e_left if self.data.D_e_left == 'None' else self.data.D_e_left.to(self.device)
        self.data_stat.shine_G = None if self.data.shine_G == 'None' else self.data.shine_G.to(self.device)
        self.data_stat.consistency_loss_ratio = self.args.consistency_loss_ratio
        self.data_stat.mean_xe = self.data.xe.to(self.device) if self.data.xe != 'None' else 'None'
        self.data_stat.original_num_hyperedges = self.data.original_num_hyperedges
        self.data_stat.batch_size = self.args.batch_size
        if self.A is None:
            self.data_stat.A = None
            self.data_stat.D = None
        else: 
            self.data_stat.A = [self.A[i].to(self.device) for i in range(2)]
            self.data_stat.D = [self.D[i].to(self.device) for i in range(2)]
            self.data_stat.I = self.I.to(self.device)
        if type(self.data.keys) == type([]):
            if (self.data.keys.__contains__('exact_num_hypergraphs') ) == True:
                self.data_stat.exact_num_hypergraphs = self.data.exact_num_hypergraphs
        else: 
            if (self.data.keys().__contains__('exact_num_hypergraphs') ) == True:
                self.data_stat.exact_num_hypergraphs = self.data.exact_num_hypergraphs

    
    def delete_data_attributes(self): # save memory
        delattr(self.data, "D_e_left")
        delattr(self.data, "D_v_left")
        delattr(self.data, "D_e_right")
        delattr(self.data, "D_v_right")
        delattr(self.data, "shine_G")
        delattr(self.data, "train_percent")
        delattr(self.data, "exact_num_feature_dim")
        delattr(self.data, "exact_num_hyperedges")
        delattr(self.data, "exact_num_labels")
        delattr(self.data, "exact_num_nodes")  
        delattr(self.data, "default_train_mask")    
        delattr(self.data, "default_val_mask")   
        delattr(self.data, "default_test_mask")      
        

    def set_optimizer(self):
        scheduler = None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd) 
        if self.args.model  == 'unigcn2':
            optimizer = torch.optim.Adam([ dict(params=self.model.reg_params, weight_decay=0.01), dict(params=self.model.non_reg_params, weight_decay=5e-4)], lr=0.01)
        if self.args.model == 'shine' and self.args.task == 'bio':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=50)
        return optimizer, scheduler
    
    def set_data_args(self): # depending on model
        if self.args.model in ['alldeepset','allsettransformer','hnhn','hcha','hgnn','unigcn2', 'disen_hgnn', "edhnn", "hsdn"]: 
            self.args.add_self_loop = True
        if self.args.model in ['hnhn','hgnn','hcha','unigcn2', 'alldeepset','allsettransformer','shine', 'disen_hgnn', "edhnn", "hsdn", "hypergat"]:
            self.args.get_norm = True
        if self.args.model != 'hnhn':
            self.args.hnhn_alpha = None
            self.args.hnhn_beta = None
        if self.args.model in ['shine']:
            self.args.get_shine_G = True
            self.args.set_mean_xe = True
        return 
        
        
    def __getitem__(self, index):
        assert index < 7
        return self.manager_outputs[index]
    




class data_stat_storage(): 
    # storing only data statistics. 
    # Stores degree of each node/hyperedge, number of labels/nodes/hyperedges/dimensions
    # Some models use this information during message passing.
    def __init__(self):
        self._exact_num_nodes = None
        self._exact_num_hyperedges = None
        self._exact_num_labels = None
        self._exact_num_feature_dim = None
        self._train_percent = None
        self._D_v_right = None
        self._D_v_left = None
        self._D_e_right = None
        self._D_e_left = None
        self._shine_G = None
        self._consistency_loss_ratio = None
        self._mean_xe = None
        self._exact_num_hypergraphs = None
        self._original_num_hyperedges = None
        self._batch_size = None
        self._A = None
        self._D = None
        self._I = None
        
    @property
    def exact_num_nodes(self):
        return self._exact_num_nodes
    
    @exact_num_nodes.setter
    def exact_num_nodes(self, value):
        self._exact_num_nodes = value
        
    @property
    def exact_num_hyperedges(self):
        return self._exact_num_hyperedges
    
    @exact_num_hyperedges.setter
    def exact_num_hyperedges(self, value):
        self._exact_num_hyperedges = value
        
    @property
    def exact_num_labels(self):
        return self._exact_num_labels
    
    @exact_num_labels.setter
    def exact_num_labels(self, value):
        self._exact_num_labels = value
        
    @property
    def exact_num_feature_dim(self):
        return self._exact_num_feature_dim
    
    @exact_num_feature_dim.setter
    def exact_num_feature_dim(self, value):
        self._exact_num_feature_dim = value
    
    @property 
    def train_percent(self):
        return self._train_percent
    
    @train_percent.setter
    def train_percent(self, value):
        self._train_percent = value
        
    @property
    def D_v_right(self):
        return self._D_v_right
    
    @D_v_right.setter
    def D_v_right(self, value):
        self._D_v_right = value
        
    @property
    def D_v_left(self):
        return self._D_v_left
    
    @D_v_left.setter
    def D_v_left(self, value):
        self._D_v_left = value
        
    @property
    def D_e_right(self):
        return self._D_e_right
    
    @D_e_right.setter
    def D_e_right(self, value):
        self._D_e_right = value
        
    @property
    def D_e_left(self):
        return self._D_e_left
    
    @D_e_left.setter
    def D_e_left(self, value):
        self._D_e_left = value
    
    @property
    def shine_G(self):
        return self._shine_G
    
    @shine_G.setter
    def shine_G(self, value):
        self._shine_G = value
        
    @property
    def consistency_loss_ratio(self):
        return self._consistency_loss_ratio
    
    @consistency_loss_ratio.setter
    def consistency_loss_ratio(self, value):
        self._consistency_loss_ratio = value 
        
    @property
    def mean_xe(self):
        return self._mean_xe
    
    @mean_xe.setter
    def mean_xe(self, value):
        self._mean_xe = value 
        
    @property
    def exact_num_hypergraphs(self):
        return self._exact_num_hypergraphs
    
    @exact_num_hypergraphs.setter
    def exact_num_hypergraphs(self, value):
        self._exact_num_hypergraphs = value 
        
    @property
    def original_num_hyperedges(self):
        return self._original_num_hyperedges
    
    @original_num_hyperedges.setter
    def original_num_hyperedges(self, value):
        self._original_num_hyperedges = value
                
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
                
    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, value):
        self._A = value
                
    @property
    def D(self):
        return self._D
    
    @D.setter
    def D(self, value):
        self._D = value
                
    @property
    def I(self):
        return self._I
    
    @I.setter
    def I(self, value):
        self._I = value