from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
import pickle
import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl
import utils
import pdb

def index_fixer(index_data): # change 1D ndarray to have 0 ~ n-1 values when they have n unique values. ex) [1,4,4,6,3,8,1] --> [0,2,2,3,1,4,0]
    if type(index_data) == np.ndarray and len(index_data.shape) == 1: # 1 dimensional ndarray
        length = index_data.shape[0]
        index_kinds, indices = np.unique(index_data, return_inverse = True)
        fixed_index_data = np.arange(index_kinds.shape[0])[indices]
        
        # check size correctness
        assert length == fixed_index_data.shape[0]
        return fixed_index_data
    raise RuntimeError("index_fixer function is not intended to handle this type of data or shape")
    
    return None

class dataset_hypergraph(InMemoryDataset):
    def __init__(self, data_dir, dataset_name = None, train_percent = 0.01, feature_noise = None, transform = None, pre_transform = None):
        
        existing_dataset = ['20newsw100', 'Mushroom', 'zoo', 
                            'NTU2012', 'ModelNet40', 
                            'coauthor_cora', 'coauthor_dblp',                           # load_citation_dataset
                            'amazon_reviews', 'walmart_trips', 'house_committees',      # load_cornell_dataset  --> add feature noise
                            'walmart_trips_100', 'house_committees_100', 'senate_committees_100', 'congress_bills_100',                # load_cornell_dataset  --> add feature noise
                            'cora', 'citeseer', 'pubmed',                               # load_citation_dataset'
                            'brca','stad', 'sarc', 'lgg', 'kipan', 'nsclc', 'hnsc', 'cesc'
                            ]
        existing_dataset = list(map(str.lower, existing_dataset))
        self.dataset_name = dataset_name
        self.feature_noise = feature_noise
        self.train_percent = train_percent
        self.data_dir = data_dir
        self.dataset_dir = None
        if self.dataset_name in ['coauthor_cora', 'coauthor_dblp']:
            self.dataset_dir = self.data_dir+"coauthorship/"+self.dataset_name.split('_')[-1]+"/"
        elif self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            self.dataset_dir = self.data_dir+"cocitation/"+self.dataset_name+"/"
        elif self.dataset_name in ['amazon_reviews', 'walmart_trips', 'house_committees', 'walmart_trips_100', 'house_committees_100', 'senate_committees_100', 'congress_bills_100']:
            self.dataset_dir = self.data_dir+'_'.join(self.dataset_name.split("_")[:2])+"/"      
        elif self.dataset_name in ['20newsw100', 'mushroom', 'zoo','ntu2012', 'modelnet40' ]:
            self.dataset_dir = self.data_dir+dataset_name+"/"    
        elif self.dataset_name in ['brca', 'stad', 'sarc', 'lgg', 'kipan', 'nsclc', 'hnsc', 'cesc']:
            self.dataset_dir = self.data_dir+dataset_name+"/"
        else:
            raise ValueError('such dataset does not exist')       
        
        assert dataset_name in existing_dataset, "check whether dataset name is correct"
        assert self.data_dir != None, "data_dir must not be None"
        assert osp.isdir(self.data_dir), "given data_dir is not valid"
        assert osp.isdir(self.dataset_dir), "dataset_dir is not valid"
        
        # inherit InMemoryDataset class. This will automatically activate process(self) function.
        super().__init__(root = self.dataset_dir, transform = transform, pre_transform = pre_transform, pre_filter = None) 
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
        
    @property
    def processed_file_names(self): # to skip processing, processed file must be found in processed_file_names
        return [f'data_noise_{self.feature_noise}.pt'] if self.feature_noise is not None else ['data.pt']
    
    def process(self):
        if self.dataset_name in ['coauthor_cora', 'coauthor_dblp', 'cora', 'citeseer', 'pubmed']:
            loaded_data = load_citation(path = self.raw_dir+"/", train_percent = self.train_percent)
        elif self.dataset_name in ['amazon_reviews', 'walmart_trips', 'house_committees', 'walmart_trips_100', 'house_committees_100', 'senate_committees_100', 'congress_bills_100']:
            dim_feat = None if len(self.dataset_name.split('_')) == 2 else int(self.dataset_name.split('_')[-1])
            loaded_data = load_cornell(path = self.raw_dir+"/", dataset_name = '_'.join(self.dataset_name.split('_')[:2]), feature_noise = self.feature_noise, feature_dim = dim_feat, train_percent = 0.025)
        elif self.dataset_name in ['20newsw100', 'mushroom', 'zoo','ntu2012', 'modelnet40' ]:
            loaded_data = load_LE(path = self.raw_dir+"/", dataset_name = self.dataset_name,  train_percent = self.train_percent)
        elif self.dataset_name in ['brca', 'stad', 'sarc', 'lgg', 'kipan', 'nsclc', 'hnsc', 'cesc']:
            loaded_data = load_cancer(path=self.raw_dir+"/", dataset_name = self.dataset_name, train_percent=self.train_percent)
        else:
            raise NotImplementedError
        
        data = loaded_data if self.pre_transform is None else self.pre_transform(loaded_data)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self):
        return '{}()'.format(self.dataset_name)


def load_cancer(path=None, dataset_name=None, train_percent = 0.025):
    assert osp.isdir(path)
    dataset_name_up = dataset_name.upper()
    features = np.load(path+dataset_name_up+"_data_all.npy")
    num_graph, num_node, feature_dim = features.shape
    edge_index = np.load(path+"edge_index_raw.npy")
    edge_index = np.unique(edge_index.T, axis = 0).T
    labels = [int(a) for a in open(path+dataset_name_up+"_labels.txt",'r').read().split('\n')[:-1]]
    num_classes = np.unique(labels).shape[0]
    assert num_graph == len(labels)
    num_hyperedges = edge_index[1].max()+1
    edge_index[1] += num_node
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    edge_index = torch.LongTensor(edge_index)
    data = Data(x = features, edge_index = edge_index, y = labels)
    
    data.exact_num_nodes = num_node
    data.exact_num_hyperedges = num_hyperedges
    data.original_num_hyperedges = num_hyperedges
    data.exact_num_labels = num_classes
    data.exact_num_feature_dim = feature_dim
    data.train_percent = train_percent
    data.exact_num_hypergraphs = num_graph
    data.m = None
    data.xe = None
    
    data.default_train_mask = None
    data.default_val_mask = None
    data.default_test_mask = None
    return data


def load_citation(path=None, train_percent = 0.025):    
    assert osp.isdir(path)
    
    ### load data
    with open(osp.join(path, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()
    with open(osp.join(path, 'labels.pickle'), 'rb') as f:
        labels = index_fixer(np.asarray(pickle.load(f)))
    with open(osp.join(path, 'hypergraph.pickle'), 'rb') as f: # dictionary of form {"hyperedge_name(string)": [list of nodes(int) in the hyperedge]}
        hypergraph = pickle.load(f)
    num_node, feature_dim = features.shape
    num_classes = np.unique(labels).shape[0]
    assert num_node == labels.shape[0]

    edge_idx = num_node
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)
        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size
        edge_idx += 1
    edge_index = np.asarray([node_list, edge_list]).astype(int)
    num_hyperedges = edge_index[1].max()+1-num_node
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    edge_index = torch.LongTensor(np.unique(edge_index.T, axis = 0).T)
    data = Data(x = features, edge_index = edge_index, y = labels)
    
    data.exact_num_nodes = num_node
    data.exact_num_hyperedges = num_hyperedges
    data.exact_num_labels = num_classes
    data.exact_num_feature_dim = feature_dim
    data.train_percent = train_percent
    data.m = None
    data.xe = None
    
    data.default_train_mask = None
    data.default_val_mask = None
    data.default_test_mask = None
    
    return data

def load_cornell(path = None, dataset_name = None, feature_noise = 0.1, feature_dim = None, train_percent = 0.025):
    assert osp.isdir(path)
    assert dataset_name != None
    #pdb.set_trace()
    ### load data files & preprocess labels & hyperedges
    df_labels = pd.read_csv(osp.join(path, f'{dataset_name}_node_labels.txt'), names = ['node_label'])
    num_node = df_labels.shape[0]
    labels = df_labels.values.flatten()
    labels -= labels.min()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_node, num_classes))

    features[np.arange(num_node), labels] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    utils.set_seed(0)
    features = np.random.normal(features, feature_noise, features.shape)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = osp.join(path, f'{dataset_name}_hyperedges.txt')
    node_list = []
    he_list = []
    he_id = num_node

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    #pdb.set_trace()
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = np.asarray([node_list, he_list]).astype(int)
    edge_index = np.unique(edge_index.T, axis = 0).T
    num_hyperedges = edge_index[1].max()+1 - num_node
    edge_index = torch.LongTensor(edge_index)
    data = Data(x = features, edge_index = edge_index, y = labels)
    
    data.exact_num_nodes = num_node
    data.exact_num_hyperedges = num_hyperedges
    data.exact_num_labels = num_classes
    data.exact_num_feature_dim = feature_dim
    data.train_percent = train_percent
    data.m = None
    data.xe = None
    
    data.default_train_mask = None
    data.default_val_mask = None
    data.default_test_mask = None
    
    return data



def load_LE(path = None, dataset_name = None,  train_percent = 0.025):
    assert osp.isdir(path)
    assert dataset_name != None
    assert osp.isfile(path+dataset_name+".content")
    assert osp.isfile(path+dataset_name+".edges")
    
    ### open dataset
    idx_features_labels = np.genfromtxt(path+dataset_name+".content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path+dataset_name+".edges", dtype=np.int32)
    edge_index = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape).T
    
    num_node = edge_index[0].max() + 1
    num_hyperedges = edge_index[1].max() - num_node + 1
    edge_index = np.unique(edge_index.T, axis = 0).T # remove duplicate
    
    feature_dim = features.shape[1]
    num_classes = np.unique(labels).shape[0]   
    
    data = Data(x = torch.FloatTensor(np.array(features[:num_node].todense())), edge_index = torch.LongTensor(edge_index), y = torch.LongTensor(labels[:num_node]))
    
    data.exact_num_nodes = num_node
    data.exact_num_hyperedges = num_hyperedges
    data.exact_num_labels = num_classes
    data.exact_num_feature_dim = feature_dim
    data.train_percent = train_percent
    data.m = None
    data.xe = None
    
    data.default_train_mask = None
    data.default_val_mask = None
    data.default_test_mask = None
        
    return data
    