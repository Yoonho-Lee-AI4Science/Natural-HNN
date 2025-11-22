import numpy as np
import utils
from trainer import Model_Trainer 
import os
import utils
import os
import torch
import data_loader as data_loader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from manager import model_manager
import pdb
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


  
class data_statistic:
    def __init__(self, args):
        self.args = args
        self._init()
        self.config_str = utils.config2string(args)
        
        
    def _init(self):
        self._task = self.args.task
        self._device = f'cuda:{self.args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        self.model_manager_result = model_manager(self.args)
        self.data = self.model_manager_result[1]
        self.data_stat = self.model_manager_result[2]

    def homophily_level(self):
        hyperedge_homo_avg, hyperedge_homo_std = self.average_hyperedge_homophily()
        edge_homo = self.edge_homophily()
        edge_homo_unweighted = self.edge_unweighted_homophily()
        node_homo = self.node_homophily()
        node_homo_unweighted= self.node_unweighted_homophily()
        print("{} : hyperedge_homo --> mean : {}  ||  std : {}  ".format(self.args.dataset, str(hyperedge_homo_avg), str(hyperedge_homo_std)))
        print("{} : hyperedge_homo --> mean : {}".format(self.args.dataset, str(hyperedge_homo_avg)))
        print("{} : edge_homo_level : {}".format(self.args.dataset, str(edge_homo)))
        print("{} : edge_homo_unweighted_level : {}".format(self.args.dataset, str(edge_homo_unweighted)))
        print("{} : node_homo_level : {}".format(self.args.dataset, str(node_homo)))
        print("{} : node_homo_unweighted_level : {}".format(self.args.dataset, str(node_homo_unweighted)))
        
    def average_hyperedge_homophily(self):
        edge_index = self.data.edge_index
        y = self.data.y
        num_labels = y.unique().size(0)
        num_he = self.data_stat.original_num_hyperedges #self.data_stat.exact_num_hyperedges # self.data_stat.original_num_hyperedges
        ahh = 0.0
        for i in range(num_he):
            indices = (edge_index[1] == i).nonzero(as_tuple=True)[0]
            if indices.size(0) == 1:
                ahh += 1.0
                continue
            labels = torch.bincount(y[edge_index[0][indices]])
            total = labels.sum().float()
            sum_labels = (labels ** 2 - labels).sum().float() #/ (labels > 0).sum().float() #num_labels #(labels > 0).sum().float()
            ahh += indices.size(0)*((sum_labels)/(total ** 2 - total)).item()
        ahh /= edge_index.size(1)
        return ahh, None
        
        
    def node_homophily(self):
        self.H = np.zeros((self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        self.H[self.data.edge_index[0], self.data.edge_index[1]-self.data_stat.exact_num_nodes] = 1
        H = torch.from_numpy(self.H).float().to_sparse()
        CE = torch.sparse.mm(H, torch.transpose(H,0,1))
        
        CE_sum = torch.sparse.sum(CE,1).to_dense()
        n = self.data_stat.exact_num_nodes
        diff_label = torch.zeros((n,n))
        for i in range(n):
            diff_label[i] = torch.sign(torch.abs(self.data.y[i]-self.data.y))
        homophily_level = (1-(CE.to_dense() * diff_label).sum(1)/CE_sum).mean()
        
        return homophily_level
    
    def node_unweighted_homophily(self):
        self.H = np.zeros((self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        self.H[self.data.edge_index[0], self.data.edge_index[1]-self.data_stat.exact_num_nodes] = 1
        H = torch.from_numpy(self.H).float().to_sparse()
        CE = torch.sign(torch.sparse.mm(H, torch.transpose(H,0,1)).to_dense())
        
        CE_sum = CE.sum(1)
        n = self.data_stat.exact_num_nodes
        diff_label = torch.zeros((n,n))
        for i in range(n):
            diff_label[i] = torch.sign(torch.abs(self.data.y[i]-self.data.y))
        homophily_level = (1-(CE * diff_label).sum(1)/CE_sum).mean()
        return homophily_level
    
    
    def edge_homophily(self):
        self.H = np.zeros((self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        self.H[self.data.edge_index[0], self.data.edge_index[1]-self.data_stat.exact_num_nodes] = 1
        H = torch.from_numpy(self.H).float().to_sparse()
        CE = torch.sparse.mm(H, torch.transpose(H,0,1))
        
        CE_sum = torch.sparse.sum(CE)
        n = self.data_stat.exact_num_nodes
        diff_label = torch.zeros((n,n))
        for i in range(n):
            diff_label[i] = torch.sign(torch.abs(self.data.y[i]-self.data.y))
        homophily_level = 1-(CE.to_dense() * diff_label).sum()/CE_sum
        
        return homophily_level
    
    def edge_unweighted_homophily(self):
        self.H = np.zeros((self.data_stat.exact_num_nodes, self.data_stat.exact_num_hyperedges))
        self.H[self.data.edge_index[0], self.data.edge_index[1]-self.data_stat.exact_num_nodes] = 1
        H = torch.from_numpy(self.H).float().to_sparse()
        CE = torch.sign(torch.sparse.mm(H, torch.transpose(H,0,1)).to_dense())
        
        CE_sum = CE.sum()
        n = self.data_stat.exact_num_nodes
        diff_label = torch.zeros((n,n))
        for i in range(n):
            diff_label[i] = torch.sign(torch.abs(self.data.y[i]-self.data.y))
        homophily_level = 1-(CE * diff_label).sum()/CE_sum
        return homophily_level

def main():
    args = utils.parse_args()
    args.root_dir = ROOT_DIR
    args.code_dir = ROOT_DIR + "/src"
    args.data_dir = ROOT_DIR + "/dataset/"
    args.check_data = True
    args.model = 'hgnn'
    dt = data_statistic(args)
    dt.homophily_level()
    

if __name__ == "__main__":
    main()