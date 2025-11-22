import numpy as np
import utils
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from manager import model_manager
import pdb
from sklearn.metrics import f1_score, accuracy_score
import shap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



class cancer_classifier_only(torch.nn.Module): 
    def __init__(self, args, data_stat):
        super(cancer_classifier_only, self).__init__()
        self.args = args
        self.data_stat = data_stat
        self.dropout = torch.nn.Dropout(args.dropout)
        self.hidden_channels = args.out_dim
        self.out_channels = data_stat.exact_num_labels
        self.fc1 = torch.nn.Linear(self.hidden_channels*self.data_stat.original_num_hyperedges, self.out_channels)
        self.reset_parameters()
            
    def reset_parameters(self):
        self.fc1.reset_parameters()
        
    def forward(self, flattend_hyperedges): 
        return self.fc1(self.dropout(flattend_hyperedges))
    
class shap_calculate:
    def __init__(self, args):
        self.args = args
        self._init()
        self.config_str = utils.config2string(args)
        self._metric = self.args.metric
        
        
    def _init(self):
        self._task = self.args.task
        self._device = f'cuda:{self.args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        self.model_manager_result = model_manager(self.args)
        self._model = self.model_manager_result[0].to(self._device)
        self._dataset = self.model_manager_result[1]
        self._data_stat = self.model_manager_result[2]
        self.args = self.model_manager_result[3]
        self._random_seed_numbers_list = self.model_manager_result[5]
        self.disen_loss_ratio = self.args.disen_loss_ratio


    def calculate(self, iterations=0):
        utils.set_seed(seed = self._random_seed_numbers_list[iterations])
        classifier = cancer_classifier_only(self.args, self._data_stat)
        he_emb = torch.Tensor(np.load(self.args.root_dir+'/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_he_emb.npy')[:,-1,:self._data_stat.original_num_hyperedges])
        he_emb =  torch.flatten(he_emb, start_dim = 1).to(self._device)
        model_name = self.config_str+'_'+str(iterations)+'.chkpt'
        self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/'+model_name, map_location=self._device)['model_link'])
        classifier.fc1.weight = self._model.fc1.weight
        classifier.fc1.bias = self._model.fc1.bias
        ex = shap.DeepExplainer(classifier, he_emb)
        print('{} : begin calculating shap values'.format(self.args.dataset))
        shap_v = torch.Tensor(np.asarray(ex.shap_values(he_emb, check_additivity=False))).reshape(self._data_stat.exact_num_labels, -1, self._data_stat.original_num_hyperedges, self.args.out_dim)
        print('done')      
        labels = torch.LongTensor([int(i) for i in open(self.args.data_dir+'{}/raw/{}_labels.txt'.format(self.args.dataset, self.args.dataset.upper()),'r').read().split('\n')[:-1]])
        mask = -torch.ones(self._data_stat.exact_num_labels,self._data_stat.exact_num_labels)+ 2* torch.eye(self._data_stat.exact_num_labels)
        mask = mask[labels].T
        shap_sum = shap_v.sum(-1) * mask.unsqueeze(-1)
        shap_final = shap_sum.sum(0).sum(0)
        sorted_shap = torch.Tensor(sorted(shap_final.tolist(), reverse= True))
        sorted_index = []
        for i in range(1497):
            found_idx = (shap_final == sorted_shap[i].item()).nonzero(as_tuple=True)[0].tolist()
            sorted_index += found_idx
            i += len(found_idx) - 1 
        sorted_index = torch.LongTensor(sorted_index)
        sorted_index_sorted = torch.LongTensor(sorted(sorted_index[:100].tolist(), reverse=True))

        file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_raw_shap_value.npy'
        np.save(self.args.root_dir+file_name, shap_v.numpy())
        
        file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_shap_calculated.npy'
        np.save(self.args.root_dir+file_name, shap_final.numpy())
          
        
   
def main():
    args = utils.parse_args()
    args.root_dir = ROOT_DIR
    args.code_dir = ROOT_DIR + "/src"
    args.data_dir = ROOT_DIR + "/dataset/"
    mt = shap_calculate(args)
    mt.calculate()

if __name__ == "__main__":
    main()
    