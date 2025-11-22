import torch
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

def config_str(data, model, dim, head):
    if model == 'disen_hgnn':
        if data == 'brca':
            file_str =  'brca_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'stad':
            file_str =  'stad_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'sarc':
            file_str =  'sarc_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'lgg':
            file_str =  'lgg_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'hnsc':
            file_str =   'hnsc_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'cesc':
            file_str =  'cesc_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'kipan':
            file_str =  'kipan_disen_hgnn_0.5_0.25_0.25_10_{}_0.0001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        if data == 'nsclc':
            file_str =   'nsclc_disen_hgnn_0.5_0.25_0.25_10_{}_0.001_{}_2_1_1_1_True_True_{}_0'.format(dim, head, dim)
        return file_str
    elif model == 'hsdn':
        if data == 'brca':
            file_str =  'brca_hsdn_0.5_0.25_0.25_10_{}_0.001_{}_2_0.01_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'stad':
            file_str =  'stad_hsdn_0.5_0.25_0.25_10_{}_0.001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'sarc':
            file_str =  'sarc_hsdn_0.5_0.25_0.25_10_{}_0.001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'lgg':
            file_str =  'lgg_hsdn_0.5_0.25_0.25_10_{}_0.001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'hnsc':
            file_str =  'hnsc_hsdn_0.5_0.25_0.25_10_{}_0.001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'cesc':
            file_str =  'cesc_hsdn_0.5_0.25_0.25_10_{}_0.0001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'kipan':
            file_str =  'kipan_hsdn_0.5_0.25_0.25_10_{}_0.0001_{}_2_0.0001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
        if data == 'nsclc':
            file_str =  'nsclc_hsdn_0.5_0.25_0.25_10_{}_0.0001_{}_2_0.001_1_1_1_True_True_{}_0'.format(dim, head, 9*head)
            
        return file_str
    exit()
    
sorted_collection = torch.Tensor(2,8,9,1497) # mode, data, combination, pathway
save_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/ablation/save_files/"
fig_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/ablation/figures/jaccard/"
for d, data in enumerate(['brca', 'stad', 'sarc', 'lgg', 'hnsc', 'cesc', 'kipan', 'nsclc']):
    if data in ['lgg','stad']:
        continue
    for i, dim in enumerate([16,32,64]):
        for j, head in enumerate([2,4,8]):
            
            
            shap_values = np.load(save_dir+config_str(data, 'disen_hgnn', dim, head)+'_shap_calculated.npy')
            sorted_shap = torch.Tensor(sorted(shap_values.tolist(), reverse= True))
        
            sorted_index = []
            for path in range(1497):
                found_idx = (torch.Tensor(shap_values) == sorted_shap[path].item()).nonzero(as_tuple=True)[0].tolist()
                sorted_index += found_idx
            sorted_collection[0,d, 3*i+j,:] = torch.LongTensor(sorted_index)
for d, data in enumerate(['brca', 'stad', 'sarc', 'lgg', 'hnsc', 'cesc', 'kipan', 'nsclc']):
    if data in ['lgg','stad']:
        continue
    for i, dim in enumerate([16,32,64]):
        for j, head in enumerate([2,4,8]):
            shap_values = np.load(save_dir+config_str(data, 'hsdn', dim, head)+'_shap_calculated.npy')
            sorted_shap = torch.Tensor(sorted(shap_values.tolist(), reverse= True))
        
            sorted_index = []
            for path in range(1497):
                found_idx = (torch.Tensor(shap_values) == sorted_shap[path].item()).nonzero(as_tuple=True)[0].tolist()
                sorted_index += found_idx
                found_length = len(found_idx)
            sorted_collection[1,d, 3*i+j,:] = torch.LongTensor(sorted_index)
            

num_path = [10,15,20,30,50,100,150,200,300,500]
jaccard_torch = torch.Tensor(2, 8, 10,9,9 )
for m in range(2):
    for d in range(8):
        if d in [2,3]:
            continue
        for npa in range(10):
            for i in range(9):
                set_1 = set(sorted_collection[m,d,i,:num_path[npa]].tolist())
                for j in range(9):
                    set_2 = set(sorted_collection[m,d,j,:num_path[npa]].tolist())
                    jaccard_torch[m,d,npa, i, j] = float(len(set_1.intersection(set_2)))/float(len(set_1.union(set_2)))
                    
for m ,model in enumerate(['disen_hgnn', 'hsdn']):
    for d, data in enumerate(['brca', 'stad', 'sarc', 'lgg', 'hnsc', 'cesc', 'kipan', 'nsclc']):
        if d in [1,3]:
            continue
        for npa in range(10):
            
            plt.figure(figsize=(8,6))
            g=sns.heatmap(jaccard_torch[m,d,npa], cmap='YlGnBu', vmin=0, vmax=1)
            g.set_xticklabels(['16,2', '16,4', '16,8','32,2', '32,4', '32,8','64,2', '64,4', '64,8'], fontsize = 16, rotation=45, ha="right", rotation_mode="anchor")
            g.set_yticklabels(['16,2', '16,4', '16,8','32,2', '32,4', '32,8','64,2', '64,4', '64,8'], fontsize = 16, rotation=45, ha="right", rotation_mode="anchor")
            plt.xlabel('hyperparameter')
            plt.ylabel('hyperparameter')
            plt.title('{}, {}, top_{}, avg : {:.3f}'.format(model, data, num_path[npa],jaccard_torch[m,d,npa].mean() ), fontsize = 16)
            plt.savefig('{}{}_{}_top_{}.svg'.format(fig_dir, model, data, str(num_path[npa]))) 
            plt.clf()
            plt.close()


        