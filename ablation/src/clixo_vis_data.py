import numpy as np
import torch 
import os
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import subprocess
import networkx as nx

cluster_numbers = []


def parse_args():
    parser = argparse.ArgumentParser(description="blablabla")
    parser.add_argument('--data', type=str, default='brca', help="Dataset name. Does not matter whether lower case or letter case") 
    parser.add_argument('--model', type=str, default='disen_hgnn', help="Dataset name. Does not matter whether lower case or letter case") 
    parser.add_argument('--a', type=float, default=0.1, help="Dataset name. Does not matter whether lower case or letter case") 
    parser.add_argument('--b', type=float, default=0.6, help="Dataset name. Does not matter whether lower case or letter case") 
    parser.add_argument('--m', type=float, default=0.005, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--s', type=float, default=0.2, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--dim', type=int, default=32, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--head', type=int, default=4, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--device', type=int, default=6, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--num_path', type=int, default=10, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--disloss', type=float, default=0.0, help="Dataset name. Does not matter whether lower case or letter case")
    parser.add_argument('--lr', type=float, default=0.001, help="Dataset name. Does not matter whether lower case or letter case")
    
    args = parser.parse_args()
    args.device = device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
    args.read_type = "bma" # bma rcmax
    args.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    args.data_dir = args.root_dir + "/dataset/cancer_subtype/"
    args.proc_dir = args.data_dir+"pre_processed/"
    args.src_dir = args.data_dir+"src/"
    args.raw_dir = args.data_dir+"raw/"
    args.fig_dir = args.root_dir + "/ablation/figures/"
    args.save_dir = args.root_dir+"/ablation/save_files/"
    args.temp_dir = args.root_dir+"/ablation/temp_save_files/"
    args.config_str = '{}_{}_0.5_0.25_0.25_10_{}_{}_{}_2_1_1_1_True_True_{}_0'.format(args.data, args.model, args.dim, args.lr, args.head, args.dim)
    if args.model =='hsdn':
        args.config_str = '{}_{}_0.5_0.25_0.25_10_{}_{}_{}_2_{}_1_1_1_True_True_{}_0'.format(args.data, args.model, args.dim, args.lr, args.head, str(args.disloss), str(9*args.head))

    return args

def get_shap(args):
    shap_values = np.load(args.save_dir+args.config_str+'_shap_calculated.npy')
    sorted_shap = torch.Tensor(sorted(shap_values.tolist(), reverse= True))
    
    sorted_index = []
    for i in range(1497):
        found_idx = (torch.Tensor(shap_values) == sorted_shap[i].item()).nonzero(as_tuple=True)[0].tolist()
        sorted_index += found_idx
        i += len(found_idx) - 1 
    sorted_index = torch.LongTensor(sorted_index)
    return sorted_index, sorted_shap
    
def get_path_list(args):
    pathway_file = "{}index_pathway_map.txt".format(args.proc_dir)
    pathway_open = open(pathway_file, 'r')
    path_list = [i for i in pathway_open.read().split('\n')]
    path_list = ['_'.join(i.split('_')[1:]) for i in path_list]
    pathway_open.close()
    return path_list

def create_partial_go_sem(args, shap_i):
    sem_dist = open(args.proc_dir+'go_sem_dist_bp_lin_bma.csv').read().split('\n')[1:-1]
    dist_mat = [i.split(',')[:args.num_hyperedge] for i in sem_dist[:args.num_hyperedge]]
    dist_mat
    for i in range(args.num_hyperedge):
        for j in range(i+1,args.num_hyperedge):
            dist_mat[j][i] = dist_mat[i][j]
    dist_np = np.asarray([[float(elem) for elem in row] for row in dist_mat])
    partial_dist_np = dist_np[shap_i[:args.num_path],:]
    partial_dist_np = partial_dist_np[:,shap_i[:args.num_path]]
    file_name = '{}_{}_{}_{}_selected_{}.csv'.format(args.data, args.model, args.dim, args.head, args.num_path)
    np.savetxt(args.temp_dir+file_name, partial_dist_np, delimiter=",")
    return partial_dist_np, dist_np

def convert_2_tsv(args, dist_np, path_list, shap_i):
    text = ""
    for i in range(args.num_path):
        for j in range(i+1, args.num_path):
            text += path_list[shap_i[i].item()] + '\t'
            text += path_list[shap_i[j].item()] + '\t'
            text += str(dist_np[i,j]) + '\n'
    file_name = args.temp_dir + '{}_{}_{}_{}_selected_{}.tsv'.format(args.data, args.model, args.dim, args.head, args.num_path)
    f = open(file_name, 'w')
    f.write(text)
    f.close()
    
    return file_name

def execute_clixo(args, tsv_file_name):
    new_file = args.temp_dir + 'lin_bma_{}_{}_{}_{}_{}_{}_selected_{}.txt'.format(args.a, args.b, args.m, args.s, args.dim, args.head, args.num_path)
    ont_file_name = args.temp_dir + 'lin_bma_{}_{}_{}_{}_{}_{}_selected_{}_ont.txt'.format(args.a, args.b, args.m, args.s, args.dim, args.head, args.num_path)
    cluster_gene_file_name = args.temp_dir + 'lin_bma_{}_{}_{}_{}_{}_{}_selected_{}_cluster_gene.txt'.format(args.a, args.b, args.m, args.s, args.dim, args.head, args.num_path)
    try: 
        subprocess.check_call(['sh',args.root_dir+'/ablation/src/exe_clixo.sh', tsv_file_name, str(args.a), str(args.b), str(args.m), str(args.s), new_file, ont_file_name, cluster_gene_file_name])
    except Exception:
        print('finished')
    return [new_file, ont_file_name, cluster_gene_file_name]


def anonymize_cluster_path(args, file_name, path_list): # instead of pathway names (too long), assign pathway index
    cluster_gene_file = open(file_name,'r')
    cluster_gene_map = [i.split('\t') for i in cluster_gene_file.read().split('\n')[:-1]]
    cluster_gene_file.close()
    min_cluster_idx = int(cluster_gene_map[0][0])
    node_list = []
    edge_list = []
    for i in range(len(cluster_gene_map)):
        clust_name = 'c_'+str(int(cluster_gene_map[i][0]) - min_cluster_idx)
        if clust_name not in node_list:
            node_list.append(clust_name)
        if cluster_gene_map[i][2] == 'gene':
            idx = path_list.index(cluster_gene_map[i][1])
            name = 'p_'+str(idx)
        elif cluster_gene_map[i][2] == 'default':
            name = 'c_'+str(int(cluster_gene_map[i][1]) - min_cluster_idx)
        if name not in node_list: 
            node_list.append(name)
        edge_list.append([clust_name, name])
    return node_list, edge_list
    

def visualize_cluster(args, node_list, edge_list, path_list): # Result that is similar to Figure 16. 
    G = nx.DiGraph(edge_list)
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax, node_size = 800, font_size=8)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.savefig('{}{}/{}/{}_{}_hierarchy_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf() 
    return

def visualize_shap_dist(args, shap_v):
    plt.plot([i for i in range(1,1498)], shap_v, label = "shap") 
    plt.legend() 
    plt.xlabel('ranking')
    plt.ylabel('SHAP value')
    plt.savefig(args.fig_dir+'shap_all.svg')
    #plt.show()
    plt.clf()
    #plt.plot([i for i in range(1,101)], shap_v[:100], label = "shap") 
    #plt.legend() 
    #plt.show()
    #plt.clf()
    plt.plot([i for i in range(1,31)], shap_v[:30], label = "shap") 
    plt.legend() 
    plt.xlabel('ranking')
    plt.ylabel('SHAP value')
    plt.savefig(args.fig_dir+'shap_30.svg')
    plt.legend() 
    #plt.show()
    plt.clf()
    return


def visualize_path_functionality(args, pathlist, shap_i, dist_np):
    num_hyperedge = len(pathlist)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    selected_index = shap_i[:args.num_path]
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    
    # first layer
    print('first layer')
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    for p_1 in tqdm(range(num_hyperedge)):
        dist_or_sim[p_1] = torch.cdist(emb[p_1].unsqueeze(-2), emb.permute(1,0,2), p = 2).squeeze().permute(1,0).mean(-1)
        
    dist_or_sim = 1/(1+dist_or_sim)
    our_sim = dist_or_sim[shap_i[:args.num_path],:]
    our_sim = our_sim[:,shap_i[:args.num_path]]
    plt.figure(figsize=(8,6))
    g=sns.heatmap(our_sim, cmap='YlGnBu')
    g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
    g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel('pathway')
    plt.ylabel('pathway')
    plt.savefig('{}{}/{}/{}_{}_path_firstlayer_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf()
    plt.close()
    
    if args.model != 'hsdn':
        print('second layer')
        emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1].to(args.device) #[0] or [1]
        for p_1 in tqdm(range(num_hyperedge)):
            dist_or_sim[p_1] =  torch.cdist(emb[p_1], emb, p = 2).view(num_hyperedge,-1).mean(-1)
        dist_or_sim = 1/(1+dist_or_sim)
        our_sim = dist_or_sim[shap_i[:args.num_path],:]
        our_sim = our_sim[:,shap_i[:args.num_path]]
        plt.figure(figsize=(8,6))
        g=sns.heatmap(our_sim, cmap='YlGnBu')
        g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        plt.xlabel('pathway')
        plt.ylabel('pathway')
        plt.savefig('{}{}/{}/{}_{}_path_secondlayer_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
        plt.clf()
        plt.close()
    
        
        print('first+second layer')
        emb = torch.cat([att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0], att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1]], -1).to(args.device) #[0] or [1]
        for p_1 in tqdm(range(num_hyperedge)):
            dist_or_sim[p_1] =  torch.cdist(emb[p_1], emb, p = 2).view(num_hyperedge,-1).mean(-1)
        dist_or_sim = 1/(1+dist_or_sim)
        our_sim = dist_or_sim[shap_i[:args.num_path],:]
        our_sim = our_sim[:,shap_i[:args.num_path]]
        plt.figure(figsize=(8,6))
        g=sns.heatmap(our_sim, cmap='YlGnBu')
        g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        plt.xlabel('pathway')
        plt.ylabel('pathway')
        plt.savefig('{}{}/{}/{}_{}_path_bothlayer_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
        plt.clf()
        plt.close()
    
    
    print('ground truth')
    plt.figure(figsize=(8,6))
    g=sns.heatmap(dist_np, cmap='YlGnBu')
    g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
    g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel('pathway')
    plt.ylabel('pathway')
    plt.savefig('{}{}/{}/{}_{}_path_groundtruth_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf()
    plt.close()
    return




def visualize_cluster_functionality(args, file_name, path_list, dist_np, shap_i):
    num_hyperedge = len(path_list)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    
    cluster_gene_file = open(file_name, 'r')
    cluster_gene_map = [i.split('\t') for i in cluster_gene_file.read().split('\n')[:-1]]
    cluster_gene_file.close()
    num_cluster = len(cluster_gene_map)
    for i in range(num_cluster):
        genes = cluster_gene_map[i][2].split(',')[:-1]
        cluster_gene_map[i] = [path_list.index(j) for j in genes]    
    
    
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    for p_1 in tqdm(range(num_hyperedge)):
        dist_or_sim[p_1] =  torch.cdist(emb[p_1], emb, p = 2).view(num_hyperedge,-1).mean(-1)
    dist_or_sim = 1/(1+dist_or_sim)
    
    our_sim = torch.zeros(num_cluster, num_cluster)
    for i in range(num_cluster):
        for j in range(num_cluster):
            temp = dist_or_sim[cluster_gene_map[i],:]
            our_sim[i,j] = temp[:, cluster_gene_map[j]].mean()
            
    plt.figure(figsize=(8,6))
    g=sns.heatmap(our_sim, cmap='YlGnBu')
    g.set_xticklabels(np.arange(num_cluster), fontsize = 16)
    g.set_yticklabels(np.arange(num_cluster), fontsize = 16)
    plt.xlabel('cluster', labelpad=10, fontsize = 16)
    plt.ylabel('cluster', labelpad=10, fontsize = 16)
    plt.savefig('{}{}/{}/{}_{}_cluster_firstlayer_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf()
    plt.close()
    truth_sim = torch.zeros(num_cluster, num_cluster)
    for i in range(num_cluster):
        for j in range(num_cluster):
            temp = dist_np[cluster_gene_map[i], :]
            truth_sim[i,j] = temp[:,cluster_gene_map[j]].mean()
    
    plt.figure(figsize=(8,6))
    g=sns.heatmap(truth_sim, cmap='YlGnBu')
    g.set_xticklabels(np.arange(num_cluster), fontsize = 16)
    g.set_yticklabels(np.arange(num_cluster), fontsize = 16)
    plt.xlabel('cluster', labelpad=10, fontsize = 16)
    plt.ylabel('cluster', labelpad=10, fontsize = 16)
    plt.savefig('{}{}/{}/{}_{}_cluster_groundtruth_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf()
    plt.close()
    return







def visualize_path_per_head_functionality(args, pathlist, shap_i, dist_np):
    num_hyperedge = len(pathlist)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    selected_index = shap_i[:args.num_path]
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    
    # first layer
    print('first layer')
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    for iter_head in range(args.head):
        for p_1 in tqdm(range(num_hyperedge)):
            dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
        dist_or_sim = 1/(1+dist_or_sim)
        our_sim = dist_or_sim[shap_i[:args.num_path],:]
        our_sim = our_sim[:,shap_i[:args.num_path]]
        plt.figure(figsize=(8,6))
        g=sns.heatmap(our_sim, cmap='YlGnBu')
        g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
        plt.xlabel('pathway')
        plt.ylabel('pathway')
        plt.savefig('{}{}/{}/per_head/{}_{}_path_firstlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
        plt.clf()
        plt.close()
    
    if args.model != 'hsdn':
        print('second layer')
        emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1].to(args.device) #[0] or [1]
        for iter_head in range(args.head):
            for p_1 in tqdm(range(num_hyperedge)):
                dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
            dist_or_sim = 1/(1+dist_or_sim)
            our_sim = dist_or_sim[shap_i[:args.num_path],:]
            our_sim = our_sim[:,shap_i[:args.num_path]]
            plt.figure(figsize=(8,6))
            g=sns.heatmap(our_sim, cmap='YlGnBu')
            g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
            g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
            plt.xlabel('pathway')
            plt.ylabel('pathway')
            plt.savefig('{}{}/{}/per_head/{}_{}_path_secondlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
            plt.clf()
            plt.close()
    
        print('first+second layer')
        emb = torch.cat([att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0], att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1]], -1).to(args.device) #[0] or [1] 
        for iter_head in range(args.head):
            for p_1 in tqdm(range(num_hyperedge)):
                dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
            dist_or_sim = 1/(1+dist_or_sim)
            our_sim = dist_or_sim[shap_i[:args.num_path],:]
            our_sim = our_sim[:,shap_i[:args.num_path]]
            plt.figure(figsize=(8,6))
            g=sns.heatmap(our_sim, cmap='YlGnBu')
            g.set_xticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
            g.set_yticklabels([i.item() for i in selected_index], rotation=45, ha="right", rotation_mode="anchor")
            plt.xlabel('pathway')
            plt.ylabel('pathway')
            plt.savefig('{}{}/{}/per_head/{}_{}_path_bothlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
            plt.clf()
            plt.close()
    
    return



def visualize_cluster_per_head_functionality(args, file_name, path_list, dist_np, shap_i):
    num_hyperedge = len(path_list)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    
    cluster_gene_file = open(file_name, 'r')
    cluster_gene_map = [i.split('\t') for i in cluster_gene_file.read().split('\n')[:-1]]
    cluster_gene_file.close()
    num_cluster = len(cluster_gene_map)
    for i in range(num_cluster):
        genes = cluster_gene_map[i][2].split(',')[:-1]
        cluster_gene_map[i] = [path_list.index(j) for j in genes]    
    
    
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    for iter_head in range(args.head):
        for p_1 in tqdm(range(num_hyperedge)):
            dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
        dist_or_sim = 1/(1+dist_or_sim)
        
        our_sim = torch.zeros(num_cluster, num_cluster)
        for i in range(num_cluster):
            for j in range(num_cluster):
                temp = dist_or_sim[cluster_gene_map[i],:]
                our_sim[i,j] = temp[:, cluster_gene_map[j]].mean()
                
        plt.figure(figsize=(8,6))
        g=sns.heatmap(our_sim, cmap='YlGnBu')
        g.set_xticklabels(np.arange(num_cluster))
        g.set_yticklabels(np.arange(num_cluster))
        plt.xlabel('cluster', labelpad=15)
        plt.ylabel('cluster', labelpad=15)
        plt.savefig('{}{}/{}/per_head/{}_{}_cluster_firstlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
        plt.clf()
        plt.close()
    
    if args.model != 'hsdn':
        emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1].to(args.device) #[0] or [1]
        for iter_head in range(args.head):
            for p_1 in tqdm(range(num_hyperedge)):
                dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
            dist_or_sim = 1/(1+dist_or_sim)
            
            our_sim = torch.zeros(num_cluster, num_cluster)
            for i in range(num_cluster):
                for j in range(num_cluster):
                    temp = dist_or_sim[cluster_gene_map[i],:]
                    our_sim[i,j] = temp[:, cluster_gene_map[j]].mean()
                    
            plt.figure(figsize=(8,6))
            g=sns.heatmap(our_sim, cmap='YlGnBu')
            g.set_xticklabels(np.arange(num_cluster))
            g.set_yticklabels(np.arange(num_cluster))
            plt.xlabel('cluster', labelpad=15)
            plt.ylabel('cluster', labelpad=15)
            plt.savefig('{}{}/{}/per_head/{}_{}_cluster_secondlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
            plt.clf()
            plt.close()
        
        
        emb = torch.cat([att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0], att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][1]], -1).to(args.device)
        for iter_head in range(args.head):
            for p_1 in tqdm(range(num_hyperedge)):
                dist_or_sim[p_1] = torch.abs(emb[:,:,iter_head] - emb[p_1,:,iter_head]).mean(-1)
            dist_or_sim = 1/(1+dist_or_sim)
            
            our_sim = torch.zeros(num_cluster, num_cluster)
            for i in range(num_cluster):
                for j in range(num_cluster):
                    temp = dist_or_sim[cluster_gene_map[i],:]
                    our_sim[i,j] = temp[:, cluster_gene_map[j]].mean()
                    
            plt.figure(figsize=(8,6))
            g=sns.heatmap(our_sim, cmap='YlGnBu')
            g.set_xticklabels(np.arange(num_cluster))
            g.set_yticklabels(np.arange(num_cluster))
            plt.xlabel('cluster', labelpad=15)
            plt.ylabel('cluster', labelpad=15)
            plt.savefig('{}{}/{}/per_head/{}_{}_cluster_bothlayer_head_{}_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, str(iter_head), args.dim, args.head, args.disloss, args.lr)) 
            plt.clf()
            plt.close()
    
    return


def visualize_path_activation_per_head(args, file_name, path_list, dist_np, shap_i):
    num_hyperedge = len(path_list)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    selected_index = shap_i[:args.num_path]   
    
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    path_activation = torch.zeros(selected_index.size(0), emb.size(-1))
    for iter_path in range(selected_index.size(0)):
        path_activation[iter_path,:] =  emb[selected_index[iter_path],:,:].mean(-2)
    #pdb.set_trace()
    
    plt.figure(figsize=(8,6))
    g=sns.heatmap(path_activation.T, cmap='YlGnBu', annot=True)
    g.set_xticklabels(np.arange(path_activation.size(0)))
    g.set_yticklabels(np.arange(emb.size(-1)))
    plt.xlabel('pathway', labelpad=15)
    plt.ylabel('factor relevance', labelpad=15)
    plt.show()
    plt.clf()
    plt.close()
    
    
def visualize_hyperedge_activation_per_head(args, file_name, path_list, dist_np, shap_i):
    num_hyperedge = len(path_list)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    selected_index = shap_i[:args.num_path]   
    
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    path_activation = torch.zeros(selected_index.size(0), emb.size(-1))
    for iter_path in range(selected_index.size(0)):
        path_activation[iter_path,:] =  emb[selected_index[iter_path],:,:].mean(-2)
    pdb.set_trace()
    
    plt.figure(figsize=(8,6))
    g=sns.heatmap(path_activation.T, cmap='YlGnBu', annot=True)
    g.set_xticklabels(np.arange(len(path_activation)))
    g.set_yticklabels(np.arange(emb.size(-1)))
    plt.xlabel('pathway', labelpad=15)
    plt.ylabel('factor relevance', labelpad=15)
    plt.show()
    plt.clf()
    plt.close()


def visualize_cluster_activation_per_head(args, file_name, path_list, dist_np, shap_i):
    num_hyperedge = len(path_list)
    dist_or_sim = torch.zeros(num_hyperedge, num_hyperedge)
    att_score = torch.from_numpy(np.load(args.save_dir+args.config_str+'_att_score.npy'))
    cluster_gene_file = open(file_name, 'r')
    cluster_gene_map = [i.split('\t') for i in cluster_gene_file.read().split('\n')[:-1]]
    cluster_gene_file.close()
    num_cluster = len(cluster_gene_map)
    for i in range(num_cluster):
        genes = cluster_gene_map[i][2].split(',')[:-1]
        cluster_gene_map[i] = [path_list.index(j) for j in genes] 
    
    emb = att_score.permute(1,2,0,3)[:,:num_hyperedge,:,:][0].to(args.device) #[0] or [1]
    cluster_activation = torch.zeros(len(cluster_gene_map), emb.size(-1))
    for iter_cluster in range(cluster_activation.size(0)):
        cluster_activation[iter_cluster,:] =  emb[cluster_gene_map[iter_cluster],:,:].mean(-2).mean(-2)
    
    plt.figure(figsize=(8,6))
    g=sns.heatmap(cluster_activation.T, cmap='YlGnBu')
    g.set_xticklabels(np.arange(len(cluster_activation)), fontsize = 16)
    g.set_yticklabels(np.arange(emb.size(-1)), fontsize = 16)
    plt.xlabel('cluster', labelpad=10, fontsize=16 )
    plt.ylabel('factor', labelpad=10, fontsize=16)
    plt.savefig('{}{}/{}/per_head/{}_{}_cluster_firstlayer_activation_{}_{}_{}_{}.svg'.format(args.fig_dir, 'visualization', args.model, args.data, args.num_path, args.dim, args.head, args.disloss, args.lr)) 
    plt.clf()
    plt.close()
    return




def main():
    args = parse_args()
    shap_i, shap_v= get_shap(args) #i: index v : value
    path_list = get_path_list(args)
    args.num_hyperedge = len(path_list)
    dist_np, dist_np_all = create_partial_go_sem(args, shap_i)
    tsv_file_name = convert_2_tsv(args, dist_np, path_list, shap_i)
    file_names = execute_clixo(args,tsv_file_name) # execute clixo algorithm
    node_list, edge_list = anonymize_cluster_path(args, file_names[-2], path_list) # assign index to pathways and clusters. 
    visualize_cluster(args, node_list, edge_list, path_list) # Result that is similar to Figure 16. 
    visualize_shap_dist(args, shap_v) # Figure 15. Visualize SHAP distribution
    
    
    visualize_path_functionality(args,path_list, shap_i, dist_np) # Not in the paper. Similar to Figure 5, but in a pathway level, rather than cluster level.
    visualize_cluster_functionality(args, file_names[-1], path_list, dist_np_all, shap_i) # Figure 5
    
    visualize_path_per_head_functionality(args,path_list, shap_i, dist_np) # Not in the paper. visualization per head
    visualize_cluster_per_head_functionality(args, file_names[-1], path_list, dist_np_all, shap_i) # Not in the paper. visualization per head
    
    visualize_cluster_activation_per_head(args, file_names[-1], path_list, dist_np, shap_i) # Not in the paper. Visualizes (average) relevance score distribution of each factor across clusters. (Heatmap of size <number of factors> X <number of clusters>)
    visualize_path_activation_per_head(args, file_names[-1], path_list, dist_np, shap_i)# Not in the paper. Visualizes (average) relevance score distribution of each factor across pathways. (Heatmap of size <number of factors> X <number of pathways>)
    return 


if __name__ == "__main__":
    main()
    

