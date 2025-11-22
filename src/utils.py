import numpy as np
import torch
import random
import os
import os.path as osp
import sys
import argparse
import trainer

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','he_emb', 'att_score', 'use_wandb', 'project_name','run_name','debug','silence','show_bar','root_dir','code_dir','data_dir','wd','epoch','num_layers','feature_noise','dropout','he_activation','normalization','Classifier_hidden','Classifier_num_layers', 'deepset_input_norm','hnhn_alpha', 'hnhn_beta', 'aggregate','task', 'metric','use_same_implicit','consistency_loss_ratio','max_sim_calculation','val_criterion','use_default_split', 'per_class_split', 'interpol_ratio', 'batch_size','use_balanced_split','hcl_spec','disen_spec','','lam0','lam1', 'check_data']:
            st_ = "{}_".format(val)
            st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    assert len(args_names) == len(args_vals)
    if args.silence:
        return 
    if args.silence:
        temp_list = []
        for i in range(len(args_names)):
            temp_list.append((str(args_names[i]),str(args_vals[i])))
        print()
        print()
        print("------------------------configs------------------------")
        print(temp_list)
        print("-------------------------------------------------------")          
    else:
        print("------------------------configs------------------------")
        for i in range(len(args_names)):
            print(str(args_names[i]) + " : " + str(args_vals[i]))
        print("-------------------------------------------------------")    
    
    
def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="blablabla")
    parser.add_argument('--dataset', type=str, default='coauthor_cora', help="Dataset name. Does not matter whether lower case or letter case") 
    parser.add_argument('--model', type=str, default='hgnn')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers. Default is 2.')
    parser.add_argument('--train_percent', type = float, default = 60, help = "train set ratio(ex: 60)")
    parser.add_argument('--valid_percent', type = float, default = 20, help = "valid set ratio(ex: 20)")
    parser.add_argument('--test_percent', type = float, default = 20, help = "test set ratio(ex: 20)")
    parser.add_argument('--feature_noise', type = float, default = 0.0, help = "feature noise standard deviation for synthetic data")
    parser.add_argument('--num_repeat', type=int, default=10, help = "Number of repretition")
    parser.add_argument('--hidden', type = int, default = 64, help = 'Dimension of hidden layer. Default is 64, Must be even number')
    parser.add_argument('--lr', type=float, default=0.001, help = "Learning rate. Default is 0.005")
    parser.add_argument('--wd', type=float, default=0.0, help = "Weight decay. Default is 0.001")
    parser.add_argument('--device', type=int, default='4', help = "The GPU number to be used. Default is 4")
    parser.add_argument('--epoch', type=int, default=500, help = "Number of epochs for training. Default is 1000")
    parser.add_argument('--dropout', type = float, default= 0.5,help="dropout rate")
    parser.add_argument('--he_activation', type = str, default= "relu",help="nonlinear activation function for hyperedge")
    parser.add_argument('--heads', type = int, default = 8, help="number of heads in PMA")
    parser.add_argument('--normalization', type = str, default = 'ln', help="normalization type : ln, bn ,None")
    parser.add_argument('--Classifier_hidden', type = int, default = 4, help="hidden layer dimension of classifier(for AllDeepSet/AllSetTransformer)")
    parser.add_argument('--Classifier_num_layers', type = int, default = 1, help="number of layers of MLP classifier(usually for AllDeepSet and AllSetTransformer)")
    parser.add_argument('--MLP_num_layers', type=int, default=2, help='Number of layers for MLP in AllDeepSet/AllSetTransformer. Default is 2.')
    parser.add_argument('--PMA', action='store_true', help='AllDeepSet(False), AllSetTransformer(True)')
    parser.add_argument('--deepset_input_norm', action='store_true', help='input normalization for deepsets')
    parser.add_argument('--hnhn_alpha',type = float, default = -1.5, help='alpha value for HNHN model' )
    parser.add_argument('--hnhn_beta',type = float, default = -0.5, help='beta value for HNHN model' )
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean','add'], help='aggregation option. One of add, sum and mean')
    parser.add_argument("--silence", action = "store_true", help = "If you set this as true, the code will show only minimum necessary info")
    parser.add_argument("--task", default = 'basic', choices = ['basic', 'bio'], help = 'type of task : basic (benchmark dataset), bio (cancer subtype classification)')
    parser.add_argument("--metric", type = str, default = 'acc', help = 'type of performance metric : acc, f1')
    parser.add_argument("--per_class_split", action = 'store_true', help = 'when splitting dataset, spilt ratio must equal for all classes')
    parser.add_argument("--use_default_split", action = 'store_true', help = 'Use split provided by the dataset')
    parser.add_argument("--val_criterion", default = 'loss', choices = ['loss', 'acc', 'micro_f1', 'macro_f1'], help = 'criterion to save best model : loss, acc, micro_f1, macro_f1')
    parser.add_argument('--disen_spec', type = int, default = 0, help = 'model version. ex: ED-HNN (disen_spec=0) vs ED-HNN2 (disen_spec=1).')
    parser.add_argument('--disen_loss_ratio', type = float, default = 0.0, help = 'disentangle loss term ratio')
    parser.add_argument('--interpol_ratio', type = float, default = 0.5, help = 'interpolation ratio between residual and current layer')
    parser.add_argument('--edhnn_mlp_layer_1', type = int, default = 1, help = 'number of mlp 1')
    parser.add_argument('--edhnn_mlp_layer_2', type = int, default = 1, help = 'number of mlp 2')
    parser.add_argument('--edhnn_mlp_layer_3', type = int, default = 1, help = 'number of mlp 3')
    parser.add_argument('--hcl_spec', type = int, default = 0, help = 'Used only for cancer subtype (hypergraph) classification task. Set this to 3.')
    parser.add_argument("--show_bar", action = "store_true", help = "show tqdm bar")
    parser.add_argument("--use_balanced_split", action = "store_true", help = "when splitting dataset, use class ratio balanced split")
    parser.add_argument("--batch_size", type=int, default=0, help = "batch size, default : 0(not using batch). Set to 50 for cancer subtype classification")
    parser.add_argument("--att_score", action='store_true', help='Get attention scores of (already) trained model.')
    parser.add_argument("--he_emb", action='store_true', help='Get hyperedge embeddings of (already) trained model.')
    parser.add_argument("--debug", action='store_true', help='debug mode')
    parser.add_argument("--use_wandb", action='store_true', help='use wandb or not')
    parser.add_argument("--project_name", type=str, default = 'hypergraph disentangle',help='wandb project name')
    parser.add_argument("--run_name", type =str, default = 'None', help='wandb run name. Use this if you want to add description @ wandb')
    parser.add_argument("--check_data", action = "store_true", help = "show tqdm bar")
    parser.add_argument("--timer", action = "store_true", help = "Measure time took for training. Not used in usual training.")
    args = parser.parse_args()

    # prepreocess argument
    #args.activation = args.activation.lower()
    args.model = args.model.lower()
    args.dataset = args.dataset.lower()
    if args.run_name == 'None':
        args.run_name = '{}_disen_{}_hcl_{}_{}_{}_lr_{}'.format(args.model, args.disen_spec, args.hcl_spec, args.hidden, args.heads, args.lr)
    # check validity of argument
    
    existing_dataset = ['20newsw100', 'Mushroom', 'zoo', 
                            'NTU2012', 'ModelNet40', 
                            'coauthor_cora', 'coauthor_dblp',                           # load_citation_dataset
                            'amazon_reviews', 'walmart_trips', 'house_committees',      # load_cornell_dataset  --> add feature noise
                            'walmart_trips_100', 'house_committees_100', 'senate_committees_100', 'congress_bills_100',                # load_cornell_dataset  --> add feature noise
                            'cora', 'citeseer', 'pubmed',                               # load_citation_dataset'
                            'BRCA', 'STAD', 'SARC', 'LGG', "KIPAN", "NSCLC", 'CESC', 'HNSC'
                            ]
    existing_dataset = list(map(str.lower, existing_dataset))
    assert args.dataset in existing_dataset, "argument validity check : invalid dataset name"
    assert args.hidden % 2 == 0, "argument error : Hidden layer dimension must be even number"
    assert args.num_repeat > 0, "argument error : number_repeat must be positive value"
    if args.val_criterion in ['micro_f1', 'macro_f1']:
        assert args.metric == 'f1'
    elif args.val_criterion == 'acc':
        assert args.metric == 'acc'

    args.train_percent /= 100
    args.valid_percent /= 100
    args.test_percent /= 100
    
    return args

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
