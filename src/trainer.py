# functions related to training / validation / testing
from torch.utils.tensorboard import SummaryWriter
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
import wandb
from time import time

    
    
class Model_Trainer:
    def __init__(self, args):
        utils.printConfig(args)
        self.args = args
        self._init()
        self.config_str = utils.config2string(args)
        if not args.silence:
            print("\n[Config] {}\n".format(self.config_str))
        self.LOG_DIR = self.args.root_dir + '/tensorboard/'+ str(args.model)+'/'+str(args.dataset)
        self._train_writer = SummaryWriter(os.path.join(self.LOG_DIR, "train"))
        self._val_writer = SummaryWriter(os.path.join(self.LOG_DIR, "val"))
        self._val_score_writer = SummaryWriter(os.path.join(self.LOG_DIR, "val_score"))
        self._train_add_loss_writer = SummaryWriter(os.path.join(self.LOG_DIR, "train_add_loss"))
        self._val_add_loss_writer = SummaryWriter(os.path.join(self.LOG_DIR, "val_add_loss"))
        self._metric = self.args.metric
        if args.use_wandb:
            wandb.init(project=args.project_name)
            wandb.config.update(args)
            wandb.run.name = args.run_name
            wandb.run.save()
        
        
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
        
        if self.args.debug:
            pdb.set_trace()
        
    def train(self):
        if self.args.task == 'bio': # cancer subtype classification task
            self.cancer_batch_train()
            return
        if self.args.he_emb: # get hyperedge embeddings of (already) trained model.
            return self.get_hyperedge_emb()
        
        # node classification task.        
        best_test_score_list = []
        self._dataset = self._dataset.to(self._device)
        
        for iterations in range(self.args.num_repeat):
            self._optimizer, self._scheduler = self.model_manager_result.set_optimizer()
            self.min_val_loss = float('inf')
            self.max_val_score = -1
            if self.args.silence:
                epoch_range = [i for i in range(self.args.epoch)]
            else:
                epoch_range = tqdm([i for i in range(self.args.epoch)])
            utils.set_seed(seed = self._random_seed_numbers_list[iterations])
            self._model.reset_parameters()                
            
            for epoch in epoch_range:
                if not self.args.silence:
                    epoch_range.set_description("training...")
                self._model.train()
                self._optimizer.zero_grad()
                output, additional_loss = self._model(self._dataset.x, self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                ##################################
                if additional_loss != None : 
                    self._train_add_loss_writer.add_scalar("Additional_Loss_{iter}".format(iter=iterations), additional_loss, epoch)
                    self._train_add_loss_writer.flush()
                ##################################
                loss = self.loss_cal(output, self._dataset.y, self._dataset.train_mask, iterations, additional_loss)
                self._train_writer.add_scalar("Loss_{iter}".format(iter=iterations), loss, epoch)
                self._train_writer.flush()
                loss.backward()
                self._optimizer.step()     
                
                self._model.eval()
                with torch.no_grad():
                    output, additional_loss = self._model(self._dataset.x, self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                    ##################################
                    if additional_loss != None : 
                        self._val_add_loss_writer.add_scalar("Additional_Loss_{iter}".format(iter=iterations), additional_loss, epoch)
                        self._val_add_loss_writer.flush()
                    ##################################
                    loss = self.loss_cal(output, self._dataset.y, self._dataset.val_mask, iterations, additional_loss)
                    
                    self._val_writer.add_scalar("Loss_{iter}".format(iter=iterations), loss, epoch)
                    self._val_writer.flush()
                    score = self.scorer(output, self._dataset.y, self._dataset.val_mask, iterations)
                    if self.args.metric == 'acc':
                        self._val_score_writer.add_scalar("acc_{iter}".format(iter=iterations), score, epoch)
                    elif self.args.metric == 'f1':
                        self._val_score_writer.add_scalar("f1_micro_{iter}".format(iter=iterations), score[0], epoch)
                        self._val_score_writer.add_scalar("f1_macro_{iter}".format(iter=iterations), score[1], epoch)
                    self._val_score_writer.flush()

                    if self._scheduler != None:
                        self._scheduler.step(loss)
                    
                    if self.args.val_criterion == 'loss' and self.min_val_loss > loss:
                        self.min_val_loss = loss
                        checkpoint = { 'model_link': self._model.state_dict(), 'epoch': epoch}
                        model_name = 'best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio)
                        torch.save(checkpoint, os.path.join(self.args.root_dir + '/checkpoints', model_name))
                        
                    if self.args.val_criterion  == 'acc' and self.max_val_score < score:
                        self.max_val_score = score
                        checkpoint = { 'model_link': self._model.state_dict(), 'epoch': epoch}
                        model_name = 'best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio)
                        torch.save(checkpoint, os.path.join(self.args.root_dir + '/checkpoints', model_name))      
                                    
                    if self.args.val_criterion == 'micro_f1' and self.max_val_score < (score[0]):
                        self.max_val_score = score[0]
                        checkpoint = { 'model_link': self._model.state_dict(), 'epoch': epoch}
                        model_name = 'best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio)
                        torch.save(checkpoint, os.path.join(self.args.root_dir + '/checkpoints', model_name))       
                        
                    if self.args.val_criterion == 'macro_f1' and self.max_val_score < (score[1]) :
                        self.max_val_score = score[1]
                        checkpoint = { 'model_link': self._model.state_dict(), 'epoch': epoch}
                        model_name = 'best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio)
                        torch.save(checkpoint, os.path.join(self.args.root_dir + '/checkpoints', model_name))                            
            self._model.eval()
            with torch.no_grad():
                self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio), map_location=self._device)['model_link'])
                output, additional_loss = self._model(self._dataset.x, self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                score = self.scorer(output, self._dataset.y, self._dataset.test_mask, iterations)  
                best_test_score_list.append(score)
                self.print_score(score, current = True, overall = False)
        self.print_score(best_test_score_list, current = False, overall = True)
        self._train_writer.close()
        self._val_writer.close()
        self._val_score_writer.close()
        self._train_add_loss_writer.close()
        self._val_add_loss_writer.close()



    def cancer_batch_train(self):
        if self.args.att_score: # get attention scores of (already) trained model
            return self.cancer_att_score()
        if self.args.he_emb: # get hyperedge embeddings of (already) trained model
            return self.get_hyperedge_emb()
        best_test_score_list = []
        self._dataset = self._dataset.to(self._device)
        length_train = self._dataset.train_mask[0].sum().item()
        length_val = self._dataset.val_mask[0].sum().item()
        length_test = self._dataset.test_mask[0].sum().item()
        num_train_batch = length_train//self.args.batch_size
        num_val_batch = length_val//self.args.batch_size
        num_test_batch = length_test//self.args.batch_size
        record_timer = []
        if length_train%self.args.batch_size >0:
            num_train_batch+=1
        if length_val%self.args.batch_size >0:
            num_val_batch+=1
        if length_test%self.args.batch_size >0:
            num_test_batch+=1
        for iterations in range(self.args.num_repeat):
            self._optimizer, self._scheduler = self.model_manager_result.set_optimizer()
            self.min_val_loss = float('inf')
            self.max_val_score = -1
            if self.args.silence and (not self.args.show_bar):
                epoch_range = [i for i in range(self.args.epoch)]
            else:
                epoch_range = tqdm([i for i in range(self.args.epoch)])
            utils.set_seed(seed = self._random_seed_numbers_list[iterations])
            self._model.reset_parameters()
            
            for epoch in epoch_range:
                if self.args.timer:
                    if epoch in [15,25,35,45,55]:
                        record_timer.append(time()-start_time)
                        if epoch == 55:
                            print("accuracy || mean : "+str(np.mean(np.asarray(score))) + " || std : "+str(np.std(np.asarray(score))))
                            print("TIME mean : {} / std : {}".format(str(np.mean(np.asarray(record_timer))), str(np.std(np.asarray(record_timer)))))
                            exit()
                    if epoch in [5,15,25,35,45]:
                        start_time = time()
                if (not self.args.silence) or (self.args.show_bar):
                    epoch_range.set_description("training...")
                    
                self._model.train()
                perm_result = torch.randperm(length_train)
                train_x = self._dataset.x[self._dataset.train_mask[iterations]][perm_result]
                train_y = self._dataset.y[self._dataset.train_mask[iterations]][perm_result]
                train_load_x = [ train_x[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_train_batch)]
                train_load_y = [ train_y[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_train_batch)]
                if self._dataset.xe != 'None':
                    train_xe = self._dataset.xe[self._dataset.train_mask[iterations]][perm_result]
                    train_load_xe = [ train_xe[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_train_batch)]
                
                for x_index in range(num_train_batch):
                    self._optimizer.zero_grad()
                    if self._dataset.xe != 'None':
                        output, additional_loss = self._model(train_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, train_load_xe[x_index], self._dataset.m)
                    else:
                        output, additional_loss = self._model(train_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                    loss = self.loss_cal_cancer(output, train_load_y[x_index], additional_loss)
                    loss.backward()
                    self._optimizer.step()
                
                self._model.eval()
                val_output_temp_list = None
                with torch.no_grad():
                    val_x = self._dataset.x[self._dataset.val_mask[iterations]]
                    val_load_x = [ val_x[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_val_batch)]
                    if self._dataset.xe != 'None':
                        val_xe = self._dataset.xe[self._dataset.val_mask[iterations]]
                        val_load_xe = [ val_xe[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_val_batch)]
                    for x_index in range(num_val_batch):
                        if self._dataset.xe != 'None':
                            output, additional_loss = self._model(val_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, val_load_xe[x_index], self._dataset.m)
                        else:
                            output, additional_loss = self._model(val_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                        
                        if x_index == 0 :
                            val_output_temp_list = output
                        else: 
                            val_output_temp_list = torch.vstack((val_output_temp_list, output))
                    score = self.scorer_cancer(val_output_temp_list, self._dataset.y, self._dataset.val_mask, iterations)
                    if self.args.use_wandb:
                        wandb.log({'valid_macro_f1':score[1].item()}, step=epoch)
                    if self._scheduler != None:
                        self._scheduler.step(loss)
                        
                    if self.args.val_criterion == 'macro_f1' and self.max_val_score < (score[1]) :
                        self.max_val_score = score[1]
                        checkpoint = { 'model_link': self._model.state_dict(), 'epoch': epoch}
                        model_name = self.config_str+'_'+str(iterations)+'.chkpt'
                        if self.args.hcl_spec == 4:
                            model_name = '4_'+ model_name
                        torch.save(checkpoint, os.path.join(self.args.root_dir + '/checkpoints', model_name))     
                if self.args.use_wandb:                      
                    self._model.eval()
                    test_output_temp_list = None
                    with torch.no_grad():
                        test_x = self._dataset.x[self._dataset.test_mask[iterations]]
                        test_load_x = [ test_x[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_test_batch)]
                        if self._dataset.xe != 'None':
                            test_xe = self._dataset.xe[self._dataset.test_mask[iterations]]
                            test_load_xe = [ test_xe[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_test_batch)]
                        for x_index in range(num_test_batch):
                            if self._dataset.xe != 'None':
                                output, additional_loss = self._model(test_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, test_load_xe[x_index], self._dataset.m)
                            else:
                                output, additional_loss = self._model(test_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                            if x_index == 0:
                                test_output_temp_list = output
                            else: 
                                test_output_temp_list = torch.vstack((test_output_temp_list, output))

                        score = self.scorer_cancer(test_output_temp_list, self._dataset.y, self._dataset.test_mask, iterations)  
                        wandb.log({'test_macro_f1':score[1].item()}, step=epoch)
                        
            self._model.eval()
            model_name = self.config_str+'_'+str(iterations)+'.chkpt'
            if self.args.hcl_spec == 4:
                model_name = '4_'+ model_name
            self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/'+model_name, map_location=self._device)['model_link'])
            test_output_temp_list = None
            with torch.no_grad():
                test_x = self._dataset.x[self._dataset.test_mask[iterations]]
                test_load_x = [ test_x[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_test_batch)]
                if self._dataset.xe != 'None':
                    test_xe = self._dataset.xe[self._dataset.test_mask[iterations]]
                    test_load_xe = [ test_xe[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range(num_test_batch)]
                for x_index in range(num_test_batch):
                    if self._dataset.xe != 'None':
                        output, additional_loss = self._model(test_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, test_load_xe[x_index], self._dataset.m)
                    else: 
                        output, additional_loss = self._model(test_load_x[x_index], self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
                    if x_index == 0:
                        test_output_temp_list = output
                    else: 
                        test_output_temp_list = torch.vstack((test_output_temp_list, output))

                score = self.scorer_cancer(test_output_temp_list, self._dataset.y, self._dataset.test_mask, iterations)  
                best_test_score_list.append(score)
                self.print_score(score, current = True, overall = False)
            if self.args.model.lower() in ['disen_hgnn', 'hsdn','disen_hgnn_ablation']:
                self.cancer_att_score(iterations)
            self.get_hyperedge_emb(iterations)
        self.print_score(best_test_score_list, current = False, overall = True)
        self._train_writer.close()
        self._val_writer.close()
        self._val_score_writer.close()
        self._train_add_loss_writer.close()
        self._val_add_loss_writer.close()

    def cancer_att_score(self, iterations=0): # get attention scores of each layer
        self._dataset = self._dataset.to(self._device)
        if True:            
            self._model.eval()
            model_name = self.config_str+'_'+str(iterations)+'.chkpt'
            if self.args.hcl_spec == 4:
                model_name = '4_'+ model_name
            self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/'+model_name, map_location=self._device)['model_link'])
            with torch.no_grad():
                for x_index in tqdm(range(self._dataset.exact_num_hypergraphs)):
                    att_score_out = self._model.get_hyperedge_attention_score(self._dataset.x[x_index].unsqueeze(0), self._dataset.edge_index)
                    if x_index == 0:
                        att_dist_list = att_score_out.unsqueeze(0)
                    else: 
                        att_dist_list = torch.vstack((att_dist_list, att_score_out.unsqueeze(0)))
        att_dist = att_dist_list.detach().cpu().numpy()
        file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_att_score.npy'
        if self.args.hcl_spec == 4:
            file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_4_att_score.npy'
        np.save(self.args.root_dir+file_name, att_dist)

        self._train_writer.close()
        self._val_writer.close()
        self._val_score_writer.close()
        self._train_add_loss_writer.close()
        self._val_add_loss_writer.close()
        
        
    def get_hyperedge_emb(self, iterations=0):
        self._dataset = self._dataset.to(self._device)
        self._model.eval()
        
        if self.args.task == 'bio': # cancer subtype classification task
            model_name = self.config_str+'_'+str(iterations)+'.chkpt'
            if self.args.hcl_spec == 4:
                model_name = '4_'+ model_name
            self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/'+model_name, map_location=self._device)['model_link'])
            with torch.no_grad():
                for x_index in tqdm(range(self._dataset.exact_num_hypergraphs)):
                    if self._dataset.xe == 'None':
                        he_emb = self._model.get_hyperedge_emb(self._dataset.x[x_index].unsqueeze(0), self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe)
                    else: 
                        he_emb = self._model.get_hyperedge_emb(self._dataset.x[x_index].unsqueeze(0), self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe[x_index].unsqueeze(0))
                    he_emb = torch.vstack(he_emb)
                    if x_index == 0:
                        he_emb_list = he_emb.unsqueeze(0)
                    else: 
                        he_emb_list = torch.vstack((he_emb_list, he_emb.unsqueeze(0)))
            he_emb_final = torch.vstack([a.unsqueeze(0) for a in he_emb_list]).detach().cpu().numpy()
            file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_he_emb.npy'
            if self.args.hcl_spec == 4:
                file_name = '/ablation/save_files/'+self.config_str+'_'+str(iterations)+'_4_he_emb.npy'
            np.save(self.args.root_dir+file_name, he_emb_final)
        elif self.args.task == 'basic': # Usual node classification task. ex) benchmark dataset
            self._model.load_state_dict(torch.load(self.args.root_dir + '/checkpoints/best_score_{}_{}_{}_{}_{}_{}_{}.chkpt'.format(self.args.model.lower(), self.args.dataset.lower(), iterations, self.args.disen_spec, self.args.dropout, self.args.disen_loss_ratio, self.args.interpol_ratio), map_location=self._device)['model_link'])
            output, additional_loss, he_emb = self._model(self._dataset.x, self._dataset.edge_index, self._dataset.edge_weight, self._dataset.xe, self._dataset.m)
            he_emb_final = torch.vstack([a.unsqueeze(0) for a in he_emb]).detach().cpu().numpy()
            np.save(self.args.root_dir+'/ablation/save_files/'+self.args.dataset+'_disloss_'+str(self.args.disen_loss_ratio)+'_dim_'+str(self.args.hidden)+'_factor_'+str(self.args.heads)+'_he_emb.npy', he_emb_final)

        self._train_writer.close()
        self._val_writer.close()
        self._val_score_writer.close()
        self._train_add_loss_writer.close()
        self._val_add_loss_writer.close()



    def scorer(self, output, answer, mask, iteration):
        preds = torch.argmax(F.log_softmax(output, dim = 1), dim = 1)
        
        
        if self._metric == 'acc':
            return accuracy_score(preds[mask[iteration]].cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy())*100
            #return torch.sum(torch.argmax(output[mask[iteration]], dim = 1) == answer[mask[iteration]]).item() / torch.sum(mask[iteration]).item() * 100
        
        elif self._metric == 'f1':
            micro_f1 = f1_score(preds[mask[iteration]].cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy(), average='micro')
            macro_f1 = f1_score(preds[mask[iteration]].cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy(), average='macro')
            return [micro_f1, macro_f1]
        else:
            raise NotImplementedError
        



    def scorer_cancer(self, output, answer, mask, iteration):
        preds = torch.argmax(F.log_softmax(output, dim = 1), dim = 1)
        if self._metric == 'acc':
            return accuracy_score(preds.cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy())*100
            #return torch.sum(torch.argmax(output[mask[iteration]], dim = 1) == answer[mask[iteration]]).item() / torch.sum(mask[iteration]).item() * 100
        
        elif self._metric == 'f1':
            micro_f1 = f1_score(preds.cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy(), average='micro')
            macro_f1 = f1_score(preds.cpu().detach().numpy(), answer[mask[iteration]].cpu().detach().numpy(), average='macro')
            return [micro_f1, macro_f1]
        else:
            raise NotImplementedError
        
    
    def loss_cal(self, output, answer, mask, iteration, additional_loss):
        if self._task == 'basic':
            loss = F.nll_loss(F.log_softmax(output, dim = -1)[mask[iteration]], answer[mask[iteration]])
            if additional_loss != None:
                loss += self.disen_loss_ratio * additional_loss
            return loss
        elif self._task == 'bio':
            loss = F.nll_loss(F.log_softmax(output, dim =-1), answer[mask[iteration]])
            if additional_loss != None:
                loss += self.disen_loss_ratio * additional_loss
            return loss
        
        else: 
            raise NotImplementedError
        
        return
    
    def loss_cal_cancer(self, output, answer, additional_loss):
        loss = F.nll_loss(F.log_softmax(output, dim =-1), answer)
        if additional_loss != None:
            loss += self.disen_loss_ratio * additional_loss
        return loss
            
            
    def print_score(self, score, current = True, overall = False): 
        assert current != overall, "cannot be both True or both False"
        if current == False and overall == True:
            if self._metric == 'acc':
                print("accuracy || mean : "+str(np.mean(np.asarray(score))) + " || std : "+str(np.std(np.asarray(score))))
                print()
                print()
            elif self._metric == 'f1': 
                score_array = np.asarray(score).T
                print("micro_f1 || mean : "+str(np.mean(score_array[0])) + " || std : "+str(np.std(score_array[0])))
                print("macro_f1 || mean : "+str(np.mean(score_array[1])) + " || std : "+str(np.std(score_array[1])))
                print()
                print()
                
            else: 
                raise NotImplementedError
            
        elif (not self.args.silence) or (self.args.show_bar):
            if self._metric == 'acc':
                print("accuracy : "+str(score)) 
            elif self._metric == 'f1': 
                print("micro_f1 : "+str(score[0])+"   ||   macro_f1 : "+str(score[1]))                
            else: 
                raise NotImplementedError            
            
        return
   