import torch
from torch import nn
import time
import numpy as np
import os
from componenets.metrics import metrics
from componenets.new_metrics import metric_new
from methods.STExplainer.stgib_series import STGIB
from methods.STExplainer.stgsat.stgsat import STGSAT
from methods.STExplainer.stgat.stgat import STGAT
import utils.util as util
import pandas as pd
import torch_geometric
import math
from tqdm import tqdm
from scipy.sparse import coo_matrix
from Params import args, logger
import copy
from torch import Tensor
def load_SE(num_node, d_model):
    # SE = torch.zeros([num_node, num_node])
    # for ind in num_node:
    #     SE[ind, ind] = 1
    # return SE
    pe = torch.zeros(num_node, d_model)
    position = torch.arange(0, num_node, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma
    return gamma(alpha + beta) / gamma(alpha) / gamma(beta) * x ** (alpha - 1) * (1 - x) ** (beta - 1)

def record_metric(data_record_dict, data_list, key_list):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        data_record_dict[key] = data
    return data_record_dict
def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)
def build_sp_tensor(adj_weight):
    coo_adj = coo_matrix(adj_weight)
    sp_adj = convert_sp_mat_to_sp_tensor(coo_adj)
    return sp_adj

def MAE_torch(pred, true, mask_value=0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def calculate_selected_nodes(edge_idx, edge_mask, top_k):
    # print(edge_mask.shape)
    
    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
    # if top_k == 12:
    #     threshold = 0.297
    hard_mask = (edge_mask > threshold).cpu()
    
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = edge_idx.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes
def graph_build_zero_filling(X, edge_index, node_mask: np.array):
    """ subgraph building through masking the unselected nodes with zero features """
    # X: B, L, N, C
    # node_mask: N or L
    if X.shape[1] == node_mask.shape[0]:
        node_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(node_mask, dim=-1), dim=-1), dim=0)
    elif X.shape[2] == node_mask.shape[0]:
        node_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(node_mask, dim=-1), dim=0), dim=0)
    else:
        raise ValueError('dim not match')
    ret_X = X * node_mask
    return ret_X, edge_index
def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError
def gnn_score(coalition: list, data: Tensor, value_func: str,
              subgraph_building_method='zero_filling', spat_dim = True, temp_dim = True, edge_idx_spat: Tensor = None, edge_idx_temp: Tensor = None) -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    if spat_dim:
        num_nodes = data.shape[2]
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.device)
        mask[coalition] = 1.0
        ret_x, ret_edge_idx_spat = subgraph_build_func(data, edge_idx_spat, mask)
    else:
        ret_x, ret_edge_idx_spat = data, edge_idx_spat
    if temp_dim:
        num_nodes = data.shape[1]
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.device)
        mask[coalition] = 1.0
        ret_x, ret_edge_idx_temp = subgraph_build_func(ret_x, edge_idx_temp, mask)
    else:
        ret_x, ret_edge_idx_temp = ret_x, edge_idx_temp

    # print(ret_edge_idx_spat.shape)
    # print(ret_edge_idx_temp.shape)
    
    score = value_func(ret_x, edge_idx_spat = ret_edge_idx_spat.long(), edge_idx_temp = ret_edge_idx_temp.long())
    # get the score of predicted class for graph or specific node idx
    # print(type(score))
    # return score.item()
    return score
class trainer():
    def __init__(self, scaler, sp_adj = None, sp_adj_w = None, temp_adj = None):
        self.scaler = scaler
        self.sp_adj = sp_adj
        self.sp_adj_w = sp_adj_w
        self.temp_adj = temp_adj
        SE = load_SE(args.num_nodes, 64)
        SE = SE.to("cuda:0")
        
        if args.model == 'STGIB':
            self.model = STGIB(sp_adj, sp_adj_w, temp_adj)
        elif args.model == 'STGSAT':
            self.model = STGSAT(sp_adj, sp_adj_w, temp_adj)
        elif args.model == 'STGAT':
            self.model = STGAT(sp_adj, sp_adj_w, temp_adj)
        else:
            raise ValueError('Model :{} error'.format(args.model))


        
        if args.testonly:
            # self.model.load("checkpoints/TaxiBJ/model_finetune.pth")
            self.model.load(args.mdir+args.name+'.pkl')
            self.model = self.model.to(args.device)
        else:
            self.model = self.model.to(args.device)
        self.optimizer, self.lr_scheduler = self.get_optim()
        self.criterion = self.get_criterion()
        
        # early stop
        self.patience = args.patience 
        self.trigger = 0
        self.last_loss = 100000
        self.last_mape_loss = 100000
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.build_beta_list(args.beta1, args.beta2)
    def decorate_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(args.device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(args.device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.decorate_batch(value)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(args.device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(self.decorate_batch(value))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        elif isinstance(batch, torch_geometric.data.batch.DataBatch):
            return batch.to(args.device)
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))
     
    def train(self, epoch, trnloader, tra_val_metric):
        tra_loss = []
        pre_loss = []
        other_loss = []
        gsat_loss_list_sp = []
        gsat_loss_list_temp = []
        ixz_loss = []
        structure_loss = []
        self.model.train()
        total_days = (args.end_date - args.start_date).days+1
        ids = np.random.permutation(list(range(args.lag, total_days)))
        num = len(ids)
        beta1 = self.beta1_list[epoch] if self.beta1_list is not None else None
        beta2 = self.beta2_list[epoch] if self.beta1_list is not None else None
        
        for idx, batch in tqdm(enumerate(trnloader)):
            # reg_info = dict()
            self.optimizer.zero_grad()
            batch = self.decorate_batch(batch)
            X, Y, TE = batch
            # print('feats :{}'.format(feats.shape)) # 58944, 1
            # print('edge_index :{}'.format(bg.edge_index.shape)) # 2, 130560
            # print(X)
            if args.model == 'STGIB':
                t_emb = TE[:, :args.lag, :] # B, T, 2
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1)
                output = self.model(X)
            elif args.model == 'STGSAT':
                t_emb = TE[:, :args.lag, :] # B, T, 2
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1)
                output = self.model(X, epoch=epoch)
            elif args.model == 'STGAT':
                t_emb = TE[:, :args.lag, :] # B, T, 2
                t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                X = torch.cat([X, t_emb], dim = -1)
                output = self.model(X, epoch=epoch)
            

            output  = self.scaler.inverse_transform(output)
            Y = self.scaler.inverse_transform(Y)
            if args.model == 'STGIB':
                main_loss = self.criterion(output, Y)
                # IB loss
                # reg_info["ixz_list"] -- len: 4, 32*num_nodes

                ixz_spat = torch.stack(self.model.reg_info["ixz_list"][:2], -1).mean(0).mean(0).sum()
                if args.only_spat is False:
                    ixz_temp = torch.stack(self.model.reg_info["ixz_list"][2:], -1).mean(0).mean(0).sum()
                    ixz = ixz_spat + ixz_temp
                else:
                    ixz = ixz_spat
                # print('ixz: {}'.format(ixz))
                if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                    ixz_1_spat = torch.stack(self.model.reg_info["ixz_DN_list"][:2], 1).mean(0).mean(0).sum()
                    if args.only_spat is False:
                        ixz_1_temp = torch.stack(self.model.reg_info["ixz_DN_list"][2:], 1).mean(0).mean(0).sum()
                        ixz_1 = ixz_1_spat + ixz_1_temp
                    else:
                        ixz_1 = ixz_1_spat
                    ixz =  ixz + ixz_1
                
                structure_kl_loss = torch.stack(self.model.reg_info["structure_kl_list"]).mean(0).mean()
                # print('structure_kl_list: {}'.format(structure_kl_loss.shape))
                if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                    structure_kl_loss_1 =  torch.stack(self.model.reg_info["structure_kl_DN_list"]).mean(0).mean()
                    structure_kl_loss = structure_kl_loss + structure_kl_loss_1
                loss = main_loss + ixz * beta1 + structure_kl_loss * beta2
                pre_loss.append(main_loss.item())
                ixz_loss.append(ixz.item() * beta1)
                structure_loss.append(structure_kl_loss.item() * beta2)
            elif args.model == 'STGSAT':
                main_loss = self.criterion(output, Y)
                ## gsat loss
                gsat_sp_loss = self.model.reg_info["loss_sp"][0].mean()
                gsat_temp_loss = self.model.reg_info["loss_temp"][0].mean()
                loss = main_loss + gsat_sp_loss*beta1 + gsat_temp_loss*beta2
                pre_loss.append(main_loss.item())
                gsat_loss_list_sp.append((gsat_sp_loss*beta1).item())
                gsat_loss_list_temp.append((gsat_temp_loss*beta2).item())
            elif args.model == 'STGAT':
                main_loss = self.criterion(output, Y)
                loss = main_loss
                pre_loss.append(main_loss.item())
            
            
            loss.backward()
            
            # add max grad clipping
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            tra_loss.append(loss.item())
            
        self.lr_scheduler.step()
        tra_loss = np.mean(tra_loss)
        pre_loss = np.mean(pre_loss)
        # other_loss = np.mean(other_loss)
        # gsat_loss_list = np.mean(gsat_loss_list)
        if args.model == 'STGIB':
            ixz_loss = np.mean(ixz_loss)
            structure_loss = np.mean(structure_loss)
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss, ixz_loss, structure_loss], ['epoch', 'train loss', 'predict loss', 'ixz_loss', 'structure_loss'])
        elif args.model == 'STGSAT' or args.model == 'STGSAT_abl':
            gsat_loss_list_sp = np.mean(gsat_loss_list_sp)
            gsat_loss_list_temp = np.mean(gsat_loss_list_temp)
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss, gsat_loss_list_sp, gsat_loss_list_temp], ['epoch', 'train loss', 'predict loss', 'gsat loss spat', 'gsat loss temp'])
        elif args.model == 'STGAT':
            
            tra_val_metric = record_metric(tra_val_metric, [epoch, tra_loss, pre_loss], ['epoch', 'train loss', 'predict loss'])
        else: 
            raise ValueError('Model :{} error, in Display Loss'.format(args.model))
        return tra_val_metric

    def build_beta_list(self, beta1=0.001, beta2 = 0.01):
        beta_init = 0
        init_length = int(args.max_epoch / 4)
        anneal_length = int(args.max_epoch / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta1_inter = beta_inter / 4 * (beta_init - beta1) + beta1
        self.beta1_list = np.concatenate([np.ones(init_length) * beta_init, beta1_inter, 
                                     np.ones(args.max_epoch - init_length - anneal_length + 1) * beta1])
                        
        beta_init = 0
        init_length = int(args.max_epoch / 4)
        anneal_length = int(args.max_epoch / 4)
        beta_inter = Beta_Function(np.linspace(0,1,anneal_length),1,4)
        beta2_inter = beta_inter / 4 * (beta_init - beta2) + beta2
        self.beta2_list = np.concatenate([np.ones(init_length) * beta_init, beta2_inter, 
                                     np.ones(args.max_epoch - init_length - anneal_length + 1) * beta2])
    
    def validation(self,  epoch, valloader, tra_val_metric):
        val_loss = []
        trues = []
        preds = []
        ids = np.random.permutation(list(range(args.lag, 921)))
        num = len(ids)
        with torch.no_grad():
            self.model.eval()

            for idx, batch in tqdm(enumerate(valloader)):
                batch = self.decorate_batch(batch)
                X, Y, TE = batch
                # print('feats :{}'.format(feats.shape))
                if args.model == 'STGIB':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X)
                elif args.model == 'STGSAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X, epoch=epoch)
                elif args.model == 'STGAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X, epoch=epoch)
                

                output = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                if args.model != 'STHSL':
                    loss = self.criterion(output, Y)
                # loss = self.criterion(output, Y)
                val_loss.append(loss.item())
                
                trues.append(Y.detach().cpu().numpy())
                preds.append(output.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        print(trues.shape, preds.shape)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        tra_val_metric = record_metric(tra_val_metric, [val_loss, mae, rmse, mape*100, smape*100, corr], ['val loss', 'mae', 'rmse', 'mape(%)', 'smape(%)', 'corr'])
        
        # stopFlg = self.earlyStop( epoch, mae, mape)
        stopFlg = self.earlyStop( epoch, mape, mape)

        return tra_val_metric, stopFlg

    
    
    def test(self, tstloader, ):
        self.model.load_state_dict(torch.load(args.mdir+args.name+'.pkl'), False)

        trues = []
        preds = []
        # trues_torch = []
        # preds_torch = []
        ids = np.random.permutation(list(range(args.lag, 921)))
        num = len(ids)
        spa_edge_weights = []
        spa_feat_weights = []
        temp_edge_weights = []
        temp_feat_weights = []
        with torch.no_grad():
            self.model.eval()

            for idx, batch in enumerate(tstloader):
                batch = self.decorate_batch(batch)
                X, Y, TE = batch
                if args.model == 'STGIB':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X)
                    spa_edge_w = self.model.reg_info['spa_alpha'][0].unsqueeze(0)
                    print(spa_edge_w.shape)
                    spa_edge_weights.append(spa_edge_w.cpu().detach().numpy())
                    temp_edge_w = self.model.reg_info['temp_alpha'][0].unsqueeze(0)
                    temp_edge_weights.append(temp_edge_w.cpu().detach().numpy())
                elif args.model == 'STGSAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    # save_file = np.load('./XAI_save/run_save.npz')
                    # spa_edge_w = save_file['spa_edge_weights']
                    # selected_nodes_spat = calculate_selected_nodes(self.model.edge_idx_spa, spa_edge_w, 90)

                    # maskout_nodes_list_spat = [node for node in range(args.num_nodes) if node not in selected_nodes_spat]

                    output = self.model(X, epoch=0)
                    # print(self.model.reg_info['spa_edge_weight'][0].cpu().detach().numpy().shape)
                    # spa_edge_weights.append(self.model.reg_info['spa_edge_weight'][0].cpu().detach().numpy())
                    # spa_feat_weights.append(self.model.reg_info['spa_feat_weight'][0].cpu().detach().numpy())
                    # temp_edge_weights.append(self.model.reg_info['temp_edge_weight'][0].cpu().detach().numpy())
                    # temp_feat_weights.append(self.model.reg_info['temp_feat_weight'][0].cpu().detach().numpy())
                elif args.model == 'STGAT':
                    t_emb = TE[:, :args.lag, :] # B, T, 2
                    t_emb = torch.unsqueeze(t_emb, dim=2).repeat(1, 1, args.num_nodes, 1)
                    X = torch.cat([X, t_emb], dim = -1)
                    output = self.model(X, epoch=0)

                output = self.scaler.inverse_transform(output)
                Y = self.scaler.inverse_transform(Y)
                
                trues.append(Y.detach().cpu().numpy())
                preds.append(output.detach().cpu().numpy())
                # trues_torch.append(Y)
                # preds_torch.append(output)

        # val_loss = np.mean(val_loss)
        trues, preds = np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)
        # if args.model == 'STGSAT':
        #     spa_edge_weights = np.concatenate(spa_edge_weights, axis=0)
        #     spa_feat_weights = np.concatenate(spa_feat_weights, axis=0)
        #     temp_edge_weights = np.concatenate(temp_edge_weights, axis=0)
        #     temp_feat_weights = np.concatenate(temp_feat_weights, axis=0)
        # elif args.model == 'STGIB':
        #     spa_edge_weights = np.concatenate(spa_edge_weights, axis=0).mean(0)
        #     temp_edge_weights = np.concatenate(temp_edge_weights, axis=0).mean(0)

        # print(spa_edge_weights)
        # print(temp_edge_weights)

        # np.savez('./XAI_save/run_save_gib.npz', trues=trues, preds=preds, spa_edge_weights=spa_edge_weights, temp_edge_weights=temp_edge_weights)
        for t in range(trues.shape[1]):
            mae, rmse, mape, smape, corr = metrics(preds[:, t, ...], trues[:, t, ...], args.mae_thresh, args.mape_thresh)
            log = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
                t + 1, mae, rmse, mape * 100, smape * 100, corr)
            logger.info(log)
        mae, rmse, mape, smape, corr = metrics(preds, trues, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, Best Epoch: {}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, sMAPE: {:.4f}%, Corr: {:.4f}".format(
            self.best_epoch, mae, rmse, mape * 100, smape * 100, corr))
        # preds_tch, trues_ch = torch.cat(preds_torch, dim = 0), torch.cat(trues_torch, dim = 0)
        # mae, mape, rmse= metric_new(preds_tch, trues_ch)
        # logger.info("Average Horizon, New Metrics: {}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #     self.best_epoch, mae, rmse, mape))




    def get_optim(self, ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=(0.9, 0.999))
        steps = args.steps
        
        lr_decay_ratio = args.lr_decay_ratio
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                                gamma=lr_decay_ratio)
        return optimizer, lr_scheduler

    def get_criterion(self, ):
        if args.criterion == 'MSE':
            return nn.MSELoss()
        elif args.criterion == 'Smooth':
            return nn.SmoothL1Loss()
        elif args.criterion == 'MAE':
            return MAE_torch
    

    def earlyStop(self, epoch, current_loss, mape_loss):
        if epoch >= 100:
            if current_loss >= self.last_loss or epoch == args.max_epoch:
        # if epoch >= 0:
        #     if epoch == 1:
                if current_loss < self.last_loss:
                    self.trigger = 0
                    self.last_loss = current_loss
                    self.last_mape_loss = mape_loss
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.model.state_dict())
                else:
                    self.trigger += 1
                if self.trigger >= self.patience or epoch == args.max_epoch:   
                    print('Early Stopping! The best epoch is ' + str(self.best_epoch))
                    if not os.path.exists(args.mdir):
                        os.makedirs(args.mdir)
                    torch.save(self.best_state,args.mdir+args.name+'.pkl')
                    return True
            else:
                self.trigger = 0
                self.last_loss = current_loss
                self.last_mape_loss = mape_loss
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                return False