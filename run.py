import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

from pprint import pprint
from progressbar import progressbar

from YahooDataReader import YahooDataReader
from train import trainAdversarialRanking
from test import testAdvarsarialRanking
import pickle as pkl
import torch
import itertools
from sklearn.model_selection import KFold
import time
from zehlike import *
from train_yahoo_dataset import on_policy_training
from YahooDataReader import YahooDataReader
from models import NNModel, LinearModel
from evaluation import evaluate_model



dr = YahooDataReader(None)
dr.data = pkl.load(open("GermanCredit/german_train_rank.pkl", "rb")) # (data_X, data_Y) 500*25*29, 500*25*1
vdr = YahooDataReader(None)
vdr.data = pkl.load(open("GermanCredit/german_test_rank.pkl","rb"))    # (data_X, data_Y) 100*25*29, 100*25*1


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace(conditional_model=True, gpu_id=None, progressbar=True, evaluate_interval=250, input_dim=29, 
                 eval_rank_limit=1000,
                fairness_version="asym_disparity", entropy_regularizer=0.0, save_checkpoints=False, num_cores=8,
                pooling='concat_avg', dropout=0.0, hidden_layer=8, summary_writing=False, 
                 group_fairness_version="asym_disparity",early_stopping=False, lr_scheduler=False, 
                 validation_deterministic=False, evalk=1000, reward_type="ndcg", baseline_type="value", 
                 use_baseline=True, entreg_decay=0.0, skip_zero_relevance=True, eval_temperature=1.0, optimizer="Adam",
               clamp=False)
torch.set_num_threads(args.num_cores)
args.progressbar = False 

args.group_feat_id = 3
args.mu = 1e-2

dr_x = np.array(dr.data[0])
dr_y = np.array(dr.data[1])
vdr_x = np.array(vdr.data[0])
vdr_y = np.array(vdr.data[1])

nc,nn,nf = np.shape(dr_x)

lamdas_list = [0.1, 1, 10, 100, 500 ,1000, 2000, 3000, 5000, 10000]#[1e-3, 1e-2, 1e-1, 1e0, 1e1] #lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
gammas_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
best_lamda = -1.0
best_ndcg = -1.0
mu = 1e-2 # No need to change?

n_splits = 3

def train_func(data):
    
    lamda, gamma = data
    args.lambda_group_fairness = lamda
    args.gamma = gamma
    # prepare cross validation
    k_fold = KFold(n_splits, True, 1)
    cv_ndcg, cv_fair_loss = 0, 0
    cv_theta = np.zeros(nf)
    for fold_idx, (train_set, val_set) in enumerate(k_fold.split(dr_x)):
        print("fold_idx: ", fold_idx)
        print("train_set: ", train_set)
        print("test_set: ", val_set)
        model = trainAdversarialRanking(dr_x[train_set], dr_y[train_set], args=args)
        results = testAdvarsarialRanking(dr_x[val_set], dr_y[val_set], model, args=args)
        cv_ndcg += results["ndcg"]
        cv_fair_loss += results["fair_loss"]
        cv_theta += model["theta"]
    cv_ndcg, cv_fair_loss, cv_theta = cv_ndcg/n_splits, cv_fair_loss/n_splits, cv_theta/n_splits
 
    return lamda, gamma, cv_ndcg, cv_fair_loss, cv_theta

def test_func(data):

    lamda = data[0][0]
    gamma = data[0][1][0]
    theta = data[0][1][3]
    args.lambda_group_fairness = lamda
    args.gamma = gamma
    model = {"theta": theta }
    results = testAdvarsarialRanking(vdr_x ,vdr_y , model, args=args)
    return lamda, results["ndcg"], results["fair_loss"]

def parallel_runs(data_list):
    ls = []
    pool = multiprocessing.Pool(processes=args.num_cores)
    #prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(train_func, data_list)
    res = {key : [result_list[idx][1:5]
      for idx in range(len(result_list)) if result_list[idx][0]== key]
      for key in set([x[0] for x in result_list])}
    for key, values in res.items():
        ls.append([(key,values[idx]) for idx in range(len(res[key])) if  values[idx][1]==max([x[1] for x in values])])
   
    pool = multiprocessing.Pool(processes=args.num_cores)
    result_list = pool.map(test_func, ls)
    print(result_list)
    sorted_result = sorted(result_list, key=lambda x: x[0])
    return sorted_result

#############################################################

def policy_parallel(data): # run policy code in parallel using multiple cores 
    
    lamda = data
    args.lambda_group_fairness = lamda
    args.lambda_reward = 1.0
    args.lambda_ind_fairness = 0.0
    args.lr = 0.001
    args.epochs = 10
    args.progressbar = False
    args.weight_decay = 0.0
    args.sample_size = 25
    args.optimizer = "Adam"
    
    model = LinearModel(D=args.input_dim)
    model = on_policy_training(dr, vdr, model, args=args)
    if lamda == 0:
        results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                             deterministic=True, args=args, num_sample_per_query=20)
        print(results)
        plt_data_pl = [results["ndcg"], results["avg_group_asym_disparity"]]
    else: # changed by me! for lamda == 0 there were 2 outputs 
        results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                                 deterministic=False, args=args, num_sample_per_query=20)
        print(results)
        print("Lambda: ", lamda)
        model_params_list.append(model.w.weight.data.tolist()[0])
        print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
        plt_data_pl = [results["ndcg"], results["avg_group_asym_disparity"]]
    
    return lamda, results["ndcg"], results["avg_group_asym_disparity"]


def policy_learning(): 
    
    lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
    pool = multiprocessing.Pool(processes=args.num_cores)
    #prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(policy_parallel, lambdas_list)
    print(result_list)
    sorted_result = sorted(result_list, key=lambda x: x[0])
    return sorted_result
    
    
# def policy_learning():
#     args.group_feat_id = 3
#     model_params_list = []
#     lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
#     plt_data_pl = np.zeros((len(lambdas_list)+1, 2))
#     for i, lgroup in enumerate(lambdas_list):
#         torch.set_num_threads(args.num_cores)
#         args.lambda_reward = 1.0
#         args.lambda_ind_fairness = 0.0
#         args.lambda_group_fairness = lgroup

#         args.lr = 0.001
#         args.epochs = 10
#         args.progressbar = False
#         args.weight_decay = 0.0
#         args.sample_size = 25
#         args.optimizer = "Adam"

#         model = LinearModel(D=args.input_dim)

#         model = on_policy_training(dr, vdr, model, args=args)
#         if i == 0:
#             results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
#                                  deterministic=True, args=args, num_sample_per_query=20)
#             print(results)
#             plt_data_pl[0] = [results["ndcg"], results["avg_group_asym_disparity"]]
#         results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
#                                  deterministic=False, args=args, num_sample_per_query=20)
#         print(results)
#         model_params_list.append(model.w.weight.data.tolist()[0])
#         print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
#         plt_data_pl[i+1] = [results["ndcg"], results["avg_group_asym_disparity"]]
#     return plt_data_pl

##############################################################
def zehlike():
    args.lr = [0.001]
    args.lambda_reward = 1.0
    plt_data_z = []
    lambdas = [0.0, 1.0, 10, 100, 1000, 10000, 100000, 1000000]
    args.weight_decay = [0.0]
    args.epochs = [10]
    disparities_mat = np.zeros((len(lambdas), 1))
    ndcg_mat = np.zeros((len(lambdas), 1))
    for i, lg in enumerate(lambdas):
        args.lambda_group_fairness = lg
        model = LinearModel(D=args.input_dim)
        model = demographic_parity_train(model, dr, vdr, vvector(200), args)
        results = evaluate_model(
                model,
                vdr,
                fairness_evaluation=False,
                group_fairness_evaluation=True,
                deterministic=True,
                args=args,
                num_sample_per_query=100)
        plt_data_z.append([results["ndcg"], results["avg_group_asym_disparity"]])
        ndcg_mat[i, 0], disparities_mat[i,0] = results["ndcg"], results["avg_group_asym_disparity"]
    return np.array(plt_data_z)
###############################################################

def ndcg_vs_disparity_plot(plt_data_mats, names, join=False, ranges=None, filename="tradeoff"):
    plt.figure(figsize=(6.5, 4))
    if ranges:
        plt.xlim(ranges[0])
        plt.ylim(ranges[1])
    #initialize_params() ##Omid
    for i, plt_data_mat in enumerate(plt_data_mats):
        if not join:
            plt.scatter(
                plt_data_mats[i][:, 0],
                plt_data_mats[i][:, 1],
                marker="*",
                label=names[i])
        else:
            plt.plot(
                plt_data_mats[i][:, 0],
                plt_data_mats[i][:, 1],
                marker="*",
                linestyle='--',
                label=names[i])
    plt.legend(fontsize=14)
#     plt.title("Utility-Fairness trade-off",y=-0.30)
    plt.xlabel("NDCG")
    plt.ylabel(r'$\hat{\mathcal{D}}_{\rm group}$', fontsize=16)
    plt.grid()
    #plt.savefig('./plots/german_tradeoff.pdf', bbox_inches='tight')
    plt.savefig('./plots/'+ filename + '.pdf', bbox_inches='tight')
    plt.show()
    
###########################################################################################
start_adv = time.time()
data_list = list(itertools.product(lamdas_list, gammas_list))
##adv_result = parallel_runs(data_list)
##plt_data_adv = np.array([[adv_result[i][1], adv_result[i][2]] for i in range(len(adv_result))])
end_adv = time.time()
#################################################################################
policy_result = policy_learning()
plt_data_pl = np.array([[policy_result[i][1], policy_result[i][2]] for i in range(len(policy_result))])
end_policy = time.time()
##################################################################################
plt_data_z = zehlike()
end_zehlike = time.time()
###################################################################################
plt_data_adv =  np.array([[7.87678189e-01, 4.31883047e-04], [7.87694120e-01, 1.07887506e-04]])
#plt_data_pl = np.array([[0.93654663, 0.02231645], [0.92797832, 0.01746921], [0.84281633, 0.00131007]])

print("plt_data_adv: ", plt_data_adv)
print("plt_data_pl: ", plt_data_pl)
print("plt_data_z: ", plt_data_z)

ndcg_vs_disparity_plot([plt_data_adv, plt_data_pl], ["Robust_Fair ($\lambda \in [0, 10000]$)",
                      "Policy_Ranking($\lambda \in [0,100]$ )"], join=True, ranges=[[0.70, 0.95], [0.00, 0.040]], filename= "german_robust_policy_tradeoff")

ndcg_vs_disparity_plot([plt_data_adv, plt_data_pl,  plt_data_z], ["Robust_Fair ($\lambda \in [0, 10^4]$)",
                      "Policy_Ranking ($\lambda \in [0,100]$ )", 
                      "Zehlike ($\lambda \in [0, 10^6]$)"], join=True, ranges=[[0.60, 0.85], [0.00, 0.040]], filename="german_robust_policy_zehlike_tradeoff")

elapsed_adv = end_adv - start_adv
elapsed_policy = end_policy - end_adv
elapsed_zehlike = end_zehlike - end_policy
print("time for Robust_Fair: ", elapsed_adv)
print("time for Policy_Learning: ", elapsed_policy)
print("time for Zehlike: ", elapsed_zehlike)



