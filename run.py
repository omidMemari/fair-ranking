import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

from pprint import pprint
from progressbar import progressbar

from YahooDataReader import YahooDataReader
from train import trainAdversarialRanking
#from train import fair_adv_training
#from test import fair_adv_testing
from test import testAdvarsarialRanking
import pickle as pkl
import torch
import itertools
from sklearn.model_selection import KFold
import time

from train_yahoo_dataset import on_policy_training
from YahooDataReader import YahooDataReader
from models import NNModel, LinearModel
from evaluation import evaluate_model

#########################################################################
# #%matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('pdf', 'png')
# plt.rcParams['savefig.dpi'] = 75

# plt.rcParams['figure.autolayout'] = False
# plt.rcParams['figure.figsize'] = 10, 6
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.titlesize'] = 16
# plt.rcParams['font.size'] = 16
# plt.rcParams['lines.linewidth'] = 2.0
# plt.rcParams['lines.markersize'] = 8
# plt.rcParams['legend.fontsize'] = 14

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.serif'] = "cm"
# plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"
#####################################################################################


# run your code
#end = time.time()

#elapsed = end - start

dr = YahooDataReader(None)
dr.data = pkl.load(open("GermanCredit/german_train_rank_3.pkl", "rb")) # (data_X, data_Y) 500*25*29, 500*25*1
vdr = YahooDataReader(None)
vdr.data = pkl.load(open("GermanCredit/german_test_rank_3.pkl","rb"))    # (data_X, data_Y) 100*25*29, 100*25*1


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

dr_x = np.array(dr.data[0][:10])
dr_y = np.array(dr.data[1][:10])
vdr_x = np.array(vdr.data[0][:10])
vdr_y = np.array(vdr.data[1][:10])

nc,nn,nf = np.shape(dr_x)

lamdas_list = [500, 2000]#[500 ,1000, 2000, 3000, 4000]#[1e-3, 1e-2, 1e-1, 1e0, 1e1] #lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
gammas_list = [1e-2]#[1e-2, 1e-1, 1e0, 1e1, 1e2]
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

def policy_learning():
    args.group_feat_id = 3
    model_params_list = []
    lambdas_list = [0.0, 10.0]#[0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
    plt_data_pl = np.zeros((len(lambdas_list)+1, 2))
    for i, lgroup in enumerate(lambdas_list):
            torch.set_num_threads(args.num_cores)
            args.lambda_reward = 1.0
            args.lambda_ind_fairness = 0.0
            args.lambda_group_fairness = lgroup

            args.lr = 0.001
            args.epochs = 10
            args.progressbar = False
            args.weight_decay = 0.0
            args.sample_size = 25
            args.optimizer = "Adam"

            model = LinearModel(D=args.input_dim)

            model = on_policy_training(dr, vdr, model, args=args)
            if i == 0:
                results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                                     deterministic=True, args=args, num_sample_per_query=20)
                print(results)
                plt_data_pl[0] = [results["ndcg"], results["avg_group_asym_disparity"]]
            results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                                     deterministic=False, args=args, num_sample_per_query=20)
            print(results)
            model_params_list.append(model.w.weight.data.tolist()[0])
            print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
            plt_data_pl[i+1] = [results["ndcg"], results["avg_group_asym_disparity"]]
    return plt_data_pl

###############################################################

def ndcg_vs_disparity_plot(plt_data_mats, names, join=False, ranges=None):
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
    plt.savefig('./plots/german_tradeoff.pdf', bbox_inches='tight')
    plt.show()
    
###########################################################################################
start_adv = time.time()
data_list = list(itertools.product(lamdas_list, gammas_list))
adv_result = parallel_runs(data_list)
plt_data_adv = np.array([[adv_result[i][1], adv_result[i][2]] for i in range(len(adv_result))])
end_adv = time.time()
plt_data_pl = policy_learning()
end_policy = time.time()


#plt_data_adv =  np.array([[7.87678189e-01, 4.31883047e-04], [7.87694120e-01, 1.07887506e-04]])
#plt_data_pl = np.array([[0.93654663, 0.02231645], [0.92797832, 0.01746921], [0.84281633, 0.00131007]])
print("plt_data_adv: ", plt_data_adv)
print("plt_data_pl: ", plt_data_pl)

ndcg_vs_disparity_plot([plt_data_adv, plt_data_pl], ["Robust_Fair ($\lambda \in [0, 0.2]$)",
                      "Policy_Ranking($\lambda \in [0,100]$ )"], join=True, ranges=[[0.60, 0.85], [0.00, 0.054]])


end = time.time()
elapsed_adv = end_adv - start_adv
elapsed_policy = end_policy - end_adv
print("time for Robust_Fair: ", elapsed_adv)
print("time for Policy_Learning: ", elapsed_policy)




