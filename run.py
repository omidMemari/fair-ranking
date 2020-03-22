import pandas as pd
import numpy as np
import multiprocessing
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


start = time.time()
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
                fairness_version="asym_disparity", entropy_regularizer=0.0, save_checkpoints=False, num_cores=1,
                pooling='concat_avg', dropout=0.0, hidden_layer=8, summary_writing=False, 
                 group_fairness_version="asym_disparity",early_stopping=False, lr_scheduler=False, 
                 validation_deterministic=False, evalk=1000, reward_type="ndcg", baseline_type="value", 
                 use_baseline=True, entreg_decay=0.0, skip_zero_relevance=True, eval_temperature=1.0, optimizer="Adam",
                clamp=False)
torch.set_num_threads(args.num_cores)
args.progressbar = False 

args.group_feat_id = 4
args.mu = 1e-2

dr_x = np.array(dr.data[0][:20])
dr_y = np.array(dr.data[1][:20])
vdr_x = np.array(vdr.data[0][:20])
vdr_y = np.array(vdr.data[1][:20])

nc,nn,nf = np.shape(dr_x)

#X_train = dr.data[0][:2] #train.data[0][:20]
#Y_train = dr.data[1][:2] #train.data[1][:20]
#X_test  = vdr.data[0][:2]  #test.data[0][:20]
#Y_test  = vdr.data[1][:2]  #test.data[1][:20]

lamdas_list = [1000, 2000]#[1e-3, 1e-2, 1e-1, 1e0, 1e1] #lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
gammas_list = [1e-1]#[1e-2, 1e-1, 1e0, 1e1, 1e2]
best_lamda = -1.0
best_ndcg = -1.0
kf = 1 #5 # k-fold
mu = 1e-2 # No need to change?

#lamda = 0.01
#gamma = 2000
n_splits = 3

def myfunc(data):
    
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

def parallel_runs(data_list): 
    pool = multiprocessing.Pool(processes=4)
    #prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(myfunc, data_list)
    print(result_list)



data_list = list(itertools.product(lamdas_list, gammas_list))
parallel_runs(data_list)

# model_params_list = []
# lambdas_list = [0.01, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0] # somehow add 0.0 , division by zero!!
# plt_data_pl = np.zeros((len(lamdas_list)+1, 2))
# for i, lamda in enumerate(lamdas_list):
#         #args.lamda = lamda
#         #torch.set_num_threads(args.num_cores)
#         args.lambda_reward = 1.0
#         args.lambda_ind_fairness = 0.0
#         args.lambda_group_fairness = lamda
#         args.lr = 0.001
#         args.epochs = 10
#         args.progressbar = False
#         args.weight_decay = 0.0
#         args.sample_size = 25
#         args.optimizer = "Adam"
       
#         for gamma in gammas_list:
#             args.gamma = gamma
#             #train_feats, train_rel = shuffle_combined(train_feats, train_rel)
 
#             # prepare cross validation
#             k_fold = KFold(n_splits, True, 1)
#             # enumerate splits
#             #for train_index, val_index in kfold.split(dr_new[0]):
#             #    print('train: %s, test: %s' % (train_index, val_index))
#             cv_ndcg, cv_fair_loss = 0, 0
#             cv_theta = np.zeros(nf)
#             for fold_idx, (train_set, val_set) in enumerate(k_fold.split(dr_x)):
#                 print("fold_idx: ", fold_idx)
#                 print("train_set: ", train_set)
#                 print("test_set: ", val_set)
#                 model = trainAdversarialRanking(dr_x[train_set], dr_y[train_set], args=args)
#                 results = testAdvarsarialRanking(dr_x[val_set], dr_y[val_set], model, args=args)
#                 cv_ndcg += results["ndcg"]
#                 cv_fair_loss += results["fair_loss"]
#                 cv_theta += model["theta"]
#             cv_ndcg, cv_fair_loss, cv_theta = cv_ndcg/n_splits, cv_fair_loss/n_splits, cv_theta/n_splits
#             if cv_ndcg > best_ndcg:
#                 best_gamma = gamma
#                 best_ndcg = cv_ndcg
#                 best_fair_loss = cv_fair_loss
#                 best_theta = cv_theta
#         print("ndcg: {}, fair_loss: {}, lamda: {}, gamma: {}".format(results["ndcg"], results["fair_loss"], lamda, gamma))
#         #theta, Q = trainAdversarialRanking(X_train, Y_train, lamda, mu, gamma)
#         #model = fair_adv_training(dr, args=args)
#         #results = fair_adv_testing(model, vdr, args=args)
#         args.gamma = best_gamma
#         model["theta"] = best_theta
#         results = testAdvarsarialRanking(vdr_x ,vdr_y , model, args=args)
#         #plt_data_pl[i] = [results["ndcg"], results["avg_group_asym_disparity"]]
#         plt_data_pl[i] = [results["ndcg"], results["fair_loss"]]
#         print("ndcg: {}, fair_loss: {}, lamda: {}, gamma: {}".format(results["ndcg"], results["fair_loss"], lamda, best_gamma))
 
end = time.time()
elapsed = end - start
print("time: ", elapsed)


