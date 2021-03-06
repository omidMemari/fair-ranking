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

filename = "german" # or adult


dr = YahooDataReader(None)
dr.data = pkl.load(open("GermanCredit/german_train_rank_3.pkl", "rb")) # (data_X, data_Y) 500*10*29, 500*10*1
vdr = YahooDataReader(None)
vdr.data = pkl.load(open("GermanCredit/german_test_rank_3.pkl","rb"))    # (data_X, data_Y) 100*10*29, 100*10*1


#dr = YahooDataReader(None)
#dr.data = pkl.load(open("adult/adult_train_rank.pkl", "rb")) # (data_X, data_Y) 500*10*109, 500*10*1
#vdr = YahooDataReader(None)
#vdr.data = pkl.load(open("adult/adult_test_rank.pkl","rb"))    # (data_X, data_Y) 100*10*109, 100*10*1


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace(conditional_model=True, gpu_id=None, progressbar=True, evaluate_interval=250, input_dim=29, #109  
                 eval_rank_limit=1000,
                fairness_version="asym_disparity", entropy_regularizer=0.0, save_checkpoints=False, num_cores=54,
                pooling='concat_avg', dropout=0.0, hidden_layer=8, summary_writing=False, 
                 group_fairness_version="asym_disparity",early_stopping=False, lr_scheduler=False, 
                 validation_deterministic=False, evalk=1000, reward_type="ndcg", baseline_type="value", 
                 use_baseline=True, entreg_decay=0.0, skip_zero_relevance=True, eval_temperature=1.0, optimizer="Adam",
               clamp=False)
torch.set_num_threads(args.num_cores)
args.progressbar = False 

args.group_feat_id = 3   # 3 for german, 5 for adult
args.sample_size =  10 #25
args.constraint = "demographic_parity" # "asym_disparity" # 


dr_x_orig = np.array(dr.data[0])
dr_y = np.array(dr.data[1])
vdr_x_orig = np.array(vdr.data[0])
vdr_y = np.array(vdr.data[1])

nc,nn,nf_orig = np.shape(dr_x_orig)
v_nc,nn,nf_orig = np.shape(vdr_x_orig)
dr_x = dr_x_orig #[[np.outer(np.transpose(dr_x_orig[i,j]), dr_x_orig[i,j]).flatten() for j in range(nn)] for i in range(nc)]
dr_x = np.array(dr_x)
vdr_x = vdr_x_orig #[[np.outer(np.transpose(vdr_x_orig[i,j]), vdr_x_orig[i,j]).flatten() for j in range(nn)] for i in range(v_nc)]
vdr_x = np.array(vdr_x)
nc,nn,nf = np.shape(dr_x)

print("np.shape(dr_x): ",np.shape(dr_x))

lamdas_list = [0.0, 1e6]#[0.0, 1e-1, 1, 2, 5, 7, 10, 50, 1e2, 250, 500, 1e3, 1e4, 1e6, 1e9]
gammas_list = [1e-2] #, 1, 10, 100]
mus_list = [1e0] #[1e-2, -1e-2, 1e-1, -1e-1, 1e0, 1e1, -1e1, 1e2, -1e2]
best_lamda = -1.0
best_ndcg = -1.0

n_splits = 5

def train_func(data):
    
    lamda, gamma, mu = data
    args.lambda_group_fairness = lamda
    args.gamma = gamma
    args.mu = mu
    # prepare cross validation
    k_fold = KFold(n_splits, True, 1)
    cv_matching_ndcg, cv_ndcg, cv_dp_fair_loss, cv_fair_loss = 0, 0, 0, 0
    cv_theta, best_theta = np.zeros(nf), np.zeros(nf)
    best_ndcg = -1
    for fold_idx, (train_set, val_set) in enumerate(k_fold.split(dr_x)):
        #print("fold_idx: ", fold_idx)
        #print("train_set: ", train_set)
        #print("test_set: ", val_set)
        model = trainAdversarialRanking(dr_x[train_set], dr_y[train_set], args=args)
        results = testAdvarsarialRanking(dr_x[val_set], dr_y[val_set], model, args=args)
        print("Train with Lambda={} and Gamma={} and mu={}: ndcg={}, fair_loss={}".format(lamda, gamma, mu, model["matching_ndcg"], model["avg_group_demographic_parity"]))
        print("Validation with Lambda={} and Gamma={} and mu={}: ndcg={}, fair_loss={}".format(lamda, gamma, mu, results["matching_ndcg"], results["avg_group_demographic_parity"]))
        cv_ndcg += results["ndcg"]##################model["ndcg"]
        cv_matching_ndcg += results["matching_ndcg"]
        cv_dp_fair_loss += results["avg_group_demographic_parity"]
        cv_fair_loss += results["avg_group_asym_disparity"]
        cv_theta += model["theta"]
        ###############################if results["ndcg"] > best_ndcg:
        if results["ndcg"] > best_ndcg:
            best_ndcg = results["ndcg"]################## result["ndcg"]
            best_theta = model["theta"]
    cv_matching_ndcg, cv_ndcg, cv_dp_fair_loss, cv_fair_loss, cv_theta = cv_matching_ndcg/n_splits, cv_ndcg/n_splits, cv_dp_fair_loss/n_splits, cv_fair_loss/n_splits, cv_theta/n_splits
    output = {
        "lambda": lamda,
        "gamma": gamma,
        "mu": mu,
        "theta": cv_theta,
        "ndcg": cv_ndcg,
        "matching_ndcg": cv_matching_ndcg,
        "avg_group_asym_disparity": cv_fair_loss,
        "avg_group_demographic_parity": cv_dp_fair_loss
    }
 
    return output
    #return lamda, gamma, mu, cv_matching_ndcg, cv_ndcg, cv_fair_loss, cv_theta##cv_ndcg, cv_fair_loss, cv_theta #best_theta

def test_func(data):

    lamda = data["lambda"]
    gamma = data["gamma"]
    mu =    data["mu"]
    theta = data["theta"]
    args.lambda_group_fairness = lamda
    args.gamma = gamma
    args.mu = mu
    model = {"theta": theta }
    results = testAdvarsarialRanking(vdr_x ,vdr_y , model, args=args)
    print("Test with Lambda={} and Gamma={}: ndcg={}, fair_loss={}".format(lamda, gamma, results["ndcg"], results["avg_group_demographic_parity"]))
    return results #lamda, results["rank_ndcg"], results["ndcg"], results["fair_loss"]

def parallel_runs(data_list):
    ls = []
    ndcg_crt = "matching_ndcg" #"ndcg" #"matching_ndcg"
    fair_crt = "avg_group_demographic_parity" #"avg_group_asym_disparity"
    pool = multiprocessing.Pool(processes=args.num_cores)
    #prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(train_func, data_list)
    #res = [result_list[i1] if result_list[i1]["lambda"]==result_list[i2]["lambda"] and result_list[i1][criteria]> result_list[i2][criteria]
    
    res = {key : [result_list[idx]
      for idx in range(len(result_list)) if result_list[idx]["lambda"]== key]
      for key in set([x["lambda"] for x in result_list])}
    print("res: ", res)
    for key, values in res.items():
        ls.append([(key,values[idx]) for idx in range(len(res[key])) if  values[idx][ndcg_crt]==max([x[ndcg_crt] for x in values])])
        ls.append([(key,values[idx]) for idx in range(len(res[key])) if  values[idx][fair_crt]==min([x[fair_crt] for x in values])])
            
    lls = np.array([ls[i][0][1] for i in range(len(ls))])
    print("lls: ", lls)
    #print("ls[0]: ", ls[0])
    #print("ls[0][0]: ", ls[0][0])
    #print("ls[0][0][1]: ", ls[0][0][1])
    #print("for lambda={} best gamma={} and mu={}".format(ls[0][0][1]["lambda"], ls[0][0][1]["gamma"], ls[0][0][1]["mu"]))
    #train_result = {"lambda":ls[0][0][0], "gamma":ls[0][0][1][0], "mu":ls[0][0][1][1], "rank_ndcg": ls[0][0][1][2], "ndcg": ls[0][0][1][3]}
    train_result = lls
    pool = multiprocessing.Pool(processes=args.num_cores)
    test_results = pool.map(test_func, lls)
    print(test_results)
    sorted_result = sorted(test_results, key=lambda x: x["lambda"])
    return train_result, sorted_result
    #return np.array(result_list)[:,:4]

#############################################################

def policy_parallel(data): # run policy code in parallel using multiple cores 
    
    lamda = data
    #args.group_feat_id = 3
    model_params_list = []
    args.lambda_group_fairness = lamda
    args.lambda_reward = 1.0
    args.lambda_ind_fairness = 0.0
    args.lr = 0.001
    args.epochs = 10
    args.progressbar = False
    args.weight_decay = 0.0
    #args.sample_size =  10 #25
    args.optimizer = "Adam"
    
    model = LinearModel(D=args.input_dim)
    model = on_policy_training(dr, vdr, model, args=args)
    if lamda == 0:
        results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                             deterministic=True, args=args, num_sample_per_query=20)
        print(results)
        
    else: # changed by me! for lamda == 0 there were 2 outputs 
        results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
                                 deterministic=False, args=args, num_sample_per_query=20)
        print(results)
        print("Lambda: ", lamda)
        model_params_list.append(model.w.weight.data.tolist()[0])
        print("Learnt model for lambda={} has model weights as {}".format(lamda, model_params_list[-1]))
    
    return lamda, results["ndcg"], results["avg_group_demographic_parity"], results["avg_group_asym_disparity"]


def policy_learning(): 
    
    #lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
    lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
    pool = multiprocessing.Pool(processes=args.num_cores)
    #prod_x=partial(prod_xy, y=10) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(policy_parallel, lambdas_list)
    print(result_list)
    sorted_result = sorted(result_list, key=lambda x: x[0])
    return sorted_result
    
##############################################################
def zehlike():
    args.lr = [0.001]
    args.lambda_reward = 1.0
    plt_data_z = []
    plt_data_z_dp = []
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
        plt_data_z_dp.append([results["ndcg"], results["avg_group_demographic_parity"]])
        ndcg_mat[i, 0], disparities_mat[i,0] = results["ndcg"], results["avg_group_demographic_parity"] #results["avg_group_asym_disparity"]
    return np.array(plt_data_z), np.array(plt_data_z_dp)
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
data_list = list(itertools.product(lamdas_list, gammas_list, mus_list))
train_result, adv_result = parallel_runs(data_list)
plt_data_adv = np.array([[adv_result[i]["ndcg"], adv_result[i]["avg_group_asym_disparity"]] for i in range(len(adv_result))])
plt_data_adv_dp = np.array([[adv_result[i]["ndcg"], adv_result[i]["avg_group_demographic_parity"]] for i in range(len(adv_result))])

plt_data_adv_matching = np.array([[adv_result[i]["matching_ndcg"], adv_result[i]["avg_group_asym_disparity"]] for i in range(len(adv_result))])
plt_data_adv_dp_matching = np.array([[adv_result[i]["matching_ndcg"], adv_result[i]["avg_group_demographic_parity"]] for i in range(len(adv_result))])

end_adv = time.time()
#################################################################################
policy_result = policy_learning()
plt_data_pl_dp = np.array([[policy_result[i][1], policy_result[i][2]] for i in range(len(policy_result))])
plt_data_pl = np.array([[policy_result[i][1], policy_result[i][3]] for i in range(len(policy_result))])
end_policy = time.time()
##################################################################################
plt_data_z, plt_data_z_dp = zehlike()
end_zehlike = time.time()
###################################################################################
#plt_data_adv =  np.array([[7.87678189e-01, 4.31883047e-04], [7.87694120e-01, 1.07887506e-04]])
#plt_data_pl = np.array([[0.93654663, 0.02231645], [0.92797832, 0.01746921], [0.84281633, 0.00131007]])

print("adv train result: ", train_result)
print("adv test result: ", adv_result)
print()
print("plt_data_adv: ", plt_data_adv)
print("plt_data_adv_matching: ", plt_data_adv_matching)
print("plt_data_pl: ", plt_data_pl)
print("plt_data_z: ", plt_data_z)
print()
print("plt_data_adv_dp: ", plt_data_adv_dp)
print("plt_data_adv_dp_matching: ", plt_data_adv_dp_matching)
print("plt_data_pl_dp: ", plt_data_pl_dp)
print("plt_data_z_dp: ", plt_data_z_dp)



#ndcg_vs_disparity_plot([plt_data_adv], ["Robust_Fair ($\lambda \in [0, 10^7]$)"], join=True, ranges=[[0.65, 0.95], [0.00, 0.040]], filename= "german_robust_tradeoff")

#ndcg_vs_disparity_plot([plt_data_adv, plt_data_pl], ["Robust_Fair ($\lambda \in [0, 10000]$)",
#                      "Policy_Ranking($\lambda \in [0,100]$ )"], join=True, ranges=[[0.70, 0.95], [0.00, 0.040]], filename= "german_robust_policy_tradeoff")

#ndcg_vs_disparity_plot([plt_data_adv, plt_data_pl,  plt_data_z], ["Robust_Fair ($\lambda \in [0, 10^4]$)",
#                      "Policy_Ranking ($\lambda \in [0,100]$ )", 
#                      "Zehlike ($\lambda \in [0, 10^6]$)"], join=True, ranges=[[0.60, 0.95], [0.00, 0.040]], filename="german_robust_policy_zehlike_tradeoff")

elapsed_adv = end_adv - start_adv
elapsed_policy = end_policy - end_adv
elapsed_zehlike = end_zehlike - end_policy
print("time for Robust_Fair: ", elapsed_adv)
print("time for Policy_Learning: ", elapsed_policy)
print("time for Zehlike: ", elapsed_zehlike)

with open("result.txt", "w") as f:
    print("filename: ", filename, file=f)
    print("group_feat_id: ", args.group_feat_id, file=f)
    print("sample_size: ", args.sample_size, file=f)
    print("k-fold: ", n_splits, file=f)
    print("lambdas_list: ", lamdas_list, file=f)
    print("gammas_list: ", gammas_list, file=f)
    print("mus_list: ", mus_list, file=f)
    print("adv test result: ", adv_result, file=f)
    print("plt_data_adv: ", plt_data_adv, file=f)
    print("plt_data_adv_matching: ", plt_data_adv_matching, file=f)
    print("plt_data_pl: ", plt_data_pl, file=f)
    print("plt_data_z: ", plt_data_z, file=f)
    print("plt_data_adv_dp: ", plt_data_adv_dp, file=f)
    print("plt_data_adv_dp_matching: ", plt_data_adv_dp_matching, file=f)
    print("plt_data_pl_dp: ", plt_data_pl_dp, file=f)
    print("plt_data_z_dp: ", plt_data_z_dp, file=f)
    print("time for Robust_Fair: ", elapsed_adv, file=f)
    print("time for Policy_Learning: ", elapsed_policy, file=f)
    print("time for Zehlike: ", elapsed_zehlike, file=f)

