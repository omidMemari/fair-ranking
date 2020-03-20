import pandas as pd
import numpy as np
from YahooDataReader import YahooDataReader
from train import trainAdversarialRanking
from test import testAdvarsarialRanking

import pickle as pkl
train = YahooDataReader(None)
train.data = pkl.load(open("GermanCredit/german_train_rank_3.pkl", "rb")) # (data_X, data_Y) 500*25*29, 500*25*1
test = YahooDataReader(None)
test.data = pkl.load(open("GermanCredit/german_test_rank_3.pkl","rb"))    # (data_X, data_Y) 100*25*29, 100*25*1
print(np.array(test.data[1]).shape)


X_train = train.data[0][:20] #train.data[0][:20]
Y_train = train.data[1][:20] #train.data[1][:20]
X_test  = test.data[0][:20]  #test.data[0][:20]
Y_test  = test.data[1][:20]  #test.data[1][:20]


#df = pd.DataFrame(list(train.data))
#print(list(df.columns.values))

lambdas_list = [1e-2, 1e-1]#[1e-3, 1e-2, 1e-1, 1e0, 1e1] #lambdas_list = [0.0, 0.1, 1.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0]
gammas_list = [1000, 2000]#[1e-2, 1e-1, 1e0, 1e1, 1e2]
best_lamda = -1.0
best_ndcg = -1.0
kf = 1 #5 # k-fold
mu = 1e-2 # No need to change?


result = []
#plt_data_pl = np.zeros((len(lambdas_list)+1, 2))
for j, gamma in enumerate(gammas_list):
    for i, lamda in enumerate(lambdas_list):
        cv_util = [] # utility for different cross validations
    
        #model = LinearModel(D=args.input_dim)

        #model = on_policy_training(dr, vdr, model, args=args)
          
        theta, Q = trainAdversarialRanking(X_train, Y_train, lamda, mu, gamma)
        ndcg, fair_loss = testAdvarsarialRanking(X_train, Y_train, lamda, mu, gamma, theta)
        #util = sum(cv_util) / len(cv_util)
        result.append([ndcg, fair_loss, lamda, gamma])
        print("ndcg: {}, fair_loss: {}, lamda: {}, gamma: {}".format(ndcg, fair_loss, lamda, gamma))
        if ndcg > best_ndcg:
            best_lamda = lamda
            best_gamma = gamma
            best_ndcg = ndcg
print("result: ", result)
print("best:: Utility: {}, lamda: {}, gamma: {}".format(best_ndcg, best_lamda, best_gamma))
    
    # if i == 0:
    #     results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
    #                         deterministic=True, args=args, num_sample_per_query=20)
    #     print(results)
    #     plt_data_pl[0] = [results["ndcg"], results["avg_group_asym_disparity"]]
    # results = evaluate_model(model, vdr, fairness_evaluation=False, group_fairness_evaluation=True, 
    #                         deterministic=False, args=args, num_sample_per_query=20)
    # print(results)
    # model_params_list.append(model.w.weight.data.tolist()[0])
    # print("Learnt model for lambda={} has model weights as {}".format(lgroup, model_params_list[-1]))
    # plt_data_pl[i+1] = [results["ndcg"], results["avg_group_asym_disparity"]]
