from __future__ import print_function
from scipy.io import arff
from prepare_data import prepare_compas,prepare_IBM_adult, prepare_law, prepare_german

import functools
import numpy as np
import pandas as pd
import sys
from fair_covariate_logloss import EOPP_fair_covariate_logloss_classifier, DP_fair_covariate_logloss_classifier #, EODD_fair_logloss_classifier
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

from create_shift import create_shift
import pdb 

def compute_error(Yhat,proba,Y):
    err = 1 - np.sum(Yhat == Y) / Y.shape[0] 
    exp_zeroone = np.mean(np.where(Y == 1 , 1 - proba, proba))
    return err, exp_zeroone

def build_trg_grp_true_estimator(A,Y):
    p1 = np.mean(np.logical_and(A == 1 , Y == 1))
    p2 = np.mean(np.logical_and(A == 0 , Y == 1))
    estimator = lambda a : p1 if (a == 1) else p2 if (a == 0) else 0
    return estimator

def build_trg_grp_estimator(X_src,A_src,Y_src,src_ratio, X_trg,A_trg,trg_ratio,C):
   #estimator = build_trg_grp_true_estimator(A_src,Y_src) # not used
   estimator = lambda a : 1 # not used
   h = EOPP_fair_covariate_logloss_classifier(tol=1e-5,max_iter=200, trg_group_estimator= estimator, C=C, random_initialization=False, verbose=False, trg_grp_marginal_matching=False)
   h.fit(X_src,Y_src,A_src, src_ratio, X_trg, A_trg, trg_ratio, mu_range=[0]) # build a covaritae shift model with ignored fairness
   p = h.predict_proba(X_trg,A_trg, trg_ratio) # A is ignored as attribute but is included in X
   p1 = np.dot(p, A_trg.astype('int')) / A_trg.shape[0]
   p0 = np.dot(p, 1- A_trg.astype('int')) / A_trg.shape[0]
   #p1 = np.mean(np.logical_and(A_trg == 1 , Y == 1))
   #p0 = np.mean(np.logical_and(A_trg == 0 , Y == 1))
   print("p1 : {:.4f}, p0 : {:.4f}".format(p1,p0))
   #pdb.set_trace()
   estimator = lambda a : p1 if (a == 1) else p0 if (a == 0) else 0
   return estimator

class experiment_logger():
    def __init__(self):
        self.db = {'zo' : [], 
                   'err' : [],
                   'violation' : [],
                   'logloss' : [],
                   'q_violation' : [],
                   'obj' : [],
                   'mu' : [],
                   'q_marg_err_1' : [],
                   'q_marg_err_0' : [],
                   'pos_rate_gr1': [],
                   'pos_rate_gr0': [],
                   'q_pos_rate_gr1': [],
                   'q_pos_rate_gr0': [],
                   'q_fairness_penalty' : [],
                   'lambda1' : [],
                   'lambda0' : []
                   }

    def record(self,h, X, Y, A, ratio):
        self.db['zo'].append(h.expected_error(X, Y, A, ratio))
        self.db['err'].append(1 - h.score(X,Y,A,ratio))
        self.db['violation'].append(h.fairness_violation(X,Y,A,ratio))
        self.db['logloss'].append(h.expected_logloss(X,Y,A,ratio))
        self.db['q_violation'].append(h.q_fairness_violation(X,A,ratio))
        self.db['obj'].append(h.q_objective(X,A,ratio))
        self.db['mu'].append(h.mu1)
        self.db['q_marg_err_1'].append(h.q_marginal_grp_estimation_error(X,A,ratio,a = 1))
        self.db['q_marg_err_0'].append(h.q_marginal_grp_estimation_error(X,A,ratio,a = 0))
        pr1, pr0 = h.positive_rate(X,A,ratio)
        self.db['pos_rate_gr1'].append(pr1)
        self.db['pos_rate_gr0'].append(pr0)
        pr1, pr0 = h.q_positive_rate(X,A,ratio)
        self.db['q_pos_rate_gr1'].append(pr1)
        self.db['q_pos_rate_gr0'].append(pr0)
        self.db['q_fairness_penalty'].append(h.q_fairness_penalty(X,A,ratio))
        self.db['lambda1'].append(h.lambdas[0])
        self.db['lambda0'].append(h.lambdas[1])

        


    def draw_scatter_plot(self,x_param, y_param,x_label, y_label, ax, label = 'mu', marker = 'o', c = None):
        if c is None:
            c = np.linspace(0,1,len(self.db[x_param]))
        ax.scatter(self.db[x_param],self.db[y_param], label=label, marker = marker, c = c) 
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        for x,y,mu in zip(self.db[x_param],self.db[y_param],self.db['mu']):
            lbl = "{:.2f}".format(mu)
            ax.annotate(lbl,(x,y),textcoords="offset points",xytext=(0,-10),ha="center")
        ax.legend()

def quadratic_features(X):
    (n,m) = X.shape
    bi_features = np.zeros((n, int(m * (m + 1) / 2)))
    l = 0
    for i in range(m):
        for j in range(i,m):
            bi_features[:,l] = X[:,i] * X[:,j]
            l += 1
    return np.hstack((X,bi_features))

def test_single_dataset_range_mu(dataset, dataX, dataA, dataY):
    
    C = .0005
    sampling = 'PCA'
    mean_a, std_b = 1, 1
    p_s, p_t = .4, .7
    ######### covariate shift
    #dataX = dataX.iloc[:,:100]
    data = pd.concat([dataA,dataX],axis = 1).values
    #tr_data, ts_data, tr_ratios, ts_ratios, tr_trg_prob, ts_trg_prob = create_shift(data.values, flag = 'feature')
    print(data.shape)
    tr_idx,ts_idx,ratios = create_shift(data, src_split = .3, flag = sampling, mean_a=mean_a, std_b = std_b, p_s=p_s, p_t=p_t)
    tr_data, tr_ratio = data[tr_idx,:], ratios[tr_idx]
    ts_data, ts_ratio = data[ts_idx,:], ratios[ts_idx]
    #tr_data = quadratic_features(tr_data)
    #ts_data = quadratic_features(ts_data)
    tr_A, tr_X = pd.DataFrame(tr_data[:,0]).squeeze(), pd.DataFrame(tr_data[:,:])
    ts_A, ts_X = pd.DataFrame(ts_data[:,0]).squeeze(), pd.DataFrame(ts_data[:,:])
    tr_Y, ts_Y = dataY.reindex(tr_idx), dataY.reindex(ts_idx)
    print(tr_X.shape)
    #tr_X =  pd.concat([tr_X, tr_X ** 2], axis=1) 
    #ts_X =  pd.concat([ts_X, ts_X ** 2], axis=1)
    # Comment out to not include A in features
    #tr_X = pd.concat([tr_X, tr_A], axis=1) 	
    #ts_X = pd.concat([ts_X, ts_A], axis=1)
    # ---------
    #pdb.set_trace()
    #for c in list(tr_X.columns):
    #    if tr_X[c].min() < 0 or tr_X[c].max() > 1:
    #        mu = tr_X[c].mean()
    #        s = tr_X[c].std(ddof=0)
    #        tr_X.loc[:,c] = (tr_X[c] - mu) / s
            #ts_X.loc[:,c] = (ts_X[c] - mu) / s
    #for c in list(ts_X.columns):
    #    if ts_X[c].min() < 0 or ts_X[c].max() > 1:
    #        mu = ts_X[c].mean()
    #        s = ts_X[c].std(ddof=0)
    #        ts_X.loc[:,c] = (ts_X[c] - mu) / s
    
    ######### i i d
    '''
    order = perm[2,:]
    tr_sz = int(np.floor(.5 * dataX.shape[0]))
    tr_idx = order[:tr_sz]
    ts_idx = order[tr_sz:]
    tr_X = dataX.reindex(tr_idx)
    ts_X = dataX.reindex(ts_idx)
    
    tr_A = dataA.reindex(tr_X.index)
    ts_A = dataA.reindex(ts_X.index)
    tr_Y = dataY.reindex(tr_X.index)
    ts_Y = dataY.reindex(ts_X.index)
    
    # Comment out to not include A in features
    tr_X = pd.concat([tr_X, tr_A], axis=1) 	
    ts_X = pd.concat([ts_X, ts_A], axis=1)
    # ---------

    for c in list(tr_X.columns):
        if tr_X[c].min() < 0 or tr_X[c].max() > 1:
            mu = tr_X[c].mean()
            s = tr_X[c].std(ddof=0)
            tr_X.loc[:,c] = (tr_X[c] - mu) / s
            ts_X.loc[:,c] = (ts_X[c] - mu) / s
    tr_ratio, ts_ratio = np.ones_like(tr_A), np.ones_like(ts_A)
    '''        

    trg_est0 = build_trg_grp_true_estimator(ts_A.values,ts_Y.values)
    trg_est = build_trg_grp_estimator(tr_X.values, tr_A.values, tr_Y.values, tr_ratio, ts_X.values, ts_A.values,ts_ratio,C)
    print(" p1: true {:.4f} vs. q-est: {:.4f} - p0 : true: {:.4f} vs. q-est: {:.4f}".format(trg_est0(1),trg_est(1), trg_est0(0),trg_est(0) ))

    trg_est = None 
    h = EOPP_fair_covariate_logloss_classifier(tol=1e-5,max_iter=1000,trg_group_estimator= trg_est, C=C, random_initialization=False, verbose=False)
    h.max_epoch = 2
    h.trg_grp_marginal_matching = False
    h.fit(tr_X.values,tr_Y.values,tr_A.values, tr_ratio, ts_X.values, ts_A.values, ts_ratio, 0)
    
    '''
    mu = -2
    theta1 = h.theta
    p1,q1 = h.compute_p_and_q(theta1,-2,ts_X.values,ts_A.values,ts_ratio)   
    v1 = h.q_fairness_violation(ts_X.values, ts_A.values, ts_ratio)
    print("mu {:.3f} : v1 : {:.5f}".format(mu,v1))
    while True:
        mu += .01 
        h.fit(tr_X.values,tr_Y.values,tr_A.values, tr_ratio, ts_X.values, ts_A.values, ts_ratio, mu_range=mu)
        v2 = h.q_fairness_violation(ts_X.values, ts_A.values, ts_ratio)
        print("mu {:.3f} : v2 : {:.5f}".format(mu,v2))
        if abs(v2 - v1) > 1e-4:
            continue
        theta2 = h.theta
        p2,q2 = h.compute_p_and_q(theta1,-2,ts_X.values,ts_A.values,ts_ratio)
        pdb.set_trace()   
    '''
    exp_rec1 = experiment_logger()
    for mu in np.linspace(-.1,.1,9):#[-.1,.05,0,.05,.1] : #, [-.2,.2]]:
        h.mu1 = mu

        exp_zo_tr = h.expected_error(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        exp_zo_ts = h.expected_error(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        err_tr = 1 - h.score(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        err_ts = 1 - h.score(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        violation_tr = h.fairness_violation(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        violation_ts = h.fairness_violation(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        logloss_tr = h.expected_logloss(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        logloss_ts = h.expected_logloss(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        q_violation_tr = h.q_fairness_violation(tr_X.values, tr_A.values, tr_ratio)
        q_violation_ts = h.q_fairness_violation(ts_X.values, ts_A.values, ts_ratio)
        print("---------------------------- Adult mu %.2f ----------------------------------" % (h.mu1))
        print("Train - predict_err : {:.3f} \t expected_logloss : {:.3f} \t fair_violation : {:.3f} ".format(err_tr, logloss_tr,violation_tr))
        print("Test  - predict_err : {:.3f} \t expected_logloss : {:.3f} \t fair_violation : {:.3f} ".format(err_ts, logloss_ts,violation_ts))
        print("mu = {:.4f}".format(h.mu1))

        exp_rec1.record(h,ts_X.values, ts_Y.values, ts_A.values, ts_ratio)

    exp_rec2 = experiment_logger()

    #ts_ratio = np.ones_like(ts_ratio)
    h.trg_grp_marginal_matching = True    
    for mu in [[-.1,.1],0]:# np.linspace(-.3,.3,11):#[-.1,.05,0,.05,.1] : #, [-.2,.2]]:
        
        h.fit(tr_X.values,tr_Y.values,tr_A.values, tr_ratio, ts_X.values, ts_A.values, ts_ratio, mu_range = mu, IW_ratio_src= None)
        exp_zo_tr = h.expected_error(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        exp_zo_ts = h.expected_error(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        err_tr = 1 - h.score(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        err_ts = 1 - h.score(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        violation_tr = h.fairness_violation(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        violation_ts = h.fairness_violation(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        logloss_tr = h.expected_logloss(tr_X.values, tr_Y.values, tr_A.values, tr_ratio)
        logloss_ts = h.expected_logloss(ts_X.values, ts_Y.values, ts_A.values, ts_ratio)
        q_violation_tr = h.q_fairness_violation(tr_X.values, tr_A.values, tr_ratio)
        q_violation_ts = h.q_fairness_violation(ts_X.values, ts_A.values, ts_ratio)
        print("---------------------------- Adult mu %.2f ----------------------------------" % (h.mu1))
        print("Train - predict_err : {:.3f} \t expected_logloss : {:.3f} \t fair_violation : {:.3f} ".format(err_tr, logloss_tr,violation_tr))
        print("Test  - predict_err : {:.3f} \t expected_logloss : {:.3f} \t fair_violation : {:.3f} ".format(err_ts, logloss_ts,violation_ts))
        print("mu = {:.4f}".format(h.mu1))
        
        exp_rec2.record(h,ts_X.values, ts_Y.values, ts_A.values, ts_ratio)

    #ax = plt.axes()
    #ax.set_facecolor('#E6E6E6')
    font_options={'family' : 'sans-serif','size' : '12'}
    plt.rc('font', **font_options)
    plt.rcParams['grid.color'] = 'w'
    plt.rcParams['grid.linestyle'] = 'solid'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['figure.facecolor'] = '#E6E6E6'
    
    fig, axs = plt.subplots(1,4, constrained_layout = True)
    if sampling is 'feature':
        fig.suptitle('sampling: feature p_s: {:.2f}, p_t: {:.2f}'.format(p_s, p_t))
    elif sampling is 'PCA':
        fig.suptitle('sampling: PCA mean_a: {:.2f}, std_b: {:.2f}'.format(mean_a, std_b))

    axs[0].grid(color='w', linestyle='solid')
    axs[0].set_facecolor('#E6E6E6')
    exp_rec2.draw_scatter_plot('q_violation','q_fairness_penalty','EOPP q_violation', 'q-fairness_penalty',axs[0])
    exp_rec1.draw_scatter_plot('q_violation','q_fairness_penalty','EOPP q_violation','q-fairness_penalty', axs[0], label = 'theta(mu=0)', marker = 'v', c = 'red')
    axs[0].set_title(dataset)

    axs[1].grid(color='w', linestyle='solid')
    axs[1].set_facecolor('#E6E6E6')
    exp_rec2.draw_scatter_plot('violation','logloss','EOPP violation','logloss', axs[1])
    exp_rec1.draw_scatter_plot('violation','logloss','EOPP violation','logloss', axs[1], label = 'theta(mu=0)', marker = 'v', c = 'red')
    axs[1].set_title(dataset)

    axs[2].grid(color='w', linestyle='solid')
    axs[2].set_facecolor('#E6E6E6')
    #exp_rec2.draw_scatter_plot('q_marg_err_1','q_marg_err_0','Q_marg_err_grp_1','Q_marg_err_grp_0', axs[2])
    exp_rec2.draw_scatter_plot('q_pos_rate_gr1','q_pos_rate_gr0','q_pos_rate_gr1','q_pos_rate_gr0', axs[2])

    axs[3].grid(color='w', linestyle='solid')
    axs[3].set_facecolor('#E6E6E6')
    #exp_rec2.draw_scatter_plot('q_marg_err_1','q_marg_err_0','Q_marg_err_grp_1','Q_marg_err_grp_0', axs[2])
    exp_rec2.draw_scatter_plot('lambda1','lambda0','lambda_gr1','lambda_gr0', axs[3]) 

    plt.show()     
    
if __name__ == '__main__':
    dataset = ""
    if sys.argv[1] == 'adult':
        dataA,dataY,dataX,perm = prepare_IBM_adult()
        dataset = 'adult'
    elif sys.argv[1] == 'compas':
        dataA,dataY,dataX,perm = prepare_compas()
        dataset = 'compas'
    elif sys.argv[1] == 'law':
        dataA,dataY,dataX,perm = prepare_law()
        dataset = 'law'
    elif sys.argv[1] == 'german':
        dataA,dataY,dataX,perm = prepare_german()
        dataset = 'german'
    
    else:
        raise ValueError('Invalid first arg')
    test_single_dataset_range_mu(dataset, dataX,dataA,dataY)
    
else:
    dataset = ""
    if sys.argv[1] == 'adult':
        dataA,dataY,dataX,perm = prepare_IBM_adult()
        dataset = 'adult'
    elif sys.argv[1] == 'compas':
        dataA,dataY,dataX,perm = prepare_compas()
        dataset = 'compas'
    elif sys.argv[1] == 'law':
        dataA,dataY,dataX,perm = prepare_law()
        dataset = 'law'
    else:
        raise ValueError('Invalid first arg')
    C = .005
    mu = 0
    criteria = sys.argv[2]
    #if criteria == 'dp':
    #    pass
        #h = DP_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)
    #elif criteria == 'eqopp':
        #h = EOPP_fair_covariate_logloss_classifier(mu=mu,C=C, random_initialization=True, verbose=False)
    #    pass
    #elif criteria == 'eqodd':
    #    pass
        #h = EODD_fair_logloss_classifier(C=C, random_initialization=True, verbose=False)    
    #else:
    #    raise ValueError('Invalid second arg')
    filename_tr = "results/fair_cov_{}_{:.3f}_{}_{}_tr.csv".format(dataset,C,mu,criteria)
    filename_ts = "results/fair_cov_{}_{:.3f}_{}_{}_ts.csv".format(dataset,C,mu,criteria)
    
    outfile_tr = open(filename_tr,"w")
    outfile_ts = open(filename_ts,"w")

    for r in range(20):
        order = perm[r,:]
        tr_sz = int(np.floor(.7 * dataX.shape[0]))
        tr_idx = order[:tr_sz]
        ts_idx = order[tr_sz:]
        tr_X = dataX.reindex(tr_idx)
        ts_X = dataX.reindex(ts_idx)
        
        tr_A = dataA.reindex(tr_X.index)
        ts_A = dataA.reindex(ts_X.index)
        tr_Y = dataY.reindex(tr_X.index)
        ts_Y = dataY.reindex(ts_X.index)
        
        # Comment out to not include A in features
        tr_X = pd.concat([tr_X, tr_A], axis=1) 	
        ts_X = pd.concat([ts_X, ts_A], axis=1)
        # ---------

        for c in list(tr_X.columns):
            if tr_X[c].min() < 0 or tr_X[c].max() > 1:
                mu = tr_X[c].mean()
                s = tr_X[c].std(ddof=0)
                tr_X.loc[:,c] = (tr_X[c] - mu) / s
                ts_X.loc[:,c] = (ts_X[c] - mu) / s
        
        trg_est = build_trg_grp_estimator(ts_X,ts_A,ts_Y)
        h = EOPP_fair_covariate_logloss_classifier(mu=0,trg_group_estimator= trg_est, C=C, random_initialization=False, verbose=True)
        h.fit(tr_X.values,tr_Y.values,tr_A.values)
        exp_zo_tr = h.expected_error(tr_X.values, tr_Y.values, tr_A.values)
        exp_zo_ts = h.expected_error(ts_X.values, ts_Y.values, ts_A.values)
        err_tr = 1 - h.score(tr_X.values, tr_Y.values, tr_A.values)
        err_ts = 1 - h.score(ts_X.values, ts_Y.values, ts_A.values)
        violation_tr = h.fairness_violation(tr_X.values, tr_Y.values, tr_A.values)
        violation_ts = h.fairness_violation(ts_X.values, ts_Y.values, ts_A.values)

        print("---------------------------- Random Split %d ----------------------------------" % (r + 1))
        print("Train - predict_err : {:.3f} \t expected_err : {:.3f} \t fair_violation : {:.3f} ".format(err_tr, exp_zo_tr,violation_tr))
        print("Test  - predict_err : {:.3f} \t expected_err : {:.3f} \t fair_violation : {:.3f} ".format(err_ts, exp_zo_ts,violation_ts))
        print("")

        #outfile_ts.write("{:.4f},{:.4f},{:.4f}\n".format(exp_zo_ts,err_ts, violation_ts))
        #outfile_tr.write("{:.4f},{:.4f},{:.4f}\n".format(exp_zo_tr,err_tr, violation_tr))
        
    outfile_tr.close()
    outfile_ts.close()