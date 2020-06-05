import pickle
from create_shift import create_shift
import os
import prepare_data 
import pandas as pd
from fair_covariate_logloss import EOPP_fair_covariate_logloss_classifier, DP_fair_covariate_logloss_classifier
from experiments.baselines.FairLR.fair_logloss import EOPP_fair_logloss_classifier
import numpy as np 
import itertools 
from sklearn.model_selection import GridSearchCV, KFold
import pdb, sys
root = "experiments"
sample_path = os.path.join(root,"samples")
baselines_path = os.path.join(root,"baselines")

sample_record_filename_template = "{}_{:.1}_{:d}_{}_{}_{}" # dataset_split*10 _ n _sampling _ param1 _ param 2
#experiment_filename_template = "{}_{}_"

data2prepare = {'adult' : prepare_data.prepare_IBM_adult, 
                'compas' : prepare_data.prepare_compas, 
                'german' : prepare_data.prepare_german, 
                'law' : prepare_data.prepare_law,
                'drug' : prepare_data.prepare_drug,
                'arrhythmia' : prepare_data.prepare_arrhythmia}



samplings =   [#{'sampling' : 'att', 'param1' : .5, 'param2' : .5, 'split' : .5, 'n' : 10},
               {'sampling' : 'att', 'param1' : .5, 'param2' : .7, 'split' : .5, 'n' : 10},
               {'sampling' : 'att', 'param1' : .3, 'param2' : .7, 'split' : .5, 'n' : 10},
               {'sampling' : 'att', 'param1' : .7, 'param2' : .3, 'split' : .5, 'n' : 10},
               {'sampling' : 'pca', 'param1' : 1, 'param2' : 1, 'split' : .5, 'n' : 10},
               {'sampling' : 'pca', 'param1' : 2, 'param2' : 1, 'split' : .5, 'n' : 10},
               {'sampling' : 'pca', 'param1' : 1.5, 'param2' : 1, 'split' : .5, 'n' : 10}
               ]

best_c = { 'compas' : .001, 'adult' : .0001, 'german' : .01, 'law' : .0001, 'drug' : .001, 'arrhythmia' : .1}
# law : 1e-5
datasets = ['adult', 'compas', 'german', 'law', 'drug', 'arrhythmia']

baselines = { 'eoppCovariateLR' : EOPP_fair_covariate_logloss_classifier(trg_group_estimator = None, verbose = False),
              'eoppLR' : EOPP_fair_logloss_classifier(verbose=False) }

def fit_LR(sample,c):
    h = baselines['eoppCovariateLR']
    h.mu1 = 0
    h.C = c 
    h.trg_grp_marginal_matching = False
    h.fit(sample['X_src'],sample['Y_src'],sample['A_src'],np.ones_like(sample['ratio_src']),sample['X_trg'],sample['A_trg'],sample['ratio_trg'],mu_range = 0 )
    return h

def fit_CovariateLR(sample,c):
    h = baselines['eoppCovariateLR']
    h.mu1 = 0 
    h.C = c 
    h.trg_grp_marginal_matching = False
    h.fit(sample['X_src'],sample['Y_src'],sample['A_src'],sample['ratio_src'],sample['X_trg'],sample['A_trg'],sample['ratio_trg'],mu_range = 0 )
    return h

def fit_eoppLR(sample,c):
    h = baselines['eoppLR']
    h.C = c 
    h.fit(sample['X_src'],sample['Y_src'],sample['A_src'])
    return h

def fit_IW(sample,c):
    h = baselines['eoppCovariateLR']
    h.mu1 = 0 
    h.C = c 
    h.trg_grp_marginal_matching = False
    h.fit(sample['X_src'],sample['Y_src'],sample['A_src'],np.ones_like(sample['ratio_src']),sample['X_trg'],sample['A_trg'],np.ones_like(sample['ratio_trg']),mu_range = 0 ,IW_ratio_src= 1 / sample['ratio_src'])
    return h
    
def fit_eoppCovariateLR(sample,c):
    h = baselines['eoppCovariateLR']
    h.trg_grp_marginal_matching = True
    h.C = c 
    h.fit(sample['X_src'],sample['Y_src'],sample['A_src'],sample['ratio_src'],sample['X_trg'],sample['A_trg'],sample['ratio_trg'],mu_range = [-1,1])
    return h


h2fit = {'eoppCovariateLR' : fit_eoppCovariateLR,
         'LR' : fit_LR,
         'covariateLR': fit_eoppCovariateLR,
         'IW': fit_IW,
         'eoppLR': fit_eoppLR}



def store_object(obj,path, name):
    filepath = os.path.join(path,name)
    with open(filepath, 'wb') as file:
        pickle.dump(obj,file)
    print("Record wrote to {}".format(filepath))
    
def load_object(path,name):
    with open(os.path.join(path,name), 'rb') as file:
        return pickle.load(file)
'''
    param1, param2 :        either mean_a, std_b or p_s, p_t    
'''    
def create_sample_record( data, n = 1, split = .5, sampling = 'att', param1 = .5, param2 = .5):
    record = dict.fromkeys(range(n))
    for i in range(n):
        tr_idx,ts_idx,ratios = create_shift(data, src_split = split, flag = sampling, mean_a=param1, std_b = param2, p_s=param1, p_t=param2)
        record[i] = {'tr_idx' : tr_idx, 'ts_idx' : ts_idx, 'ratios' : ratios}
    
    return record

def make_sample_record_filename(dataset, **kwargs):
    return sample_record_filename_template.format(dataset, kwargs['split'], kwargs['n'], kwargs['sampling'], kwargs['param1'], kwargs['param2']).replace('.','-')

def load_data_xay(dataset):
    dataA,dataY,dataX,_ = data2prepare[dataset]()
    #dataX = pd.concat([dataA,dataX],axis = 1).values
    return dataX, dataA, dataY

def create_store_sample_for_dataset(dataset, **kwargs):
    dataX, dataA, _ = load_data_xay(dataset)
    data = pd.concat([dataA,dataX],axis = 1).values
    rec = create_sample_record(data,**kwargs)
    name = make_sample_record_filename(dataset=dataset,**kwargs)
    file_dir = os.path.join(sample_path,dataset)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    store_object(rec,file_dir, name)
    return rec

def prepare_sample(**kwargs):
    dataset = kwargs['dataset']
    sample_name = make_sample_record_filename(**kwargs)
    sample_record = load_object(os.path.join(sample_path, dataset), sample_name)
    i = kwargs['i'] # sample index
    tr_idx, ts_idx, ratios = sample_record[i]['tr_idx'], sample_record[i]['ts_idx'], sample_record[i]['ratios']
    dataX, dataA, dataY = load_data_xay(dataset)
    tr_X, tr_ratio = dataX.iloc[tr_idx,:], ratios[tr_idx]
    ts_X, ts_ratio = dataX.iloc[ts_idx,:], ratios[ts_idx]
    tr_A, tr_Y = dataA.iloc[tr_idx].squeeze(), dataY.iloc[tr_idx].squeeze()
    ts_A, ts_Y = dataA.iloc[ts_idx].squeeze(), dataY.iloc[ts_idx].squeeze()

    print(tr_X.shape, ts_X.shape)    
    dataset = dict( X_src = tr_X.values, A_src = tr_A.values, Y_src = tr_Y.values, ratio_src = tr_ratio, X_trg = ts_X.values, A_trg = ts_A.values, Y_trg = ts_Y.values, ratio_trg = ts_ratio)
    #tr_data = quadratic_features(tr_data)
    #ts_data = quadratic_features(ts_data)
    #tr_A, tr_X = pd.DataFrame(tr_data[:,0]).squeeze(), pd.DataFrame(tr_data[:,:])
    #ts_A, ts_X = pd.DataFrame(ts_data[:,0]).squeeze(), pd.DataFrame(ts_data[:,:])
    #tr_Y, ts_Y = dataY.reindex(tr_idx), dataY.reindex(ts_idx)
    return dataset

def record_classifier(h, X, Y, A, ratio):
    db = { }
    if isinstance(h, EOPP_fair_covariate_logloss_classifier):
        db['zo'] = h.expected_error(X, Y, A, ratio)
        db['err'] = 1 - h.score(X,Y,A,ratio)
        db['violation'] = h.fairness_violation(X,Y,A,ratio)
        db['logloss'] = h.expected_logloss(X,Y,A,ratio)
        db['q_violation'] = h.q_fairness_violation(X,A,ratio)
        db['objective']= h.q_objective(X,A,ratio)
        db['mu']= h.mu1
        db['q_marg_err_1']= h.q_marginal_grp_estimation_error(X,A,ratio,a = 1)
        db['q_marg_err_0']= h.q_marginal_grp_estimation_error(X,A,ratio,a = 0)
        pr1, pr0 = h.positive_rate(X,A,ratio)
        db['pos_rate_gr1']= pr1
        db['pos_rate_gr0']= pr0
        pr1, pr0 = h.q_positive_rate(X,A,ratio)
        db['q_pos_rate_gr1']= pr1
        db['q_pos_rate_gr0']= pr0
        db['q_fairness_penalty']= h.q_fairness_penalty(X,A,ratio)
        db['lambda1']= h.lambdas[0]
        db['lambda0']= h.lambdas[1]
        print("----------------------------  mu %.2f , C %.4f----------------------------------" % (h.mu1, h.C))
        print("Test  - predict_err : {:.3f} \t logloss : {:.3f} \t fair_violation : {:.3f} \t q_fair_violation : {:.3f}".format(db['err'], db['logloss'],db['violation'],db['q_violation']))
    else:
        db['zo'] = h.expected_error(X, Y, A, ratio)
        db['err'] = 1 - h.score(X,Y,A,ratio)
        db['violation'] = h.fairness_violation(X,Y,A,ratio)
        db['logloss'] = h.expected_logloss(X,Y,A,ratio)
        print("----------------------------  C %.4f----------------------------------" % (h.C))
        print("Test  - predict_err : {:.3f} \t logloss : {:.3f} \t fair_violation : {:.3f} ".format(db['err'], db['logloss'],db['violation']))
    return db

def make_experiment_filename(dataset,classifier, **kwargs):
    sample_name = make_sample_record_filename(dataset,**kwargs)
    return "{}_{}".format(classifier,sample_name)

    
def run_experiment(dataset, classifier, **kwargs):
    experiment_db = {}
    for i in range(kwargs['n']):
        print("------------------------------- {} sample {:d} ---------------------------------".format(dataset,i))
        sample = prepare_sample(dataset = dataset, i = i, **kwargs)
        h = h2fit[classifier](sample,best_c[dataset])
        db = record_classifier(h,sample['X_trg'],sample['Y_trg'],sample['A_trg'],sample['ratio_trg'])
        experiment_db[i] = db
    experiment_filename = make_experiment_filename(dataset,classifier,**kwargs)
    file_dir = os.path.join(baselines_path,classifier)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    store_object(experiment_db,file_dir,experiment_filename)
    #experiment_db2 = load_object(file_dir,experiment_filename)
    #assert(experiment_db2 == experiment_db)

def create_all_samples():
    for (dataset,sampling) in itertools.product(datasets,samplings):
        create_store_sample_for_dataset(dataset = dataset, **sampling)

def cross_validate_c(dataset):
    hyperparams = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
    kf = KFold(n_splits=3)
    dataX, dataA, dataY = load_data_xay(dataset)
    score = {}
    for c in hyperparams:
        sum_score = 0
        for train_idx, test_idx in kf.split(dataY.values):
            print(train_idx)
            X_tr, X_ts = dataX.values[train_idx,:], dataX.values[test_idx,:]
            y_tr, y_ts = dataY.values[train_idx], dataY.values[test_idx]
            a_tr, a_ts = dataA.values[train_idx], dataA.values[test_idx]
            #h = EOPP_fair_covariate_logloss_classifier(trg_group_estimator= None,verbose=False)
            h = baselines['eoppCovariateLR']
            #h.lambdas = np.zeros((2,1)   )
            #pdb.set_trace()
            h.trg_grp_marginal_matching = False
            h.C = c
            h.fit(X_tr,y_tr,a_tr,np.ones_like(a_tr),X_ts,a_ts,np.ones_like(a_ts),mu_range=0)
            print(h.expected_logloss(X_ts,y_ts,a_ts,np.ones_like(a_ts)))
            sum_score += h.expected_logloss(X_ts,y_ts,a_ts,np.ones_like(a_ts))
        score[c] = sum_score / 5
        sum_score = 0
    print(score)
    key_min = min(score.keys(), key=(lambda k: score[k]))
    return key_min

if __name__ == '__main__':

    sampling1 = dict(split = .5 , n = 1, sampling = 'feature', param1 = .5, param2= .5)
    #datasets = datasets[1:]
    for (dataset,sampling) in itertools.product(datasets,samplings):
        run_experiment(dataset,sys.argv[1],**sampling)
    #print(cross_validate_c(sys.argv[1]))