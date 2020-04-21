import numpy as np
#import matlab.engine
import scipy.io
import math
from scipy import optimize
from numpy import linalg
from birkhoff import birkhoff_von_neumann_decomposition as decomp
from projectBistochasticADMM import projectBistochasticADMM as Project
from baselines import *



last_P = []

def ranking_q_object(q_init, X, gamma, mu, lambda_group_fairness, group_feat_id, theta):
    
    global last_P
    #print("test.py: ranking_q_object")
    nc,nn,nf = np.shape(X)
    X = np.array(X)
    fc = fairnessConstraint(X, group_feat_id)
    
    
    if lambda_group_fairness == 0.0: # No Fairness Constraints
        q = np.reshape(q_init, (-1, nn)) # (500*25,) --> (500, 25)
        alpha = np.zeros(nc)
        P = minP(q,fc ,alpha ,vvector(nn) ,mu) # P: 500*25*25 ###########################
        Pv = np.matmul(P, vvector(nn)) # Pv: 500*25, P: 500*25*25, v:25*1
        qPv = np.dot(q.flatten(), Pv.flatten())
        print("qPv:", qPv/nc)
        theta = np.squeeze(theta)
        PSI = np.squeeze(np.matmul(X,theta)) 
        print("<q,PSI>: ", sum([np.dot(q[i], PSI[i]) for i in range(nc)])/nc)
        obj = qPv - sum([np.dot(q[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten())
        #fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        #obj = obj + np.dot(alpha, fPv)
        #print("alpha * fPv: ", np.dot(alpha, fPv)/nc)
        print("mu/2*||P||: ",(mu/2) *np.dot(P.flatten(), P.flatten())/nc)
        print("mu/2*||q||: ",(mu/2) *np.dot(q.flatten(), q.flatten())/nc)
        print("gamma/2*||theta||: ",(gamma/2) *np.dot(theta, theta))
        #print("alpha: ", alpha)
        #print("lambda/2*||alpha||: ",(lambda_group_fairness/2) *np.dot(alpha, alpha))
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta) 
        gr_q = np.array((Pv - PSI + mu*q)/nc) ####################/nc I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
        g = gr_q.flatten()
        print("obj: ", obj)

    elif lambda_group_fairness > 0.0:
        
        q_alpha = np.reshape(q_init, (-1, nn+1)) # (500*26,) --> (500, 26)
        q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
        alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
        P = minP(q,fc ,alpha ,vvector(nn) ,mu) # P: 500*25*25 ########################
        Pv = np.matmul(P, vvector(nn)) # Pv: 500*25, P: 500*25*25, v:25*1
        qPv = np.dot(q.flatten(), Pv.flatten())
        print("qPv:", qPv/nc)
        theta = np.squeeze(theta)
        PSI = np.squeeze(np.matmul(X,theta)) # works but not sure! #PSI = np.squeeze(np.sum(x*theta, axis=2))# not sure about axis! I tested it, it works!
        # x: 500*25*29, theta: 29*1 -> PSI: 500*25 We reduced dimension of features
        print("<q,PSI>: ", sum([np.dot(q[i], PSI[i]) for i in range(nc)])/nc)
        obj = qPv - sum([np.dot(q[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten())
        
        fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        ##fPv = fair_loss(X, P, vvector(nn), group_feat_id)
        obj = obj + np.dot(alpha, fPv)
        print("alpha * fPv: ", np.dot(alpha, fPv)/nc)
        print("mu/2*||P||: ",(mu/2) *np.dot(P.flatten(), P.flatten())/nc)
        print("mu/2*||q||: ",(mu/2) *np.dot(q.flatten(), q.flatten())/nc)
        print("gamma/2*||theta||: ",(gamma/2) *np.dot(theta, theta))
        print("alpha: ", alpha)
        print("lambda/2*||alpha||: ",(lambda_group_fairness/2) *np.dot(alpha, alpha))
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        obj = obj + (lambda_group_fairness/2) * np.dot(alpha, alpha)
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta)

        gr_q = np.array((Pv - PSI + mu*q)/nc) #####/nc I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
        gr_alpha = np.array((fPv + lambda_group_fairness*alpha)/nc) ## /nc # fPv: 500*1, alpha: 500*1, Gr_alpha: 500*1, [Gr, Gr_alpha]: 500*26
        #########################gr_alpha = np.array(fPv/nc) ##########################
        gr_alpha = np.reshape(gr_alpha, (-1, 1)) # convert 1*500 to 500*1
        gr_new = np.array([np.append(gr_q[i] , gr_alpha[i]) for i in range(len(gr_q))])
        g = gr_new.flatten()

    return obj, g


def fair_loss(X, P, v, group_feat_id): # fPv
    nc,nn,nf = np.shape(X)
    f = fairnessConstraint(X, group_feat_id)
    P_matching = np.zeros((nc,nn,nn))
    for i in range(0, nc):
        P_matching[i] = get_matching(P[i])
    Pv = np.matmul(P_matching, vvector(nn))
    fPv = np.array([np.dot(f[i], Pv[i]) for i in range(nc)])
    return fPv
    
    

def fairnessConstraint(x, group_feat_id): # seems f is okay, f: 500*25
    g1 = [sum(x[i][:][group_feat_id]) for i in range(len(x))] # |G_1| for each ranking sample
    g0 = [len(x[i])-g1[i] for i in range(len(x))] # |G_0| for each ranking sample # Male, priviledged
    f = [[max(0, int(x[i][j][group_feat_id] == 0)/g0[i] - int(x[i][j][group_feat_id] == 1)/g1[i]) for j in range(len(x[i]))] for i in range(len(x))]
    f = np.array(f)
    #print("test.py: fairnessConstraint")
    return f



def minP(q,f,alpha,v, mu): # P: 500*25*25 #####################
    # Find optimal P
    global last_P
    nc = len(q) #500
    nn = len(q[0]) #25
    P = np.zeros((nc,nn,nn))
    print("test.py: minP")
    for i in range(0, nc):
        Pi_init = np.zeros((nn,nn))#last_P[i] #[[1.0/nn for _ in range(nn)] for _ in range(nn)] # checked! initiate with something else? like bipartite code?
        # run ADMM
        S = (q[i] + alpha[i]*f[i])
        R = 1/mu * np.outer(S, np.transpose(v)) # 25*1 multiply by 1*25 should give 25*25 matrix
        P[i] = Project( R , Pi_init)
    last_P = P
    return P


def findUtility(P, u): # U(P|q) = u^T P v
    
    u = np.array(u) # u: 500*25
    v = [1/math.log(1+j, 2) for j in range(1, len(P[0])+1)] # q: 500*25, v: 25*1  
    Pv = np.matmul(P, v) # Pv: 500*25, P: 500*25*25, v:25*1
    uPv = [np.dot(u[i], Pv[i]) for i in range(len(u))] # uPv: 500*1
    Ideal_utility = [np.dot(sorted(u[i], reverse=True), v) for i in range(len(u))] # 500*1
    return np.array(uPv)/np.array(Ideal_utility)



def vvector(N):

    v = [1/math.log(1+j, 2) for j in range(1, N+1)] # q: 500*25, v: 25*1
    return np.array(v)

def BvN(P, u, vvector):
    ndcgs = np.zeros(len(P))
    for i in range(len(P)):
        #print("P[i]", P[i])
        result = decomp(P[i])
        for coefficient, permutation_matrix in result:
            ndcgs[i] += coefficient * get_ndcg(permutation_matrix, u[i], vvector)
            #print('coefficient:', coefficient)
            #print('permutation matrix:', permutation_matrix)
    return sum(ndcgs)/len(ndcgs)


def evaluate(P, x, u, vvector, group_feat_id):
    test_losses = []
    dp_test_losses = []
    test_ndcgs = []
    test_rank_ndcgs = [] ##
    test_matching_ndcgs = []
    nc = len(u)
    
    #ranking = sample_ranking(probs, False)
    #ndcg, dcg = compute_dcg(ranking, rel, args.evalk)
    for i in range(nc):
        groups = np.array(x[i][:, group_feat_id], dtype=np.int)
        test_loss = get_fairness_loss(P[i], u[i], vvector, groups) ##############
        dp_test_loss = get_dp_fairness_loss(P[i], u[i], vvector, groups) ##############
        test_losses.append(test_loss)
        dp_test_losses.append(dp_test_loss)
        test_ndcg = get_ndcg(P[i], u[i], vvector)
        test_rank_ndcg = get_rank_ndcg(P[i], u[i], vvector) ##
        test_matching_ndcg = get_matching_ndcg(P[i], u[i], vvector)
        test_ndcgs.append(test_ndcg)
        test_rank_ndcgs.append(test_rank_ndcg) ##
        test_matching_ndcgs.append(test_matching_ndcg)
    return np.mean(test_matching_ndcgs), np.mean(test_rank_ndcgs), np.mean(test_ndcgs), np.mean(dp_test_losses), np.mean(test_losses)


def testAdvarsarialRanking(x ,u , model, args):
    
    nc,nn,nf = np.shape(x)
    global last_P
    last_P = [[[1.0/nn for _ in range(nn)] for _ in range(nn)] for _ in range(nc)]
    print("test.py: testAdversarialRanking")
    theta = model["theta"]
    gamma = args.gamma
    mu = args.mu
    f = fairnessConstraint(x, args.group_feat_id)  
    
    if args.lambda_group_fairness > 0.0:
    
        q_alpha_init = np.random.random(nc*(nn+1)) # 500*(25+1) because we added alpha to q
        bd =[(0.0,1.0)]*nn
        bd.append((args.lambda_group_fairness, args.lambda_group_fairness))  #bd.append((None, None)) # bounds for alpha
        bd = bd*nc

        optim = optimize.minimize(ranking_q_object, x0 = q_alpha_init, args=(x, gamma, mu, args.lambda_group_fairness, args.group_feat_id, theta), method='L-BFGS-B', jac=True,  bounds=bd, options={'eps': 1, 'ftol' : 100 * np.finfo(float).eps})

        q_alpha = np.reshape(optim.x, (-1, nn+1)) # (500*26,) --> (500, 26)
        q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
        alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
        
        
    elif args.lambda_group_fairness == 0.0: # No Fairness Constraints

        q_init = np.random.random(nc*nn) # no alpha
        #q_init = np.random.uniform(-1.0, 1.0, size=nc*nn) ##
        bd =[(0.0,1.0)]*nn*nc
        optim = optimize.minimize(ranking_q_object, x0 = q_init, args=(x, gamma, mu, args.lambda_group_fairness, args.group_feat_id, theta), method='L-BFGS-B', jac=True,  bounds=bd, options={'eps': 1, 'ftol' : 100 * np.finfo(float).eps})

        q = np.reshape(optim.x, (-1, nn)) # (500*25,) --> (500, 25)
        alpha = np.zeros(nc)
    
    P_optimal = minP(q,f,alpha,vvector(nn), mu)
    U = findUtility(P_optimal, u)
    #print("U is: ", sum(U)/len(U))
    #groups = np.array(x[i][:, group_feat_id], dtype=np.int)
    matching_ndcg, rank_ndcg, ndcg, dp_fair_loss, fair_loss = evaluate(P_optimal, x,  u, vvector(nn), args.group_feat_id)
    #print("ndcg: ", ndcg)
    #print("fair_loss: ", fair_loss)
    result = {
        "lambda": args.lambda_group_fairness,
        "ndcg": ndcg,
        "rank_ndcg": rank_ndcg,
        "matching_ndcg": matching_ndcg,
        "avg_group_asym_disparity": fair_loss,
        "avg_group_demographic_parity": dp_fair_loss
    }
    return result
    
    
