import numpy as np
import scipy.io
import math
from scipy import optimize
from numpy import linalg
from baselines import *

#last_P = []


def ranking_q_object(q_init, X, u, gamma, mu, lambda_group_fairness, group_feat_id):
    
    #global last_P
    #print("train.py: ranking_q_object")
    nc,nn,nf = np.shape(X)
    X = np.array(X)
    fc = fairnessConstraint(X, group_feat_id)
    
    if lambda_group_fairness == 0: # No Fairness Constraints
        
        q = np.reshape(q_init, (-1, nn)) # (500*25,) --> (500, 25)
        theta = maxTheta(q, X, u, gamma) # theta: 29*1 there is another maxTheta in trainAdversarialRanking! I think we don't need this one!
        alpha = np.zeros(nc)
        P = minP(q,fc,alpha,vvector(nn), mu) # P: 500*25*25
        Pv = np.matmul(P, vvector(nn)) # Pv: 500*25, P: 500*25*25, v:25*1
        qPv = np.dot(q.flatten(), Pv.flatten())
        q_minus_u = q - u # 500*25
        theta = np.squeeze(theta)
        PSI = np.squeeze(np.matmul(X,theta))

        obj = qPv - sum([np.dot(q_minus_u[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten()) 
        fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        #obj = obj + np.dot(alpha, fPv) # 500*1
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta)  ##################################
        gr_q = np.array((Pv - PSI + mu*q)/nc) ####################/nc I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
        g = gr_q.flatten()
        print("obj: ", obj)

        
    
    else:
        
        q_alpha = np.reshape(q_init, (-1, nn+1)) # (500*26,) --> (500, 26)
        q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
        alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
        #get the optimal tetha & P given Q
        theta = maxTheta(q, X, u, gamma) # theta: 29*1 there is another maxTheta in trainAdversarialRanking! I think we don't need this one!
        P = minP(q,fc,alpha,vvector(nn), mu) # P: 500*25*25
        Pv = np.matmul(P, vvector(nn)) # Pv: 500*25, P: 500*25*25, v:25*1
        qPv = np.dot(q.flatten(), Pv.flatten())
        q_minus_u = q - u # 500*25
        theta = np.squeeze(theta)
        PSI = np.squeeze(np.matmul(X,theta)) # works but not sure! #PSI = np.squeeze(np.sum(x*theta, axis=2))# not sure about axis! I tested it, it works! 
        # x: 500*25*29, theta: 29*1 -> PSI: 500*25 We reduced dimension of features

        obj = qPv - sum([np.dot(q_minus_u[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten()) 
        fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        #fPv = fair_loss(X, P, vvector(nn), group_feat_id)
        obj = obj + np.dot(alpha, fPv) # 500*1
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        ####################obj = obj + (lambda_group_fairness/2) * np.dot(alpha, alpha)  # alpha[i]??
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta)##########################
        gr_q = np.array((Pv - PSI + mu*q)/nc) ##########/nc I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
        gr_alpha = np.array((fPv + lambda_group_fairness*alpha)/nc) # fPv: 500*1, alpha: 500*1, Gr_alpha: 500*1, [Gr, Gr_alpha]: 500*26
        #########################gr_alpha = np.array(fPv/nc)############################3
        gr_alpha = np.reshape(gr_alpha, (-1, 1)) # convert 1*500 to 500*1
        gr_new = np.array([np.append(gr_q[i] , gr_alpha[i]) for i in range(len(gr_q))])
        g = gr_new.flatten()
        print("obj: ", obj)
    #print()
    #print("g: ", gr_q[0])
    #print()
    #print("theta: ", theta)
    print()
    print(' '.join('{:06.5f}'.format(item) for item in u[0]))
    print()
    print(' '.join('{:06.5f}'.format(item) for item in q[0]))
    print()
    print('\n'.join([''.join(['{:06.6f}'.format(item) for item in row]) for row in P[0]]))
    #print(np.matrix(last_P[0]))
    print()
    print()
        #print("last_P: ", last_P[0])


    return obj, g
    


def maxTheta(q, x , u , gamma):

    #print("train.py: maxTheta")
    nc,nn,nf = np.shape(x)
    th = np.zeros(nf)
    q_minus_u = q - u
   
    for k in range(0, nf):
        temp = np.array(x)[:,:,k] #500*25*1
        th[k] = (-1  / (gamma * nc) ) * sum([np.dot(q_minus_u[i], temp[i]) for i in range(nc)]) ################3
        
    return th#/sum(th) #####################
    

       
def trainAdversarialRanking(x, u, args):
    
    nc,nn,nf = np.shape(x)
    #global last_P
    last_P = [[[1.0/nn for _ in range(nn)] for _ in range(nn)] for _ in range(nc)]
    print("train.py: trainAdversarialRanking")
    gamma = args.gamma
    mu = args.mu
    f = fairnessConstraint(x, args.group_feat_id)
    
    if args.lambda_group_fairness > 0.0:
        
        q_alpha_init = np.random.random(nc*(nn+1)) # add alpha to q
        bd =[(0.0,1.0)]*nn
        bd.append((args.lambda_group_fairness, args.lambda_group_fairness))  #bd.append((None, None)) # bounds for alpha
        bd = bd*nc
        optim = optimize.minimize(ranking_q_object, x0 = q_alpha_init, args=(x, u, gamma, mu, args.lambda_group_fairness, args.group_feat_id), method='L-BFGS-B', jac=True,  bounds=bd, options={'eps': 1, 'ftol' : 100 * np.finfo(float).eps})
        q_alpha = np.reshape(optim.x, (-1, nn+1)) # (500*26,) --> (500, 26)
        q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
        alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
        P = minP(q,f,alpha,vvector(nn), mu)
        theta = maxTheta(q, x, u, gamma)
        
    
    elif args.lambda_group_fairness == 0.0: # No Fairness Constraints
        
        q_init = np.random.random(nc*nn) # no alpha
        #q_init = np.random.uniform(-1.0, 1.0, size=nc*nn) ##
        bd =[(0.0,1.0)]*nn*nc 
        optim = optimize.minimize(ranking_q_object, x0 = q_init, args=(x, u, gamma, mu, args.lambda_group_fairness, args.group_feat_id), method='L-BFGS-B', jac=True,  bounds=bd, options={'eps': 1, 'ftol' : 100 * np.finfo(float).eps})
        q = np.reshape(optim.x, (-1, nn)) # (500*25,) --> (500, 25)
        alpha = np.zeros(nc)
        P = minP(q,f,alpha,vvector(nn), mu)######
        theta = maxTheta(q, x, u, gamma)
        
    ##matching_ndcg, rank_ndcg, ndcg, dp_fair_loss, fair_loss = evaluate(P, x,  u, vvector(nn), args.group_feat_id)
    result = evaluate(P, x,  u, vvector(nn), args.group_feat_id)
    print(optim)
    model = {
        "theta": theta,
        "q": q,
        "ndcg": result["ndcg"],
        "matching_ndcg": result["matching_ndcg"],
        "avg_group_asym_disparity": result["avg_group_asym_disparity"],
        "avg_group_demographic_parity": result["avg_group_demographic_parity"]
    }
    
    return model    
