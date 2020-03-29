import numpy as np
#import matlab.engine
import scipy.io
import math
from scipy import optimize
from numpy import linalg
from projectBistochasticADMM import projectBistochasticADMM as Project
from test import evaluate

def ranking_q_object(q_init, X, u, gamma, mu, lambda_group_fairness):
    
    print("train.py: ranking_q_object")
    nc,nn,nf = np.shape(X)
    X = np.array(X)
    fc = fairnessConstraint(X)
    
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

        obj = qPv + sum([np.dot(q_minus_u[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten()) 
        fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        #obj = obj + np.dot(alpha, fPv) # 500*1
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta)  ##################################
        gr_q = np.array((Pv + PSI + mu*q)/nc) # I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
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

        obj = qPv + sum([np.dot(q_minus_u[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten()) 
        fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
        obj = obj + np.dot(alpha, fPv) # 500*1
        obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
        obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
        #  regularization
        obj = obj + (lambda_group_fairness/2) * np.dot(alpha, alpha)  # alpha[i]??
        obj = obj / nc
        obj = obj + (gamma/2) * np.dot(theta, theta)##########################
        gr_q = np.array((Pv + PSI + mu*q)/nc) # I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
        gr_alpha = np.array((fPv + lambda_group_fairness*alpha)/nc) # fPv: 500*1, alpha: 500*1, Gr_alpha: 500*1, [Gr, Gr_alpha]: 500*26
        gr_alpha = np.reshape(gr_alpha, (-1, 1)) # convert 1*500 to 500*1
        gr_new = np.array([np.append(gr_q[i] , gr_alpha[i]) for i in range(len(gr_q))])
        g = gr_new.flatten()
        print("obj: ", obj)


    return obj, g
    

def fairnessConstraint(x): # seems f is okay, f: 500*25
    g1 = [sum(x[i][:][3]) for i in range(len(x))] # |G_1| for each ranking sample
    g0 = [len(x[i])-g1[i] for i in range(len(x))] # |G_0| for each ranking sample
    f = [[int(x[i][j][3] == 0)/g0[i] - int(x[i][j][3] == 1)/g1[i] for j in range(len(x[i]))] for i in range(len(x))]
    f = np.array(f)
    return f


def maxTheta(q, x , u , gamma):

    print("train.py: maxTheta")
    nc,nn,nf = np.shape(x)
    th = np.zeros(nf)
    q_minus_u = q - u
   
    for k in range(0, nf):
        temp = np.array(x)[:,:,k] #500*25*1
        th[k] = (-1 / gamma * nc) * sum([np.dot(q_minus_u[i], temp[i]) for i in range(nc)]) ################3
        
    return th/sum(th) #####################
    
def minP(q,f,alpha,v, mu): # P: 500*25*25  
    # Find optimal P
    nc = len(q) #500
    nn = len(q[0]) #25
    P = np.zeros((nc,nn,nn)) 
    print("train.py: minP")
    for i in range(0, nc):
        Pi_init = [[1.0/nn for _ in range(nn)] for _ in range(nn)] # checked! initiate with something else? like bipartite code?
        # run ADMM
        S = (q[i] + alpha[i]*f[i])
        R = 1/mu * np.outer(S, np.transpose(v)) # 25*1 multiply by 1*25 should give 25*25 matrix
        P[i] = Project( R , Pi_init)
    return P
   
def vvector(N):

    v = [1/math.log(1+j, 2) for j in range(1, N+1)] # q: 500*25, v: 25*1
    return np.array(v)

       
def trainAdversarialRanking(x, u, args):
    
    print("train.py: trainAdversarialRanking")
    gamma = args.gamma
    mu = args.mu
    nc,nn,nf = np.shape(x)
    f = fairnessConstraint(x)
    
    if args.lambda_group_fairness > 0.0:
        
        q_alpha_init = np.random.random(nc*(nn+1)) # add alpha to q
        bd =[(0.0,1.0)]*nn
        bd.append((None, None))
        bd = bd*nc
        optim = optimize.minimize(ranking_q_object, x0 = q_alpha_init, args=(x, u, gamma, mu, args.lambda_group_fairness), method='L-BFGS-B', jac=True,  bounds=bd)
        q_alpha = np.reshape(optim.x, (-1, nn+1)) # (500*26,) --> (500, 26)
        q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
        alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
        P = minP(q,f,alpha,vvector(nn), mu)
        theta = maxTheta(q, x, u, gamma)
        
    
    elif args.lambda_group_fairness == 0.0: # No Fairness Constraints
        
        q_init = np.random.random(nc*nn) # no alpha
        bd =[(0.0,1.0)]*nn*nc 
        optim = optimize.minimize(ranking_q_object, x0 = q_init, args=(x, u, gamma, mu, args.lambda_group_fairness), method='L-BFGS-B', jac=True,  bounds=bd)
        q = np.reshape(optim.x, (-1, nn)) # (500*25,) --> (500, 25)
        alpha = np.zeros(nc)
        P = minP(q,f,alpha,vvector(nn), mu)######
        theta = maxTheta(q, x, u, gamma)
        
    ndcg, fair_loss = evaluate(P, x,  u, vvector(nn), args.group_feat_id)
    print(optim)
    model = {
        "theta": theta,
        "q": q,
        "ndcg": ndcg,
        "fair_loss": fair_loss
    }
    
    return model    
    
