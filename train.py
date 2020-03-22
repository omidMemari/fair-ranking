import numpy as np
#import matlab.engine
import scipy.io
import math
from scipy import optimize
from numpy import linalg
from projectBistochasticADMM import projectBistochasticADMM as Project

def ranking_q_object(q_init, X, u, gamma, mu, lambda_group_fairness):
    
    print("train.py: ranking_q_object")
    #print("q_init: ", q_init)
    
    #q_alpha = mat['q']
    #rs = len(q_alpha[0]) - 1 # each ranking problem is in size of 25. q_alpha: 500*26, q: 500*25
    #q = np.array([q_alpha[i][:rs] for i in range(len(q_alpha))])
    #alpha = np.array([q_alpha[i][rs] for i in range(len(q_alpha))])
    #alpha = np.array([0.5 for _ in range(len(x))])  # change this. maybe in m file with q!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    nc,nn,nf = np.shape(X)
    #alpha = np.ones(nc)/20 #######
    
    q_alpha = np.reshape(q_init, (-1, nn+1)) # (500*26,) --> (500, 26)
    q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
    alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
    fc = fairnessConstraint(X)
    #get the optimal tetha & P given Q
    theta = maxTheta(q, X, u, gamma) # theta: 29*1 there is another maxTheta in trainAdversarialRanking! I think we don't need this one!
    
    v = [1/math.log(1+j, 2) for j in range(1, nn+1)] # q: 500*25, v: 25*1 
    
    P = minP(q,fc,alpha,v, mu) # P: 500*25*25
    
    
    Pv = np.matmul(P, v) # Pv: 500*25, P: 500*25*25, v:25*1
    qPv = np.dot(q.flatten(), Pv.flatten())
    
    #qPv = [np.matmul(np.matmul(q[i], P[i]), v) for i in range(0, len(P))] # for every 500 ranking samples, we calculate qPv for that sample, resulting in 500*1 vector
    
    q_minus_u = q - u # 500*25
    theta = np.squeeze(theta)
    X = np.array(X)
    PSI = np.squeeze(np.matmul(X,theta)) # works but not sure! #PSI = np.squeeze(np.sum(x*theta, axis=2))# not sure about axis! I tested it, it works! 
    # x: 500*25*29, theta: 29*1 -> PSI: 500*25 We reduced dimension of features
    
    obj = qPv + sum([np.dot(q_minus_u[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten()) 
    ##fPv = np.array([np.dot(fc[i].flatten(), Pv[i].flatten()) for i in range(len(fc))]) #fPv: 500*1, fc:500*25, Pv:500*25
    fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
    ##obj = obj + np.dot(alpha.flatten(), fPv.flatten())
    obj = obj + np.dot(alpha, fPv) # 500*1
    obj = obj - (mu/2) * np.dot(P.flatten(), P.flatten())
    obj = obj + (mu/2) * np.dot(q.flatten(), q.flatten())
    #  regularization
    obj = obj + (lambda_group_fairness/2) * np.dot(alpha, alpha)  # alpha[i]??
    obj = obj / nc
    obj = obj + (gamma/2) * np.dot(theta, theta)
    
    # gradient??? add alpha to it
    gr_q = np.array((Pv + PSI + mu*q)/nc) # I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
    gr_alpha = np.array((fPv + lambda_group_fairness*alpha)/nc) # fPv: 500*1, alpha: 500*1, Gr_alpha: 500*1, [Gr, Gr_alpha]: 500*26
    gr_alpha = np.reshape(gr_alpha, (-1, 1)) # convert 1*500 to 500*1
    gr_new = np.array([np.append(gr_q[i] , gr_alpha[i]) for i in range(len(gr_q))])
    g = gr_new.flatten()
    #scipy.io.savemat('data1.mat', dict(g=g, gr_q=gr_q, gr_alpha=gr_alpha, gr_new=gr_new))
    #scipy.io.savemat('data2.mat', dict(f=obj))
    #print("gr: ", gr_q)
    print("obj: ", obj)


    return obj, g
    

def fairnessConstraint(x): # seems f is okay, f: 500*25
    g1 = [sum(x[i][:][3]) for i in range(len(x))] # |G_1| for each ranking sample
    g0 = [len(x[i])-g1[i] for i in range(len(x))] # |G_0| for each ranking sample
    f = [[int(x[i][j][3] == 0)/g0[i] - int(x[i][j][3] == 1)/g1[i] for j in range(len(x[i]))] for i in range(len(x))]
    f = np.array(f)
    print("train.py: fairnessConstraint")
    #scipy.io.savemat('data4.mat', dict(f=f))
    return f


def maxTheta(q, x , u , gamma):
    # Find optimal theta given Q
    # This is a closed form solution
    nc,nn,nf = np.shape(x)
    #n = len(x) # num of ranking samples: 500
    #nf = int(len(x[0][0])) # num of features: 29
    th = np.zeros(nf)
    print("train.py: maxTheta")
    
    q_minus_u = q - u
    #######print("q_minus_u in maxTheta: ", q_minus_u)
    #temp =np.array(x)[:,:,k]
    #scipy.io.savemat('test.mat', dict(temp=temp))
    #print(len(temp.flatten(order='F')))
    for k in range(0, nf):
        temp = np.array(x)[:,:,k] #500*25*1
        #print("np.array(x)[:,:,k] #500*25*1 : ", temp)
        #th[k] = (-1 / lamda * nc) * np.dot(q_minus_u.flatten(), temp.flatten()) # not tested!
        th[k] = (-1 / gamma * nc) * sum([np.dot(q_minus_u[i], temp[i]) for i in range(nc)])
        #####print("sum([np.dot(q_minus_u[i], temp[i]) for i in range(nc)]) : ", sum([np.dot(q_minus_u[i], temp[i]) for i in range(nc)]))
        ######print("(-1 / lamda * nc) : ", (-1 / lamda * nc))
        ###########print("th[k] #1 : ", th[k])
    
    return th
    
def minP(q,f,alpha,v, mu): # P: 500*25*25  
    # Find optimal P
    nc = len(q) #500
    nn = len(q[0]) #25
    P = np.zeros((nc,nn,nn)) 
    print("train.py: minP")
    for i in range(0, nc):
        Pi_init = [[1.0/nn for _ in range(nn)] for _ in range(nn)] # checked! initiate with something else? like bipartite code?
        # run ADMM
        
        ############print("Pi_init: ", Pi_init)
        S = (q[i] + alpha[i]*f[i])
        #############print("1/mu * (q[i] + alpha[i]*f[i]): ", S)
        #S = np.transpose(S)
        R = 1/mu * np.outer(S, np.transpose(v)) # 25*1 multiply by 1*25 should give 25*25 matrix
        #############print("1/mu * (q[i] + alpha[i]*f[i])*v: ", R)
        #scipy.io.savemat('data6.mat', dict(S=S, vt=v, R=R))
        #P[i] = projectBistochasticADMM( R , Pi_init)
        P[i] = Project( R , Pi_init)
        #print("in minP Projection :",i)
        #############print("after projection P[i] is: ", P[i])
        #scipy.io.savemat('data6.mat', dict(P=P[i]))
    return P
   
def projectBistochasticADMM(X  , Z_init):
    
    scipy.io.savemat('data5.mat', dict(X_data=X, Z_data=Z_init))
    eng = matlab.engine.start_matlab()
    Z = eng.projectBistochasticADMMpython() 
    newZ = np.transpose(Z)  ### check this
    print("train.py: projectBistochasticADMM")
    
    return newZ

       
def trainAdversarialRanking(x, u, args):
    
    #x = dr[0]
    #u = dr[1]
    #x, u = data_reader
    lambda_group_fairness = args.lambda_group_fairness
    gamma = args.gamma
    mu = args.mu
    nc,nn,nf = np.shape(x)
    q_alpha_init = np.random.random(nc*(nn+1))
    #print("q_init", q_init)
    
    
    bd = [(0.0,1.0)]*nc*(nn+1) # change it! alpha should be flexible
    #lb = list(np.zeros(nc*nn))
    #ub = list(np.ones(nc*nn))
    #print("lb", bound)
    
    #print(np.shape(q_init),np.shape(lb),np.shape(ub))
    optim = optimize.minimize(ranking_q_object, x0 = q_alpha_init, args=(x, u, gamma, mu, lambda_group_fairness), method='L-BFGS-B', jac=True,  bounds=bd)
    print(optim)

    #q = np.array([q_alpha[i][:25] for i in range(len(q_alpha))])
    #scipy.io.savemat('data8.mat', dict(q=q, q_alpha=q_alpha))
    print("train.py: trainAdversarialRanking")
    q_alpha = np.reshape(optim.x, (-1, nn+1)) # (500*26,) --> (500, 26)
    ##############################3
    #rs = len(q_alpha[0]) - 1 # each ranking problem is in size of 25. q_alpha: 500*26, q: 500*25
    q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
    alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
    #alpha = np.ones(nc)/20
    f = fairnessConstraint(x)
    v = [1/math.log(1+j, 2) for j in range(1, nn+1)] # q: 500*25, v: 25*1 
    ################################
    P = minP(q,f,alpha,v, mu)
    theta = maxTheta(q, x, u, gamma)
    #print("final theta : ", theta)
    model = {
        "theta": theta,
        "q": q
    }
    
    return model    
    
