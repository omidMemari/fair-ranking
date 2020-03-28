import numpy as np
#import matlab.engine
import scipy.io
import math
from scipy import optimize
from numpy import linalg
from birkhoff import birkhoff_von_neumann_decomposition as decomp
from projectBistochasticADMM import projectBistochasticADMM as Project
from baselines import *

#vvector = lambda N: 1. / np.log2(2 + np.arange(N))

def ranking_q_object(q_init, X, gamma, mu, lambda_group_fairness, theta):

    print("test.py: ranking_q_object")
    #print("q_init: ", q_init)

    #q_alpha = mat['q']
    #rs = len(q_alpha[0]) - 1 # each ranking problem is in size of 25. q_alpha: 500*26, q: 500*25
    #q = np.array([q_alpha[i][:rs] for i in range(len(q_alpha))])
    #alpha = np.array([q_alpha[i][rs] for i in range(len(q_alpha))])
    #alpha = np.array([0.5 for _ in range(len(x))])  # change this. maybe in m file with q!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    nc,nn,nf = np.shape(X)
    #alpha = np.ones(nc)/20 #######

    q_alpha = np.reshape(q_init, (-1, nn+1)) # (500*26,) --> (500, 26)
    #rs = len(q_alpha[0]) - 1 # each ranking problem is in size of 25. q_alpha: 500*26, q: 500*25
    q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
    alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
    print("q_alpha shape: ", np.shape(q_alpha))
    print("q shape: ", np.shape(q))
    print("alpha shape: ", np.shape(alpha))

    fc = fairnessConstraint(X)
    #get the optimal tetha & P given Q
    #theta = maxTheta(q, X, u, lamda) # theta: 29*1 there is another maxTheta in trainAdversarialRanking! I think we don't need this one!

    v = [1/math.log(1+j, 2) for j in range(1, nn+1)] # q: 500*25, v: 25*1

    P = minP(q,fc ,alpha ,v ,mu) # P: 500*25*25

    Pv = np.matmul(P, v) # Pv: 500*25, P: 500*25*25, v:25*1
    qPv = np.dot(q.flatten(), Pv.flatten())
    print("qPv:", qPv/nc)

    #qPv = [np.matmul(np.matmul(q[i], P[i]), v) for i in range(0, len(P))] # for every 500 ranking samples, we calculate qPv for that sample, resulting in 500*1 vector

    #############q_minus_u = q - u # 500*25
    theta = np.squeeze(theta)
    X = np.array(X)
    PSI = np.squeeze(np.matmul(X,theta)) # works but not sure! #PSI = np.squeeze(np.sum(x*theta, axis=2))# not sure about axis! I tested it, it works!
    # x: 500*25*29, theta: 29*1 -> PSI: 500*25 We reduced dimension of features
    print("<q,PSI>: ", sum([np.dot(q[i], PSI[i]) for i in range(nc)])/nc)
    obj = qPv + sum([np.dot(q[i], PSI[i]) for i in range(nc)]) ##np.dot(q_minus_u.flatten(), PSI.flatten())
    ##fPv = np.array([np.dot(fc[i].flatten(), Pv[i].flatten()) for i in range(len(fc))]) #fPv: 500*1, fc:500*25, Pv:500*25
    fPv = np.array([np.dot(fc[i], Pv[i]) for i in range(nc)]) #fPv: 500*1, fc:500*25, Pv:500*25
    ##obj = obj + np.dot(alpha.flatten(), fPv.flatten())
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

    # gradient??? add alpha to it
    gr_q = np.array((Pv + PSI + mu*q)/nc) # I think we don't need nc!! Gr:500*25,  Gr[i]: 25*1, q[i]: 25*1
    gr_alpha = np.array((fPv + lambda_group_fairness*alpha)/nc) # fPv: 500*1, alpha: 500*1, Gr_alpha: 500*1, [Gr, Gr_alpha]: 500*26
    gr_alpha = np.reshape(gr_alpha, (-1, 1)) # convert 1*500 to 500*1
    gr_new = np.array([np.append(gr_q[i] , gr_alpha[i]) for i in range(len(gr_q))])
    g = gr_new.flatten()
    #scipy.io.savemat('data1.mat', dict(g=g, gr_q=gr_q, gr_alpha=gr_alpha, gr_new=gr_new))
    #scipy.io.savemat('data2.mat', dict(f=obj))

    return obj, g

#if group_fairness_evaluation:
#            rel_mean_g0 = np.mean(rel[group_identities == 0])
#            rel_mean_g1 = np.mean(rel[group_identities == 1])
#            # skip for candidate sets when there is no diversity
#            if (np.sum(group_identities == 0) == 0
#                    or np.sum(group_identities == 1) == 0
#                ) or rel_mean_g0 == 0 or rel_mean_g1 == 0:
#                # print(group_identities, rel)
#                group_exposure_disparities.append(0.0)
#                group_asym_disparities.append(0.0)
#                # if there is only one group
#            else:
#                exposure_mean_g0 = np.mean(exposures[group_identities == 0])
#                exposure_mean_g1 = np.mean(exposures[group_identities == 1])
#                # print(exposure_mean_g0, exposure_mean_g1)
#                disparity = exposure_mean_g0 / rel_mean_g0 - exposure_mean_g1 / rel_mean_g1
#                group_exposure_disparity = disparity**2
#                sign = +1 if rel_mean_g0 > rel_mean_g1 else -1
#                one_sided_group_disparity = max([0, sign * disparity])
#                # print(group_exposure_disparity, exposure_mean_g0,
#                # exposure_mean_g1, rel, group_identities)
#                group_exposure_disparities.append(group_exposure_disparity)
#                group_asym_disparities.append(one_sided_group_disparity)

def fairnessConstraint(x): # seems f is okay, f: 500*25
    g1 = [sum(x[i][:][3]) for i in range(len(x))] # |G_1| for each ranking sample
    g0 = [len(x[i])-g1[i] for i in range(len(x))] # |G_0| for each ranking sample
    f = [[int(x[i][j][3] == 0)/g0[i] - int(x[i][j][3] == 1)/g1[i] for j in range(len(x[i]))] for i in range(len(x))]
    f = np.array(f)
    print("test.py: fairnessConstraint")
    #scipy.io.savemat('data4.mat', dict(f=f))
    return f



def minP(q,f,alpha,v, mu): # P: 500*25*25
    # Find optimal P
    nc = len(q) #500
    nn = len(q[0]) #25
    P = np.zeros((nc,nn,nn))
    print("test.py: minP")
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


def findUtility(P, u): # U(P|q) = u^T P v
    
    u = np.array(u) # u: 500*25
    v = [1/math.log(1+j, 2) for j in range(1, len(P[0])+1)] # q: 500*25, v: 25*1  
    Pv = np.matmul(P, v) # Pv: 500*25, P: 500*25*25, v:25*1
    uPv = [np.dot(u[i], Pv[i]) for i in range(len(u))] # uPv: 500*1
    Ideal_utility = [np.dot(sorted(u[i], reverse=True), v) for i in range(len(u))] # 500*1
    return np.array(uPv)/np.array(Ideal_utility)



#def get_best_rankmatrix(true_rel_vector):
#    N = len(true_rel_vector)
#    bestranking = np.zeros((N, N))
#    bestr = np.argsort(true_rel_vector)[::-1]
#    for i in range(N):
#        bestranking[bestr[i], i] = 1
#    return bestranking


#returns DCG value

#def get_DCG(ranking, relevances, vvector):
#    N = len(relevances)
#    return np.matmul(np.matmul(relevances, ranking), vvector.transpose())


#def get_ndcg(ranking, relevances, vvector):
#    if np.all(relevances == 0):
#        return 1.0
#    bestr = get_best_rankmatrix(relevances)
#    return get_DCG(ranking, relevances, vvector) / get_DCG(
#        bestr, relevances, vvector)

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
    test_ndcgs = []
    nc = len(u)
    for i in range(nc):
        groups = np.array(x[i][:, group_feat_id], dtype=np.int)
        #test_loss = get_fairness_loss(P[i], u[i], vvector, groups) ##############
        test_loss = get_dp_fairness_loss(P[i], u[i], vvector, groups) ##############
        test_losses.append(test_loss)
        test_ndcg = get_ndcg(P[i], u[i], vvector)
        test_ndcgs.append(test_ndcg)
    return np.mean(test_ndcgs), np.mean(test_losses)
    #plt_data[j] = [np.mean(test_ndcgs), np.mean(test_losses)]

def testAdvarsarialRanking(x ,u , model, args):
    
    theta = model["theta"]
    lambda_group_fairness = args.lambda_group_fairness
    gamma = args.gamma
    mu = args.mu
    #x, u = data_reader
    #x = vdr[0]
    #u = vdr[1]
    group_feat_id = args.group_feat_id
    nc,nn,nf = np.shape(x)
    print("x shape: ", np.shape(x))
    q_alpha_init = np.random.random(nc*(nn+1)) # 500*(25+1) because we added alpha to q
    #print("q_init", q_init)


    #bd = [(0.0,1.0)]*nc*(nn+1) # change it! so alpha can be flexible
    bd =[(0.0,1.0)]*nn
    bd.append((None, None))
    bd = bd*nc
    print("bd: ", bd)
    #lb = list(np.zeros(nc*nn))
    #ub = list(np.ones(nc*nn))
    #print("lb", bound)

    #print(np.shape(q_init),np.shape(lb),np.shape(ub))
    optim = optimize.minimize(ranking_q_object, x0 = q_alpha_init, args=(x, gamma, mu, lambda_group_fairness, theta), method='L-BFGS-B', jac=True,  bounds=bd)
    #print(optim)
    
    
    #scipy.io.savemat('data.mat', dict(x=x, u=u, lamda=lamda, mu=mu, gamma=gamma))
    #eng = matlab.engine.start_matlab()
    #Q_alpha = eng.trainAdversarial()
    #eng.quit()
    #q_alpha = np.transpose(Q_alpha)
    q_alpha = np.reshape(optim.x, (-1, nn+1)) # (500*26,) --> (500, 26)
    ##############################3
    #rs = len(q_alpha[0]) - 1 # each ranking problem is in size of 25. q_alpha: 500*26, q: 500*25
    q = np.array([q_alpha[i][:nn] for i in range(len(q_alpha))])
    alpha = np.array([q_alpha[i][nn] for i in range(len(q_alpha))])
    #alpha = np.ones(nc)/20
    f = fairnessConstraint(x)
    v = [1/math.log(1+j, 2) for j in range(1, nn+1)] # q: 500*25, v: 25*1
    ################################
    P_optimal = minP(q,f,alpha,vvector(nn), mu)


    #scipy.io.savemat('data.mat', dict(x=x, u=u, lamda=lamda, mu=mu, gamma=gamma, theta=theta))
    #eng = matlab.engine.start_matlab()
    #Q_alpha = eng.testAdversarial()
    #eng.quit()
    #q_alpha = np.transpose(Q_alpha)
    #q = np.array([q_alpha[i][:25] for i in range(len(q_alpha))])
    #scipy.io.savemat('data8.mat', dict(q=q, q_alpha=q_alpha))
    print("test.py: testAdversarialRanking")
    #mat = scipy.io.loadmat('p.mat') # load last P to calculate utility based on that
    #P = mat['P']
    U = findUtility(P_optimal, u)
    print("U is: ", sum(U)/len(U))
    #theta = maxTheta(q, x, u, lamda)
    #bvn_result = BvN(P_optimal, u, vvector(nn))
    #groups = np.array(x[i][:, group_feat_id], dtype=np.int)
    ndcg, fair_loss = evaluate(P_optimal, x,  u, vvector(nn), group_feat_id)
    print("ndcg: ", ndcg)
    print("fair_loss: ", fair_loss)
    result = {
        "ndcg": ndcg,
        "fair_loss": fair_loss
    }
    return result #sum(U)/len(U)
    
    
