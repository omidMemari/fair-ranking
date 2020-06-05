import numpy as np
from abc import abstractmethod
from scipy.optimize import fmin_bfgs, minimize, check_grad
import pdb
import math
import matplotlib.pyplot as plt

def _log_logistic(X):
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----

    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def _dot_intercept(w, X):
    """ This function is used from scikit-learn source code. Source link below """

    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    Returns
    -------
    w : ndarray, shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.
    c : float
        The intercept.
    yz : float
        y * np.dot(X, w).
    
    Notes
	-----
	Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/logistic.py

    """
    c = 0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = np.dot(X, w) + c
    return z

gr = (math.sqrt(5) + 1) / 2

def _binary_search(f, a, b, tol=1e-5):
    a, b = min(a,b), max(a,b)
    while f(a) > 0:
        a -= a
    while f(b) < 0:
        b += b
    
    while abs(a - b) > tol:
        c = (a + b) / 2
        fc = f(c)
        if fc < 0 :
            a = c
        elif fc > 0:
            b = c
        else:
            return c
        print("mu: [{:.3f}, {:.3f},b] c {:.4f}: {:.4f}".format(a,b,c,fc))

    return (a + b) / 2
    
        

     
def gss(f, a, b, tol=1e-5):
    """Golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678

    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    #c, d = a, b
    while abs(c - d) > tol:
        fc, fd = f(c), f(d)
        if fc < fd:
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        print("mu: c {:.4f}: {:.4f} , d {:.4f} : {:.4f}".format(c,fc,d,fd))
    return (b + a) / 2

def plot_newton(mu,delta_f,st_ratio,z, solution,a1,a0):
        fun = lambda x : np.log(1-x) - np.log(x) + mu * x * delta_f + st_ratio * z #-  a1 - a0  
        eps = 1e-4
        x = np.linspace(eps,1 - eps,20)
        y = np.array([fun(i)[:100] for i in x]).T
        print(y.shape)
        for i in range(y.shape[0]):
            plt.plot(x,y[i,:])
        #print(x)
        #print(y)
        plt.scatter(solution[:100], fun(solution)[:100], c = 'red')
        plt.show()
        plt.cla()

class fair_covariate_logloss_classifier:
    def __init__(self, tol=1e-6, verbose=True, max_iter=10000, C = .001, random_initialization=False, root_finding_alg = 'newton', trg_grp_marginal_matching = True):
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.C = C
        self.random_start = random_initialization
        self.root_finding_alg = root_finding_alg
        #self.mu = mu
        self.theta = None
        self.max_root_alg_iter = 100
        self.lb_prob = 1e-12
        #self.root_finding_alg = 'halley'
        self.trg_grp_marginal_matching = trg_grp_marginal_matching
        self.max_epoch = 5
    @abstractmethod
    def f_value(self, a, y):
        pass
    
    def _newton_update(self,p,mu,delta_f, st_ratio, z):
        eps = self.lb_prob
        iter = 0
        while iter < self.max_root_alg_iter:
            f = np.log(1-p) - np.log(p) + mu * p * delta_f + st_ratio * z 
            fst_derv = (-1/(1 - p)) - (1/p) + mu * delta_f
            nxt_updt = f / fst_derv  
            if np.all(np.abs(nxt_updt) < 1e-8):
                break
            p = p - nxt_updt
            p = np.maximum(eps, np.minimum(1 - eps, p))
            #pdb.set_trace()
            iter += 1
        return p

    def _halley_update(self, p,mu, delta_f, st_ratio, z):
        eps = self.lb_prob
        iter = 0
        while iter < self.max_root_alg_iter:
            f = np.log(1-p) - np.log(p) + mu * p * delta_f + st_ratio * z 
            fst_derv = (-1/(1 - p)) - (1/p) + mu * delta_f
            snd_derv = -1/((1-p)**2) + 1/p**2
            nxt_updt = (2 * f * fst_derv) / (2 * fst_derv**2 - f * snd_derv)    
            if np.any(np.isnan(nxt_updt)) or np.all(np.abs(nxt_updt) < .01):
                break
            #if np.any( np.logical_or(p - nxt_updt < 0, p - nxt_updt > 1)):
            #    break
            p = p - nxt_updt
            p = np.maximum(eps, np.minimum(1 - eps, p))
            #pdb.set_trace()
            iter += 1
        return p
    
    @abstractmethod
    def compute_theta_loss_grad(self,theta,mu1,mu2,X_src,Y_src,A_src,src_st_ratio):
        pass
    @abstractmethod
    def compute_lambda_loss_grad(self,lambdas,mu1,mu2,X_trg,A_trg,trg_st_ratio):
        pass
    
    def find_theta_with_grad(self,mu1,mu2,X_src,Y_src,A_src,src_st_ratio):        

        #n = np.size(Y)
        #X_src = np.hstack((X_src,np.ones((n,1))))
        #m = X_src.shape[1]
        if self.theta is None:
            if self.random_start:
                theta = np.random.random_sample((m,)) - .5 
            else:
                theta = np.zeros((m,))
        else:
            theta = self.theta
        #lambdas = np.zeros((2,1)) + .01
        #theta = self.find_theta(mu1,mu2,X_src,Y_src,A_src,src_st_ratio)
        #if self.trg_grp_marginal_feature:
        #    theta = theta.reshape((-1,1))
        #    lambdas = lambdas.reshape((-1,1))
        #    theta = np.vstack((theta,lambdas))
        #    theta = theta.squeeze()
        rate = .0005 # learning rate
        max_itr = self.max_iter
        min_gradient= 1e-5

        S_g = np.ones_like(theta) * 1e-8  #prevent dividing by zero
        l_0 = 0
        l_1 = (1+math.sqrt(1 + 4 * l_0**2))/2
        delta_1 = 0
        t = 1
        #pdb.set_trace()
        while True:
            t = t + 1
            decay = math.sqrt(1000 / (1000 + t))
            l_2 = (1 + math.sqrt(1 + 4 * l_1**2)) / 2
            l_3 = (1 - l_1)/l_2
            _,G = self.compute_theta_loss_grad(theta,self.lambdas, mu1, mu2, X_src,Y_src, A_src, src_st_ratio)
            if self.verbose:
                print("Theta gnorm {:.7f}".format(np.linalg.norm(G)))
            if np.linalg.norm(G) < min_gradient: # convergence threshold
                print('Optimization stops by reaching minimum gradient. {:.9f}'.format(np.linalg.norm(G)))
                break
            elif t > max_itr:
                print('Optimization stops by reaching maximum iteration. {:.9f}'.format(np.linalg.norm(G)))
                break
            S_g = S_g + np.square(G) # for adaptive gradient  
            delta_2 = theta - decay * rate * G / np.sqrt(S_g)  # adaptive gradient and Nesterov's Accelerated Gradient Descent
            theta = (1 - l_3) * delta_2 + l_3 * delta_1
            delta_1 = delta_2
            l_1 = l_2
            #pdb.set_trace()
        self.theta = theta
        #self.lambdas = lambdas
        #if self.trg_grp_marginal_feature:
        #    self.theta = theta[:-2,:]
        #    self.lambdas = theta[-2:,:]
        return self.theta #, self.lambdas

    def find_theta(self,mu1,mu2,X_src,Y_src,A_src,src_st_ratio):
        #n = np.size(Y_src)
        #X_src = np.hstack((X_src,np.ones((n,1))))
        
        m = X_src.shape[1]
        
        if True or self.theta is None:
            if self.random_start:
                theta = np.random.random_sample((m,)) - .5 
            else:
                theta = np.zeros((m,))
        else:
            theta= self.theta
        #f = lambda w : self.compute_loss_grad(w,X, Y)[0]
        #grad = lambda w : self.compute_loss_grad(w,X, Y)[1]
        #self.trg_grp_marginal_feature = False
        #lambdas = np.zeros((2,1)) 
        #if self.trg_grp_marginal_feature:
        #    theta = np.vstack((theta,lambdas)) 
        #theta = theta.squeeze()
        #def callback(w):
        #    f, g = self.compute_loss_grad(w,mu1,mu2,X,A_src,A_trg,Y,st_ratio,lambdas)
        #    print("fun_value {:.4f} \t gnorm {:.4f}".format(f,np.linalg.norm(g)))
        #res = fmin_bfgs(f,theta, grad, gtol=self.tol, maxiter=self.max_iter,full_output=False,disp=True, retall=True, callback = callback)
        #res = minimize(self.compute_loss_grad, theta,args=(X, Y), method='L-BFGS-B',jac=True, tol=self.tol, options={'maxiter':self.max_iter, 'disp':False}, callback=callback)
        res = minimize(self.compute_theta_loss_grad, theta,args=(self.lambdas,mu1,mu2, X_src,Y_src, A_src,src_st_ratio), method='L-BFGS-B',jac=True, tol=1e-12, options={'maxiter':self.max_iter, 'disp':self.verbose, 'gtol':self.tol})
        #func = lambda x : self.compute_loss_grad(x,mu1,mu2, X, A, Y,st_ratio)[0]
        #grad = lambda x : self.compute_loss_grad(x,mu1,mu2, X, A, Y,st_ratio)[1]
        #print(check_grad(func,grad,theta)) 
        self.theta = res.x
        #self.lambdas = lambdas
        #if self.trg_grp_marginal_feature:
        #    self.lambdas = res.x[-2:]
        #    self.theta = res.x[:-2]
        #pdb.set_trace()
        #self.trg_grp_marginal_feature = True
        return self.theta #, self.lambdas
    
    def find_lambdas(self, mu1, mu2, X_trg,A_trg,trg_st_ratio):
        lambdas = self.lambdas
        res = minimize(self.compute_lambda_loss_grad, lambdas,args=(mu1 ,mu2, X_trg,A_trg,trg_st_ratio), method='Newton-CG',jac=True, tol=1e-12, options={'maxiter':self.max_iter, 'disp':self.verbose, 'gtol':self.tol})
        self.lambdas = res.x
        return self.lambdas


    def find_theta_lambda_with_grad(self,mu1,mu2,X_src,Y_src,A_src,src_st_ratio,X_trg,A_trg,trg_st_ratio):        

        lambdas = self.lambdas
        #theta = self.find_theta(mu1,mu2,X_src,Y_src,A_src,src_st_ratio)
        theta = self.theta
        lambda_rate = 1 # learning rate
        theta_rate = .01
        max_itr = self.max_iter
        min_val_t = self.tol
        min_val_l = 1e-6

        S_g_t = np.ones_like(theta) * 1e-8  #prevent dividing by zero
        S_g_l = np.ones_like(lambdas) * 1e-8  #prevent dividing by zero
        l_0 = 0
        l_1 = (1+math.sqrt(1 + 4 * l_0**2))/2
        delta_1_l, delta_1_t = 0,0
        t = 1
        #pdb.set_trace()
        while True:
            t = t + 1
            decay = math.sqrt(1000 / (1000 + t))
            l_2 = (1 + math.sqrt(1 + 4 * l_1**2)) / 2
            l_3 = (1 - l_1)/l_2
            _,G_l = self.compute_lambda_loss_grad(lambdas,theta,mu1, mu2, X_trg, A_trg, trg_st_ratio)
            _,G_t = self.compute_theta_loss_grad(theta,lambdas,mu1, mu2, X_src,Y_src, A_src, src_st_ratio)
            if self.verbose:
                print("Lambda gnorm {:.7f}, Theta norm {:.7f}".format(np.linalg.norm(G_l),np.linalg.norm(G_t)))
            if np.linalg.norm(G_t) < min_val_t and np.linalg.norm(G_l) < min_val_l: # convergence threshold
                print('Optimization stops by reaching minimum gradient. {:.9f} lambda gradient {:.9f}'.format(np.linalg.norm(G_t),np.linalg.norm(G_l)))
                break
            elif t > max_itr:
                print('Optimization stops by reaching maximum iteration. {:.9f} lambda gradient {:.9f}'.format(np.linalg.norm(G_t),np.linalg.norm(G_l)))
                break
            S_g_t = S_g_t + np.square(G_t) # for adaptive gradient  
            S_g_l = S_g_l + np.square(G_l) # for adaptive gradient  
            delta_2_l = lambdas - decay * lambda_rate * G_l / np.sqrt(S_g_l)  # adaptive gradient and Nesterov's Accelerated Gradient Descent
            delta_2_t = theta - decay * theta_rate * G_t / np.sqrt(S_g_t)  # adaptive gradient and Nesterov's Accelerated Gradient Descent
            lambdas = (1 - l_3) * delta_2_l + l_3 * delta_1_l
            theta = (1 - l_3) * delta_2_t + l_3 * delta_1_t
            delta_1_l = delta_2_l
            delta_1_t = delta_2_t
            l_1 = l_2
            #pdb.set_trace()
        #pdb.set_trace()
        self.lambdas = lambdas
        self.theta = theta
        #if self.trg_grp_marginal_feature:
        #    self.theta = theta[:-2,:]
        #    self.lambdas = theta[-2:,:]
        return self.theta, self.lambdas #, self.lambdas
    
    def find_lambda_with_grad(self,mu1,mu2,X_trg,A_trg,trg_st_ratio):        

        lambdas = self.lambdas
        rate = 1 # learning rate
        max_itr = self.max_iter
        min_val= 1e-5

        S_g = np.ones_like(lambdas) * 1e-8  #prevent dividing by zero
        l_0 = 0
        l_1 = (1+math.sqrt(1 + 4 * l_0**2))/2
        delta_1 = 0
        t = 1
        #pdb.set_trace()
        while True:
            t = t + 1
            decay = math.sqrt(1000 / (1000 + t))
            l_2 = (1 + math.sqrt(1 + 4 * l_1**2)) / 2
            l_3 = (1 - l_1)/l_2
            _,G = self.compute_lambda_loss_grad(lambdas,self.theta,mu1, mu2, X_trg, A_trg, trg_st_ratio)
            if self.verbose:
                print("Lambda gnorm {:.7f}".format(np.linalg.norm(G)))
            if np.linalg.norm(G) < min_val: # convergence threshold
                print('Optimization stops by reaching minimum gradient. {:.9f}'.format(np.linalg.norm(G)))
                break
            elif t > max_itr:
                print('Optimization stops by reaching maximum iteration. {:.9f}'.format(np.linalg.norm(G)))
                break
            S_g = S_g + np.square(G) # for adaptive gradient  
            delta_2 = lambdas - decay * rate * G / np.sqrt(S_g)  # adaptive gradient and Nesterov's Accelerated Gradient Descent
            lambdas = (1 - l_3) * delta_2 + l_3 * delta_1
            delta_1 = delta_2
            l_1 = l_2
            #pdb.set_trace()
        self.lambdas = lambdas
        #self.lambdas = lambdas
        #if self.trg_grp_marginal_feature:
        #    self.theta = theta[:-2,:]
        #    self.lambdas = theta[-2:,:]
        return self.lambdas #, self.lambdas
    def find_best_mu(self, mu_range, X_src,Y_src,A_src,src_st_ratio, X_trg,A_trg,trg_st_ratio ):
        mu0 = mu_range[0]
        mu1 = mu_range[1]
        def f(mu):
            theta = self.find_theta_with_grad(mu,X_src,Y_src,A_src,src_st_ratio)
            print(self.q_expected_logloss(X_trg,A_trg,trg_st_ratio, mu))
            print(self.q_fairness_violation(X_trg,A_trg,trg_st_ratio, mu))
            return self.q_fairness_violation(X_trg,A_trg,trg_st_ratio, mu) #compute violation on trg
        
        a, b = mu0, mu1
        a, b = min(a,b), max(a,b)
        while f(a) > 0:
            print(a)
            a -= .2
        while f(b) < 0:
            print(b)
            b += .2
        
        while abs(a - b) > 1e-4:
            c = (a + b) / 2
            fc = f(c)
            print("mu: [{:.3f}, {:.3f},b] c {:.4f}: {:.4f}".format(a,b,c,fc))
            if fc < 0 :
                a = c
            elif fc > 0:
                b = c
            else:
                return c
            #self.mu = c
        return (a + b) / 2
        
    @abstractmethod
    def compute_p_and_q(self,theta,lambdas,mu1,mu2,X,A,st_ratio):
        pass    
    def predict_p_q(self,X,A,st_ratio,theta = None,lambdas = None,mu1 = None, mu2 = None):
        if theta is None:
            theta = self.theta
        if mu1 is None:
            mu1 = self.mu1
        if mu2 is None:
            mu2 = self.mu2
        if lambdas is None:
            lambdas = self.lambdas
        #if self.trg_grp_marginal_feature:
        #    X = self.build_trg_grp_marginal_feature(X,A,st_ratio)
        p,q = self.compute_p_and_q(theta,lambdas,mu1,mu2,X,A, st_ratio)
        return p,q
    @abstractmethod    
    def predict_proba(self,X,A, st_ratio):
        p,q = self.predict_p_q(X,A, st_ratio,self.theta, self.lambdas, self.mu1,self.mu2)
        return p

    def predict(self,X,A, st_ratio):
        return np.round(self.predict_proba(X,A, st_ratio))

    @abstractmethod
    def fairness_violation(self,X,Y,A,ratio):
        pass
    @abstractmethod
    def _q_fairness_violation(self,p,q,A):
        pass
    def q_fairness_violation(self,X,A,st_ratio, mu1 = None, mu2 = None):
        p, q = self.predict_p_q(X,A,st_ratio,self.theta,self.lambdas,mu1, mu2)
        return self._q_fairness_violation(p,q,A) 
    
    def q_marginal_grp_estimation_error(self, X, A, st_ratio, a= 1, mu1 = None, mu2 = None):
        p, q = self.predict_p_q(X,A,st_ratio,self.theta,self.lambdas,mu1,mu2)
        if a == 0:
            A = 1 - A
        err = np.dot(q,A) /A.shape[0] - self.trg_group_estimator(a) 
        return err

    def score(self,X,Y,A,st_ratio):
        return 1 - np.mean(abs(self.predict(X,A,st_ratio) - Y))
    def expected_error(self,X,Y,A,ratio):
        proba = self.predict_proba(X,A,ratio)
        return np.mean(np.where(Y == 1 , 1 - proba, proba))
    def expected_logloss(self,X,Y,A,st_ratio):
        p,q = self.predict_p_q(X,A,st_ratio)
        print("P :min {:.4f}, max {:.4f}, < .5 {:.4f}, avg {:.4f}".format(np.min(p), np.max(p), np.sum(np.less(p,.5)), np.mean(p)))
        print("Q: min {:.4f}, max {:.4f}, < .5 {:.4f}, avg {:.4f}".format(np.min(q), np.max(q), np.sum(np.less(q,.5)), np.mean(q)))
        return np.mean( - Y * np.log(p) - (1-Y) * np.log(1 - p))

    def q_expected_logloss(self,X,A,st_ratio, mu1 = None, mu2 = None):
        p,q = self.predict_p_q(X,A,st_ratio,self.theta,None,mu1, mu2)
        return np.mean( - q * np.log(p) - (1-q) * np.log(1 - p))
    @abstractmethod    
    def q_objective(self,X,A,st_ratio,mu1 = None, mu2 = None):
        pass

    
                


class EOPP_fair_covariate_logloss_classifier(fair_covariate_logloss_classifier):
    def __init__(self, trg_group_estimator, tol=1e-8, verbose=True, max_iter=10000, C = .1, random_initialization=False, trg_grp_marginal_matching = True):
        super().__init__( tol = tol, verbose = verbose, max_iter = max_iter, C = C, random_initialization= random_initialization)
        self.trg_group_estimator = trg_group_estimator
        self.trg_grp_marginal_matching = trg_grp_marginal_matching
        self.mu2 = 0
        self.lambdas = np.zeros((2,))
        
    def f_value(self,A,Y = None):
        if Y is None:
            # Y unavailable in target mode
            grp1 = np.where(A == 1, 1/self.trg_group_estimator(1), 0)
            grp2 = np.where(A == 0, -1/self.trg_group_estimator(0), 0)
            #grp1 = np.where(A == 1, 1/np.mean(A), 0)
            #grp2 = np.where(A == 0, -1/np.mean(1-A), 0)
            
        #else:    
        #    f1 = np.mean(np.logical_and(A == 1, Y == 1))
        #    f0 = np.mean(np.logical_and(A == 0, Y == 1))
        
        #    grp1 = np.where(np.logical_and(A == 1, Y == 1), 1/f1, 0)
        #    grp2 = np.where(np.logical_and(A == 0, Y == 1), -1/f0, 0)
        f_val= grp1 + grp2
        return f_val
    
    def _solve_p_binary_search(self,mu,A,st_ratio,z,lambdas):
        eps = self.lb_prob
        a, b = np.zeros_like(z) + eps, np.ones_like(z) - eps
        f = self.f_value(A)
        #pdb.set_trace()
        fun = lambda x : np.log(1-x) - np.log(x) + mu * x * f + st_ratio * z + lambdas[0] * A + lambdas[1] * (1 - A)
        while np.any(abs(a - b) > eps):
            c = (a + b) / 2
            fc = fun(c)
            #print("p: [{:.3f}, {:.3f}] c {:.4f}: {:.4f}".format(a[0],b[0],c[0],fc[0]))
            a = np.where(fc >= 0, c , a)
            b = np.where(fc <= 0, c , b)
            #pdb.set_trace()
        return (a + b) / 2

    def compute_p_and_q(self, theta, lambdas, mu1 , mu2, X, A, st_ratio):
        mu2 = 0 # ignored
        z = _dot_intercept(theta,X)
        n = X.shape[0]
        
        p = self._solve_p_binary_search(mu1,A,st_ratio,z,lambdas)
            
        q = p / (1 - p * (1 - p) * mu1 * self.f_value(A))
        q = np.maximum(self.lb_prob, np.minimum(q, 1-self.lb_prob))
        return p,q
    def compute_lambda_loss_grad(self,lambdas, theta, mu1 ,mu2, X_trg,A_trg,st_ratio):
        if not self.trg_grp_marginal_matching: # or self.mu1 == 0: # inactive fairness
            return 0, 0
        C = 0
        p , q = self.compute_p_and_q(theta,lambdas,mu1, mu2, X_trg,A_trg,st_ratio)
        n = q.shape[0]
        g1 = np.dot(q,A_trg) / n - self.trg_group_estimator(1)  
        g0 = np.dot(q,1-A_trg) / n - self.trg_group_estimator(0)
        #z = _dot_intercept(theta,X_trg)
        #loss = - q * np.log(p) - (1 - q) * np.log(1 - p) + mu1 * p * q * self.f_value(A_trg) #+ z * (q - Y_trg)
        #loss = np.mean(loss) + g1 * lambdas[0] + g0 * lambdas[1] + .5 * C * np.dot(lambdas,lambdas)
        loss = g1 * lambdas[0] + g0 * lambdas[1] + .5 * C * np.dot(lambdas,lambdas)
        grad = np.array([g1, g0]).reshape((-1,))
        grad = grad + C * lambdas
        return loss, grad

    """
    This function computes loss and grad for passing to optimization function.
    Lambdas are concated to the returned grad
    """
    def compute_theta_loss_grad(self, theta, lambdas, mu1, mu2, X_src, Y_src, A_src, src_st_ratio):
        # The gradient is computed on src data 
        mu2 = 0
        #if self.trg_grp_marginal_feature:
        #    lambdas = theta[-2:]
        #    theta = theta[:-2]
        p,q = self.compute_p_and_q(theta,lambdas,mu1,mu2,X_src,A_src,src_st_ratio)
        n = X_src.shape[0]
        z = _dot_intercept(theta,X_src)
        #loss = trg_density * ( - q * np.log(p) - (1 - q) * np.log(1 - p) + mu * p * ( q * self.f_value(A,1) + (1 - q) * self.f_value(A,0)) + ratio * z * (q - Y))
        loss =  1 / src_st_ratio * ( - q * np.log(p) - (1 - q) * np.log(1 - p) + mu1 * p * q * self.f_value(A_src)) + z * (q - Y_src) #- abs(np.dot(q,A)/n - self.trg_group_estimator(1)) - abs(np.dot(q,1-A)/n - self.trg_group_estimator(0))
        loss = np.mean(loss)  + .5 * self.C * np.dot(theta,theta)

        grad = np.reshape(q - Y_src, (-1,1)) * X_src # todo no ratio
        
        grad = np.sum(grad,axis=0) / n + self.C * theta 
        
        #if self.trg_grp_marginal_feature:
        #ld_loss, _ = self.compute_lambda_loss_grad(self.lambdas,mu1,mu2,X_trg,A_trg,trg_st_ratio) # compute grp matching on target
        #    grad = grad.reshape((-1,1))
        #    ld_grad = ld_grad.reshape((-1,1))
        #    grad = np.vstack((grad, ld_grad))
        #    grad = grad.squeeze()
            #loss += ld_loss
        #grad += self.C * theta
        #pdb.set_trace()
        return loss,grad
    def grid_serach_mu(self,f, grid):
        fval = {}
        mu0,f0 = grid[0], f(grid[0])
        result = []
        fval[mu0] = f0
        for i,mu in enumerate(grid[1:]):
            _f = f(mu)
            fval[mu] = _f 
            print("mu0: [{:.3f}, {:.3f}] , mu1: [{:.3f}: {:.3f}]".format(mu0,f0,mu,_f))
            if _f * f0 < 0:
                result.append([mu0,mu])
            mu0,f0 = mu,_f
        if len(result) == 0 :
            if all(map(lambda x : x > 0, fval.values())):
                sorted_fval = sorted(fval.items(), key = lambda x : x[1])
            else:
                sorted_fval = sorted(fval.items(), key = lambda x : x[1],reverse=True)
            best = sorted_fval[:2]
            #mu1,f1 = sorted_fval.popitem()
            #pdb.set_trace()
            result.append([best[0][0],best[1][0]])
        return result

    def fit(self, X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu_range=[-1,1], IW_ratio_src = None):
        
        if not self.trg_group_estimator:
            self.build_trg_grp_estimator(X_src,A_src,Y_src,src_st_ratio,X_trg,A_trg,trg_st_ratio)

        if IW_ratio_src is not None:
            assert(IW_ratio_src.shape == src_st_ratio.shape)
            src_st_ratio = np.ones_like(src_st_ratio)
            trg_st_ratio = np.ones_like(trg_st_ratio)
            IW_ratio_src = IW_ratio_src.reshape((-1,1))
            #pdb.set_trace()
            X_src = IW_ratio_src * X_src
            self.trg_grp_marginal_matching = False 

        if np.isscalar(mu_range):
            self.mu1 = mu_range
        elif len(mu_range) < 2:
            self.mu1 = mu_range[0]
        else:
            mu0 = mu_range[0]
            mu1 = mu_range[1]
            def f(mu):
                self._fit_given_mu(X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu)
                return self.q_fairness_violation(X_trg,A_trg,trg_st_ratio, mu) #compute violation on trg
            self.zero_regions = self.grid_serach_mu(f,np.arange(mu0,mu1+.1,.1))            
            print(self.zero_regions)
            for r in self.zero_regions:
                a,b = r[0],r[1]
                fa = f(a)
                fb = f(b)
                if fa > fb:
                    a,b = b,a
                if fa * fb > 0:
                    print("mu: [{:.3f}, {:.3f}] {:.4f}: {:.4f}".format(a,b,fa,fb))
                    print("mu range is all positive or negative {}, left boundary would be returned.")
                    self.mu1 = mu0
                elif fa == 0:
                    self.mu1 = mu0
                elif fb == 0: 
                    self.mu1 = mu1
                else:
                    while abs(a - b) > 1e-4:
                        c = (a + b) / 2
                        fc = f(c)
                        print("mu: [{:.3f}, {:.3f}] c {:.4f}: {:.4f}".format(a,b,c,fc))
                        if abs(fc) < 1e-3:
                            break 
                        elif fc < 0 :
                            a = c
                        elif fc > 0:
                            b = c
                        else:
                            break
                    self.mu1 = c
        self._fit_given_mu(X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, self.mu1) 
        return self
    
    def _fit_given_mu(self,X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu):
        #if self.trg_grp_marginal_feature:
        #    X_src = self.build_trg_grp_marginal_feature(X_src,A_src,src_st_ratio)
        X_src = np.hstack((X_src,np.ones((X_src.shape[0],1))))
        X_trg = np.hstack((X_trg,np.ones((X_trg.shape[0],1))))
        #if np.isscalar(mu_range):
        #    self.mu1 = mu_range
        #elif len(mu_range) < 2:
        #    self.mu1 = mu_range[0]
        #else:
        #    self.mu1 = self.find_best_mu(mu_range,X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio)
        #self.theta = self.find_theta(self.mu, 0, X_src,Y_src,A_src, src_st_ratio)
        theta = self.find_theta(mu,0,X_src,Y_src,A_src,src_st_ratio)
        #lambdas0 = self.find_lambda_with_grad(mu, 0, X_trg,A_trg,trg_st_ratio)
        #theta0 = self.find_theta_with_grad(self.mu1, 0, X_src,Y_src,A_src,src_st_ratio)
        epoch = 1
        while True:
            self.theta,self.lambdas = self.find_theta_lambda_with_grad(mu,0,X_src,Y_src,A_src,src_st_ratio,X_trg,A_trg,trg_st_ratio)    
            _,l_g = self.compute_lambda_loss_grad(self.lambdas,self.theta,mu,0,X_trg,A_trg,trg_st_ratio)
            _,t_g = self.compute_theta_loss_grad(self.theta,self.lambdas,mu,0,X_src,Y_src,A_src,src_st_ratio)
            #pdb.set_trace()
            if np.linalg.norm(l_g) < self.tol and np.linalg.norm(t_g) < self.tol:
                print("Converged |lambda| {:.7f}, |theta| {:.7f}".format(np.linalg.norm(l_g),np.linalg.norm(t_g)))
                break 
            elif epoch >= self.max_epoch:
                print("Not converged |lambda| {:.7f}, |theta| {:.7f}".format(np.linalg.norm(l_g),np.linalg.norm(t_g)))
                break
            epoch += 1
        return self

    def fairness_violation(self, X,Y,A, ratio):
        proba = self.predict_proba(X,A, ratio)
        #pdb.set_trace()
        #return abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)])) 
        return np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)]) 
        #return  abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)]))  \
        #    +   abs(np.mean(proba[np.logical_and(Y == 0, A == 1)]) - np.mean(proba[np.logical_and(Y == 0, A == 0)]))
    
    def _q_fairness_violation(self,p,q,A):
        #p, q = self.compute_p_and_q(theta,mu,X,A,ratio)
        #pdb.set_trace()
        return np.dot(p * q, self.f_value(A)) / p.shape[0]
        #return np.dot(p*q, A) / np.dot(q,A) - np.dot(p*q, 1-A) / np.dot(q,1-A)
        #return ((np.sum((p * q)[A == 1]) / self.trg_group_estimator(1,1)) - (np.sum((p * q)[A == 0]) / self.trg_group_estimator(0,1))) / p.shape[0]
        #return ((np.dot(p * q,A) / self.trg_group_estimator(1)) - (np.dot(p * q,1-A) / self.trg_group_estimator(0))) / p.shape[0]
    def q_fairness_penalty(self,X,A,ratio):
        p, q = self.compute_p_and_q(self.theta,self.lambdas,self.mu1,0,X,A,ratio)
        return self.mu1 * self._q_fairness_violation(p,q,A)
    
    def build_trg_grp_marginal_feature(self,X,A,st_ratio):
        a1 = np.where(A == 1, 1 / st_ratio, 0).reshape((-1,1))
        a0 = np.where(A == 0, 1 / st_ratio, 0).reshape((-1,1))
        return np.hstack((X,a1,a0))
    def q_objective(self,X,A,st_ratio,mu1 = None, mu2 = None):
        if mu1 is None:
            mu1 = self.mu1
        p,q = self.predict_p_q(X,A,st_ratio,self.theta,self.lambdas,mu1,mu2)
        return np.mean( - q * np.log(p) - (1-q) * np.log(1 - p)) + mu1 * self._q_fairness_violation(p,q,A) + self.compute_lambda_loss_grad(self.lambdas,self.theta,self.mu1,0,X,A,st_ratio)[0]
    
    def positive_rate(self,X,A,st_ratio):
        p = self.predict_proba(X,A,st_ratio)
        p1 = np.dot(p,A) / p.shape[0]
        p0 = np.dot(p,1-A) / p.shape[0]
        return p1,p0

    def q_positive_rate(self,X,A,st_ratio):
        p,q = self.predict_p_q(X,A,st_ratio)
        p1 = np.dot(p * q,A) / p.shape[0]
        p0 = np.dot(p * q,1-A) / p.shape[0]
        return p1,p0
    
    def build_trg_grp_estimator(self,X_src,A_src,Y_src,src_ratio, X_trg,A_trg,trg_ratio):
        #estimator = build_trg_grp_true_estimator(A_src,Y_src) # not used
        estimator = lambda a : 1 # not used
        h = EOPP_fair_covariate_logloss_classifier(tol=1e-5,max_iter=200, trg_group_estimator= estimator, C=self.C, random_initialization=False, verbose=False, trg_grp_marginal_matching=False)
        h.fit(X_src,Y_src,A_src, src_ratio, X_trg, A_trg, trg_ratio, mu_range=[0]) # build a covaritae shift model with ignored fairness
        p = h.predict_proba(X_trg,A_trg, trg_ratio) # A is ignored as attribute but is included in X
        p1 = np.dot(p, A_trg.astype('int')) / A_trg.shape[0]
        p0 = np.dot(p, 1- A_trg.astype('int')) / A_trg.shape[0]
        #p1 = np.mean(np.logical_and(A_trg == 1 , Y == 1))
        #p0 = np.mean(np.logical_and(A_trg == 0 , Y == 1))
        print("p1 : {:.4f}, p0 : {:.4f}".format(p1,p0))
        #pdb.set_trace()
        estimator = lambda a : p1 if (a == 1) else p0 if (a == 0) else 0
        self.trg_group_estimator = estimator
        return estimator    
      

class DP_fair_covariate_logloss_classifier(fair_covariate_logloss_classifier):
    def __init__(self, trg_group_estimator, tol=1e-8, verbose=True, max_iter=10000, C = .1, random_initialization=False):
        super().__init__( tol = tol, verbose = verbose, max_iter = max_iter, C = C, random_initialization= random_initialization)
        self.trg_group_estimator = None
        self.mu2 = 0
        
    def f_value(self,A):
        f_val = np.where(A == 1, 1 / np.mean(A == 1) , -1 / np.mean(A == 0))
        #grp2 = np.where(A == 0, -1 / np.mean(A == 0), 0)
        #f_val= grp1 + grp2
        return f_val
    
    def compute_p_and_q(self, theta, mu1, mu2, X, A, st_ratio):
        mu2 = 0 # ignored
        z = _dot_intercept(theta,X)
        p = np.exp(_log_logistic(st_ratio * z))
        q =  p * (1 - p) * mu1 * self.f_value(A) + p 
        q = np.maximum(0, np.minimum(q, 1))
        return p,q

    def compute_loss_grad(self, theta, mu, mu2, X, A, Y, st_ratio):
        # The gradient is computed on src data 
        p,q = self.compute_p_and_q(theta,mu,X,A,st_ratio)
        n = X.shape[0]
        z = _dot_intercept(theta,X)
        logp = _log_logistic(st_ratio * z)
        log1_p = _log_logistic(- st_ratio * z)
        loss =  1 / st_ratio * ( - q * logp - (1 - q) * log1_p + mu * p * self.f_value(A)) + z * (q - Y)
        #loss =  1 / st_ratio * ( - q * np.log(p) - (1 - q) * np.log(1 - p) + mu * p * self.f_value(A)) + z * (q - Y)
        loss = np.mean(loss)  + .5 * self.C * np.dot(theta,theta)

        grad = np.reshape(q - Y, (-1,1)) * X # todo no ratio
        grad = np.sum(grad,axis=0) / n + self.C * theta 
        #pdb.set_trace()
        return loss,grad
    
    def fit(self,X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu_range=[-1,1]):
        if np.isscalar(mu_range):
            self.mu1 = mu_range
        elif len(mu_range) < 2:
            self.mu1 = mu_range[0]
        else:
            self.mu1 = self.find_best_mu(mu_range,X_src,Y_src,A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio)
        self.theta = self.find_theta(self.mu1,0,X_src,Y_src,A_src, src_st_ratio)
        return self

    def fairness_violation(self, X,Y,A, ratio):
        proba = self.predict_proba(X,A, ratio)
        #pdb.set_trace()
        #return abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)])) 
        return np.mean(proba[A == 1]) - np.mean(proba[A == 0]) 
        #return  abs(np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(proba[np.logical_and(Y == 1, A == 0)]))  \
        #    +   abs(np.mean(proba[np.logical_and(Y == 0, A == 1)]) - np.mean(proba[np.logical_and(Y == 0, A == 0)]))
    
    #def _q_fairness_violation(self,X,Y,A, theta, mu, ratio):
    def _q_fairness_violation(self,p,q,A):
        #p, q = self.compute_p_and_q(theta,mu,X,A,ratio)
        #pdb.set_trace()
        return ((np.dot(p,A) / np.mean(A)) - (np.dot(p,1-A) / np.mean(1-A))) / p.shape[0]
        #return ((np.sum((p * q)[A == 1]) / self.trg_group_estimator(1,1)) - (np.sum((p * q)[A == 0]) / self.trg_group_estimator(0,1))) / p.shape[0] 
    def expected_logloss(self,X,Y,A,st_ratio):
        p,q = self.predict_p_q(X,A,st_ratio)
        print("P :min {:.4f}, max {:.4f}, < .5 {:.4f}, avg {:.4f}".format(np.min(p), np.max(p), np.sum(np.less(p,.5)), np.mean(q)))
        print("Q: min {:.4f}, max {:.4f}, < .5 {:.4f}, avg {:.4f}".format(np.min(q), np.max(q), np.sum(np.less(q,.5)), np.mean(q)))
        z = _dot_intercept(self.theta,X)
        logp = _log_logistic(st_ratio * z)
        log1_p = _log_logistic(- st_ratio * z) 
        return np.mean( - Y * logp - (1-Y) * log1_p) 
        
    def q_objective(self,X,A,st_ratio,mu1 = None):
        if mu1 is None:
            mu1 = self.mu1
        p,q = self.predict_p_q(X,A,st_ratio,self.theta,mu1,mu2)
        z = _dot_intercept(self.theta,X)
        logp = _log_logistic(st_ratio * z)
        log1_p = _log_logistic(- st_ratio * z)         
        return np.mean( - q * logp - (1-q) * log1_p) + mu1 * self._q_fairness_violation(p,q,A)