import numpy as np
from numpy import linalg
from projectSimplex import projectSimplex

def projectBistochasticADMM(X, Z_init):
    #Project to Bistochastic matrix
    #   Using ADMM 
    #   X : Input Matrix 
    #   Z_init : initialize solution

    n = len(X)

    # initialize
#    if nargin > 1:  ##############3
#        Z = Z_init
#        Y = Z_init
#    else:
#        Z = np.random.rand(n,n) #rand(n,n)
#        Y = np.random.rand(n,n) #rand(n,n)
    
    Z = np.array(Z_init)
    Y = np.array(Z_init)
    W = np.zeros((n,n))

    # history
    Y_old = Y

    # penalty parameters
    par_increase = 2.0
    par_decrease = 2.0
    par_mu = 10.0

    # tolerance
    max_iter = 100 
    tol_abs = 1e-4
    tol_rel = 1e-2

    # step size init
    rho = 1.0

    iter = 1
    while True:
        A = (X + rho * (Y - W)) / (1.0 + rho)
        # update Z -> column wise projection 
        for i in range(n):
            Z[i,:] = projectSimplex(A[i,:]) #Z(:, i) = projectSimplex(A(:,i))


        B = (X + rho * (Z + W)) / (1.0 + rho)
        # update U -> row wise projection 
        for i in range(n):
            Y[:, i] = projectSimplex(B[:, i])

        W = W + Z - Y

        # stopping criteria
        norm_r = linalg.norm(Z - Y, 'fro') #norm(Z - Y, 'fro')
        norm_s = linalg.norm(rho * (Y - Y_old), 'fro') #norm(rho * (Y - Y_old), 'fro')
        eps_primal = n * tol_abs + tol_rel * max(linalg.norm(Z, 'fro'), linalg.norm(Y, 'fro'))   # sqrt(n*n) = n
        eps_dual = n * tol_abs + tol_rel * linalg.norm(rho * W, 'fro')

        # update rho
        if norm_r > (par_mu * norm_s):
            rho = par_increase * rho
        elif norm_s > (par_mu * norm_r):
            rho = rho / par_decrease
            

        if iter >= max_iter:
            break
        elif norm_r < eps_primal and norm_s < eps_dual:
            break

        Y_old = Y
        iter = iter + 1

    #fprintf('# iter : %d\n', iter)
    return Z