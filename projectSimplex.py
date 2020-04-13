def projectSimplex(v):
#Computest the minimum L2-distance projection of vector v onto the probability simplex
    nVars = len(v)
    mu = sorted(v, reverse=True) #sort(v,'descend')
    sm = 0
    row, sm_row = 0, 0 #### Omid! I don't know!!!
    for j in range(nVars):
        sm = sm+mu[j]
        if (mu[j] - (1/(j+1))*(sm-1)) > 0:
            row = j
            sm_row = sm
    
    theta = (1/(row+1))*(sm_row-1)
    w = [max(v[i]-theta,0) for i in range(len(v))]#max(v-theta,0)
    return w
