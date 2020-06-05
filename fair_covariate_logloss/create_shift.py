import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
import pdb

def create_shift(data, flag, src_split = .5, mean_a = 2, std_b = 1, feature_ind = 0, threshold = 0.5, p_s = 0.7, p_t = 0.5):
    '''
    create shift according to the flag
    flag = 1, based on PCA and 1d gaussian
    flag = 2, based on sensitive feature, a (using \mathbb{E}[p(a)] = 0.7 as example)

    data: [m, n+1] with label at the last dimension
    flag: "PCA" or "Feature"
    mean_a, std_b: the parameter that distorts the gaussian used in sampling
                   according to the first principle n_components
    feature_ind, threshold, p_s, p_t: feature_ind for sensitive features, threshold and 
                                      source and target probabilities, respectively.
                                      For example, default setting is, 
                                      \mathbb{E}_{p_s}[p(a_0 > 0.5)] = 0.7 
                                      \mathbb{E}_{p_t}[p(a_0 > 0.5)] = 0.5 

    output: if PCA, [m/2, n+1] as source, [m/2, n+1] as target
    '''
    #features = data[:, 0:-2]
    #labels = data[:, -1]
    m = np.shape(data)[0]
    source_size = int(m * src_split)
    target_size = m - source_size
    if flag is 'pca':
    #PCA
        pca = PCA(n_components=1)
        pc = pca.fit_transform(data)
        # or use certain feature dimension to sample
        #pc = data[:,0]
        print(pc)
        pc_mean = np.mean(pc)
        pc_std = np.std(pc)

        sample_mean = pc_mean/mean_a
        sample_std = pc_std/std_b

        print(sample_mean)
        print(pc_mean)
 
        # sample according to the probs
        prob_s = norm.pdf(pc, loc = sample_mean, scale = sample_std)
        sum_s = np.sum(prob_s)
        prob_s = prob_s/sum_s
        prob_t = norm.pdf(pc, loc = pc_mean, scale = pc_std)
        sum_t = np.sum(prob_t)
        prob_t = prob_t/sum_t
        # test = np.random.choice(range(m), 2, replace = False, p = prob_s)
        print(prob_s)
        source_ind = np.random.choice(range(m), size = source_size, replace = False, p = np.reshape(prob_s, (m)) )
        target_ind = np.random.choice(range(m), size = target_size, replace = False, p = np.reshape(prob_t, (m)) )
        source_data = data[source_ind, :]
        target_data = data[target_ind, :]


        ratios = prob_s/prob_t
        source_ratios = ratios[source_ind]
        target_ratios = ratios[target_ind]
        #pdb.set_trace()
    else:
        source_ind = []
        target_ind = []
        prob_s = np.zeros(m)
        prob_t = np.zeros(m)
        # sample according to a's value
        a_value = data[:, feature_ind]
        for i in range(m):
            # if equality, change to "=="
            if a_value[i] > threshold:
                rand = np.random.uniform(0, 1, 1)
                prob_s[i] = p_s
                if rand < p_s:
                    source_ind.append(i)
                     
            else:
                rand = np.random.uniform(0, 1, 1)
                prob_s[i] = 1 - p_s
                if rand > p_s:
                    source_ind.append(i)
                    

        for i in range(m):
            # if equality, change to "=="
            if a_value[i] > threshold:
                rand = np.random.uniform(0, 1, 1)
                prob_t[i] = p_t
                if rand < p_t:
                    target_ind.append(i)
                    
            else:
                rand = np.random.uniform(0, 1, 1)
                prob_t[i] = 1 - p_t
                if rand > p_t:
                    target_ind.append(i)
                    

        source_data = data[source_ind, :]
        target_data = data[target_ind, :]
        #print(np.shape(source_data))
        #print(np.shape(target_data))

        

        ratios = prob_s/prob_t

        source_ratios = ratios[source_ind]
        target_ratios = ratios[target_ind]

        # second uniform sampling according to size
         
        #source_size_ind_unif = np.random.choice(range(source_data.shape[0]), size = source_size, replace = False)
        #target_size_ind_unif = np.random.choice(range(target_data.shape[0]), size = target_size, replace = False)

        #source_data = source_data[source_size_ind_unif, :]
        #target_data = target_data[target_size_ind_unif, :]

        #source_ratios = ratios[source_size_ind_unif]
        #target_ratios = ratios[target_size_ind_unif]
         
        
    print(np.shape(source_data))
    print(np.shape(target_data))
    print(source_ratios)
    print(target_ratios)
    #return source_data, target_data, source_ratios, target_ratios, source_t_prob, target_t_prob, source_ind, target_ind
    return source_ind, target_ind, np.squeeze(ratios)

# def estimate_ratios(source_data, target_data, flag ):





def main():
    # load data
    data = np.genfromtxt("ecoli.csv", delimiter=",")
    print(data)
    print(np.shape(data))

    create_shift(data, 'Feature')



if __name__ == '__main__':
    main()