import pandas as pd 
import numpy as np 

def logistic_row(row):
    return np.exp(row) / np.sum(np.exp(row))

def dim_est(obs_T, obs_C, treated_probability, Q):
    n1,n0 = len(obs_T), len(obs_C)
    tau1 = np.sum(obs_T) / (Q*treated_probability)
    tau0 = np.sum(obs_C)/(Q * (1-treated_probability))
    estimate = tau1 - tau0
    var = (np.sum((obs_T / treated_probability - estimate) ** 2) + np.sum(( - obs_C / (1-treated_probability) - estimate) ** 2)) / Q
    return estimate, var

def DGP_new_heterogeneous(J, Q, K, promo_ratio, query_matrix, X_goodbads, X_utility, treated_probability=0.5, treat_control_pool = [True, False]):
    ## Randomize over the treatment assignment matrix 
    treatment_dict = {}
    for j in range(J):
        treatment_dict[j] = np.random.choice(treat_control_pool, 1, p=[treated_probability, 1 - treated_probability])

    W_matrix = []
    for each_query in range(Q):
         W_matrix = np.append(W_matrix, [treatment_dict[ind] for ind in query_matrix[each_query]])

    outcome_noise =  np.random.normal(size=(Q, K)) 
    W_matrix = W_matrix.reshape(Q,K)
    # W_matrix = W_matrix.reshape(Q,K)
    final_score_matrix = W_matrix * promo_ratio * X_goodbads   + X_utility

    X_logit = np.apply_along_axis(logistic_row, axis=1, arr=final_score_matrix)
    expose_indices = np.array([np.random.choice(np.arange(K), size = 1, p = X_logit[i,:]) for i in range(Q)])
    inddds = np.array(list(np.arange(K)) * Q).reshape(Q,K)
    exposure_matrix = np.array([inddds[i,:] == expose_indices[i] for i in range(Q)])

    ## Outcome model  
    ## First: a true outcome model of Exponential 
    outcome_potential = X_utility

    return query_matrix, X_goodbads, X_utility,W_matrix, exposure_matrix, outcome_potential, X_logit


def crossfitted_estimate_var(hfuncs_each_fold, debias_terms_each_fold):
    undebias_point = np.mean([np.mean(hfuncs_each_fold[f]) for f in hfuncs_each_fold])
    undebias_var = np.mean([np.mean((hfuncs_each_fold[f] - undebias_point) ** 2) for h in hfuncs_each_fold])
    debias_point = np.mean([np.mean(hfuncs_each_fold[f] - debias_terms_each_fold[f]) for f in hfuncs_each_fold])
    debias_var = np.mean([np.mean((hfuncs_each_fold[f] - debias_terms_each_fold[f] - undebias_point) ** 2) for f in hfuncs_each_fold])
    return debias_point, debias_var, undebias_point, undebias_var


def is_invertible(matrix):
    return np.linalg.det(matrix) != 0

    
def permute_treatment_dict(J, L):
    perm_dict = {}
    for j in range(J):
        perm_dict[j] = np.random.choice(L+1, 1)
    return perm_dict

## Helper function for cross validation
def generate_indices(n, K):
    ## Split original sample of size n into K sets 
    indices = np.linspace(0, n, K+1, dtype=int)
    return list(zip(indices[:-1], indices[1:]))


def train_test_split(input_data, all_inds, kth_test):
    
    training_ind = [all_inds[i] for i in range(len(all_inds)) if i != kth_test]
    test_start, test_end = all_inds[kth_test]
    if not tf.is_tensor(input_data):
        training_data = np.concatenate([input_data[elm[0]:elm[1]] for elm in training_ind])
    else:
        
        training_data = tf.concat([input_data[elm[0]:elm[1]] for elm in training_ind], axis = 0)
    testing_data = input_data[test_start:test_end]
    return training_data, testing_data 

def generate_environment(J = 30, K = 5, Q = 3000, uplift_factor = 1.0):
    """
    ## J: Number of videos 
    ## K: Consideration set size 
    ## Q: Generate some queries along with the recommendation model 
    ## Uplift factor 
    """
    
    utility_score_matrix = np.exp(np.random.normal(size=(Q,J)))
    
    good_bad_dict = {} 
    treatment_dict = {} 
    utility_score = {} 
    for j in range(J):
        good_bad_dict[j] = np.random.choice([True,False], 1)
        utility_score[j] = np.random.uniform()
    X_goodbads = []
    X_utility = []
    query_matrix = []
    for each_query in range(Q):
        ## Form the consideration set 
        selected_indices = np.random.choice(np.arange(J), K, replace= False)
        query_matrix += [selected_indices]
        X_goodbads = np.append(X_goodbads,[good_bad_dict[ind] for ind in selected_indices])
        X_utility = np.append(X_utility, [utility_score_matrix[each_query, ind] for ind in selected_indices])
    X_goodbads = X_goodbads.reshape(Q, K)
    X_utility = X_utility.reshape(Q, K)
    X_utility = X_utility + X_goodbads
    return X_utility, X_goodbads, np.array(query_matrix), utility_score_matrix, treatment_dict, utility_score, good_bad_dict

def find_ate_ground_truth(J, K, Q, uplift_factor):
    ground_truth = []
    for _ in range(100):
        (X_utility, X_goodbads, query_matrix, _, _, _, _) = generate_environment(J = J, K = K, Q = Q, uplift_factor = uplift_factor)
        (_, _, _, _, _, _, X_logit_T) = DGP_new_heterogeneous(J, Q, K, uplift_factor, query_matrix, 
                                                              X_goodbads, X_utility,  treat_control_pool = [True, True])
        (_, _, _, _, _, _, X_logit_C) = DGP_new_heterogeneous(J, Q, K, uplift_factor, query_matrix, 
                                                              X_goodbads, X_utility,  treat_control_pool = [False, False])
        T_gt = np.sum(X_logit_T * X_utility , axis = 1 )
        C_gt = np.sum(X_logit_C * X_utility , axis = 1 )
        ground_truth.append(np.mean(T_gt) - np.mean(C_gt))
    return np.mean(ground_truth), np.std(ground_truth) / np.sqrt(len(ground_truth))


def compose_filename(prefix, extension):
    """
    Creates a unique filename.
    Useful when running in parallel on Sherlock.
    """
    # Tries to find a commit hash
    try:
        commit = subprocess\
            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time.time() * 1e8 % 1e8))
    ident = filter(None, [prefix, commit, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname

def compute_loss_gradient(predict_p, exposure_matrix, treatment_matrix, predict_outcome, observed_outcome):
    dl1dtheta0 = predict_p - exposure_matrix
    dl1dtheta0 = dl1dtheta0[:, 1:] 
    dl1dtheta1 = treatment_matrix * (predict_p - exposure_matrix)
    dl2dmu = exposure_matrix * (predict_outcome - observed_outcome)
    gradient_vector_l = np.concatenate([dl1dtheta0, dl1dtheta1, dl2dmu], axis =1)
    return gradient_vector_l

def compute_value_gradient(predict_p_treat, predict_outcome_treat, predict_p_control, predict_outcome_control):
    Ey1 = np.sum(predict_p_treat * predict_outcome_treat, axis=1, keepdims=True)
    Ey0 = np.sum(predict_p_control * predict_outcome_control, axis=1, keepdims=True)
    dHdtheta0 = predict_p_treat * (predict_outcome_treat - Ey1) - predict_p_control * (predict_outcome_control - Ey0)
    dHdtheta0 = dHdtheta0[:, 1:]
    dHdtheta1 = predict_p_treat * (predict_outcome_treat - Ey1) 
    dHdmu = predict_p_treat - predict_p_control
    gradient_vector_H = np.concatenate([dHdtheta0, dHdtheta1, dHdmu], axis =1 )
    return gradient_vector_H

def compute_hessian_instance(W_matrix_m, exposure_matrix, predict_p_m):
    f_size = len(W_matrix_m)
    predict_p_1minusp_m = predict_p_m * (1 - predict_p_m)
    W_p_1minusp_m = W_matrix_m * predict_p_1minusp_m

    ## Off-diagonal terms 
    p_outer_p = np.array([np.outer(row_, row_) for row_ in predict_p_m])
    d2l2dtheta1 =  np.array([np.outer(row_, row_) for row_ in W_matrix_m]) * p_outer_p
    ## Modify diagonal terms
    for i in range(f_size):
        np.fill_diagonal(d2l2dtheta1[i], W_p_1minusp_m[i])
    
    d2l2dtheta0 = - np.array([np.outer(row_, row_) for row_ in predict_p_m])

    ## FIX: iterate over all l1, l2 
    d2ldthetal1dthetal2 = {} 
    for l in range(L):
        d2l2dtheta0dtheta1 = - W_matrix_m[:, np.newaxis, :] * p_outer_p
        for i in range(f_size):
            np.fill_diagonal(d2l2dtheta0dtheta1[i], W_p_1minusp_m[i])
        ## NOTE: -1 to indicate the baseline theta 
        d2ldthetal1dthetal2[(-1,l)] = d2l2dtheta0dtheta1[:,1:,:]
        d2ldthetal1dthetal2[(l,-1)] = np.transpose(d2l2dtheta0dtheta1[:,1:,:], (0,2,1))
        d2ldthetal1dthetal2[(l,l)] = d2l2dtheta1

    d2l2dmu = np.zeros(d2l2dtheta1.shape)
    for i in range(d2l2dmu.shape[0]):
        np.fill_diagonal(d2l2dtheta0[i], predict_p_1minusp_m[i, :])
        np.fill_diagonal(d2l2dmu[i], predict_p_m[i,:])

    d2l2dtheta0 = d2l2dtheta0[:,1:, 1:]
    Hessian_first_row = np.concatenate([d2l2dtheta0] + [d2ldthetal1dthetal2[(-1, l)] for l in range(L)] + [np.zeros((f_size, K-1, K))], axis=2)

    ## 1 to L + 1 row 
    Hessian_middle_dict = {}
    for l in range(L):
        row_l = np.concatenate([d2ldthetal1dthetal2[(l, -1)]] + [d2ldthetal1dthetal2[(l, l_prime)] for l_prime in range(L)] +[np.zeros((f_size, K, K))], axis=2)
        Hessian_middle_dict[l] = row_l                                                                           

    Hessian_third_row = np.concatenate((np.zeros((f_size, K, K  * (L + 1 ) - 1 )), d2l2dmu), axis =2)
    Hessian = np.concatenate([Hessian_first_row] + [Hessian_middle_dict[l] for l in range(L)] + [Hessian_third_row], axis = 1 )
    
    return Hessian