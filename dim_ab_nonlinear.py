import pandas as pd 
import numpy as np 

import random
import time 
from joblib import Parallel, delayed
from utils import *
from model_utils import *

import argparse

parser = argparse.ArgumentParser(description='Process Settings')
parser.add_argument(
    '-e',
    '--exp_name',
    type=str,
    default='exp',
    help='name of experiment')
parser.add_argument(
    '--J',
    type=int,
    default=500,
    help='number of items')
parser.add_argument(
    '--K',
    type=int,
    default=3,
    help='size of consideration set')
parser.add_argument(
    '--Q',
    type=int,
    default=1000,
    help='number of queries')
parser.add_argument(
    '--M',
    type=int,
    default=500,
    help='number of MC for hessian')
parser.add_argument(
    '--n_folds',
    type=int,
    default=3,
    help='number of crossfitting folds')
parser.add_argument(
    '--uplift_factor',
    type=float,
    default=0.3,
    help='uplift factor')
parser.add_argument(
    '-s',
    '--n_sims',
    type=int,
    default=100,
    help='number of simulations')
parser.add_argument(
    '--noise_std',
    type=float,
    default=0.1,
    help='outcome noise std')
parser.add_argument(
    '--epochs',
    type=int,
    default=500,
    help='number of simulations')

args = parser.parse_args()



J = args.J 
K = args.K
Q = args.Q 
epochs = args.epochs
noise_std = args.noise_std
uplift_factor = args.uplift_factor
L = 1
d = 2

np.random.seed(0)

item_embeddings = np.random.uniform(size=(J, d))
item_promotion = np.random.choice([1, 0], size=J)

np.random.seed(int(time.time() * 1e8 % 1e8))

L = 1

M = args.M ## Number of iterations for Hessian matrix estimation 
n_folds = args.n_folds

## Modifying the tensor for 3d input 



begin_time = time.time()

def exp(s):
    user_embeddings = np.random.uniform(size=(Q, d))
    query_matrix = []
    for each_query in range(Q):
        selected_indices = np.random.choice(np.arange(J), size = K, replace= False)
        query_matrix += [selected_indices]
    query_matrix = np.array(query_matrix)
    true_estimate, true_stderr = get_ground_truth(uplift_factor, item_embeddings, 
                                                  item_promotion, user_embeddings, query_matrix)
    
    (promotions, embeddings, W_matrix, outcome_potential, 
     exposure_matrix) = DGP_new_heterogeneous_embeddings(uplift_factor, item_embeddings, 
                                                         item_promotion, user_embeddings, 
                                                         query_matrix, treated_probability=0.5, noise_std=noise_std)
    observed_outcome = np.sum(outcome_potential * exposure_matrix, axis = 1 )
    observed_queries_treatment = np.sum(exposure_matrix * W_matrix, axis = 1 )
    T = observed_outcome[observed_queries_treatment == 1]
    C = observed_outcome[observed_queries_treatment == 0] 
    dim_point, dim_var = dim_est(T, C, 0.5, Q)
        
    
    ## Cross-fitting indices 
    all_inds = generate_indices(Q, n_folds)

    ## Iterate over each fold for cross-validation. 
    hfuncs_each_fold,  debias_terms_each_fold = {}, {}
    loss_each_fold = {}
    for f in range(n_folds):
        f_start, f_end = all_inds[f]
        f_size = f_end - f_start
        
        ## Cross-fitting
        promotions_train, promotions_test =  train_test_split(promotions, all_inds, f) 
        embeddings_train, embeddings_test =  train_test_split(embeddings, all_inds, f) 
        W_matrix_train, W_matrix_test = train_test_split(W_matrix, all_inds, f)  
        exposure_matrix_train, exposure_matrix_test =train_test_split(exposure_matrix, all_inds, f) 
        observed_outcome_train, observed_outcome_test = train_test_split(observed_outcome, all_inds, f)
        
        inputs_3d_train = np.concatenate([embeddings_train, promotions_train[:, :, np.newaxis], 
                                          W_matrix_train[:, :, np.newaxis], exposure_matrix_train[:, :, np.newaxis]], axis = -1)
        inputs_3d_test = np.concatenate([embeddings_test, promotions_test[:, :, np.newaxis], W_matrix_test[:, :, np.newaxis], 
                                         exposure_matrix_test[:, :, np.newaxis]], axis = -1)
        output_3d_train = np.concatenate([exposure_matrix_train.astype(dtype=float), observed_outcome_train[:, np.newaxis]], axis = 1)

        myModelMultiple = MyModel_embeddings(K, d, 1)
        myModelMultiple.compile(loss=custom_loss, optimizer=tf.keras.optimizers.legacy.Adam())
        history_f = myModelMultiple.fit(inputs_3d_train, output_3d_train, epochs=epochs, verbose=0)
        
        ## Store the training history 
        
        loss_each_fold[f] = history_f
        predict_p_test, _, _, predict_outcome_test = np.split(myModelMultiple.predict(inputs_3d_test, verbose=0), [K, 2*K, 2*K+1], axis=1)

        input_3d_test_treat = np.concatenate([embeddings_test, promotions_test[:, :, np.newaxis], np.ones_like(W_matrix_test)[:, :, np.newaxis], 
                                              exposure_matrix_test[:, :, np.newaxis]], axis = -1)
        input_3d_test_control = np.concatenate([embeddings_test, promotions_test[:, :, np.newaxis], np.zeros_like(W_matrix_test)[:, :, np.newaxis], 
                                                exposure_matrix_test[:, :, np.newaxis]], axis = -1)
        
        predict_p_treat, _, _, predict_outcome_treat = np.split(myModelMultiple.predict(input_3d_test_treat, verbose=0), [K, 2*K, 2*K+1], axis=1)
        predict_p_control, _, _, predict_outcome_control = np.split(myModelMultiple.predict(input_3d_test_control, verbose=0), [K, 2*K, 2*K+1], axis=1)
        

        ## 1. COMPUTE THE GRADIENT OF LOSSS  
        gradient_vector_l = compute_loss_gradient(predict_p_test, exposure_matrix_test, W_matrix_test, 
                                                  predict_outcome_test, observed_outcome_test)




        ## 2. COMPUTE  THE GRADIENT OF H FUNCTION
        gradient_vector_H = compute_value_gradient(predict_p_treat, predict_outcome_treat, predict_p_control, predict_outcome_control)

        
        # 3. FIND THE EXPECTATION OF HESSIAN MATRIX 
        Hessian_all = np.zeros((f_size, (L+2) * K - 1,  (L+2) * K - 1))
        for m in range(M):
            treat_dict_m = permute_treatment_dict(J, L)
            W_matrix_m = []
            for each_query in query_matrix[f_start:f_end]:
                W_matrix_m.append([treat_dict_m[ind] for ind in each_query])
            W_matrix_m = np.array(W_matrix_m)
            inputs_m = np.concatenate([embeddings_test, promotions_test[:, :, np.newaxis], W_matrix_m[:, :, np.newaxis], exposure_matrix_test[:, :, np.newaxis]], axis = -1)
            predict_p_m, _, _, _ = np.split(myModelMultiple.predict(inputs_m, verbose=0), [K, 2*K, 2*K+1], axis=1)
            Hessian = compute_hessian_instance(W_matrix_m, predict_p_m)
            Hessian_all = Hessian_all + Hessian
        Hessian_final = Hessian_all / M
        
        count_finite = 0
        debias_term_f = np.zeros(len(Hessian_final))
        for i in range(f_size):
            if is_invertible(Hessian_final[i]):
                try:
                    debias_term_f[i] = gradient_vector_H[i]@np.linalg.inv(Hessian_final[i])@gradient_vector_l[i]
                    count_finite += 1 
                except: 
                    print("Fail for inversion")


        ## END OF FOR LOOP FOR EACH ITERATION OVER CROSS FITTING
        hfuncs_each_fold[f] = np.sum(predict_p_treat * predict_outcome_treat, axis=1) - np.sum(predict_p_control * predict_outcome_control, axis=1)
        debias_terms_each_fold[f] = debias_term_f
        
    (debias_point, debias_var, undebias_point, undebias_var) = crossfitted_estimate_var(hfuncs_each_fold, debias_terms_each_fold)
    
    path = compose_filename(f"results/{args.exp_name}_synthetic_ab", "csv")
    result_df = pd.DataFrame({"debias_point": [debias_point], "debias_var":[debias_var], "dim": [dim_point], 
                              "dim_var":[dim_var], "undebias_point": [undebias_point], "undebias_var": [undebias_var], 
                              "J" : [J], "Q": [Q],  "K":[K],  "M": [M], "epochs" :[epochs], "n_folds": [n_folds], "uplift_factor": [uplift_factor],
                              "truth": [true_estimate], "truth_stderr": [true_stderr] })
    # result_df.to_csv(path)
    print("finish simulation.")

    result_df.to_csv(path)

for s in range(args.n_sims):
    exp(s)

print(f"Finished in {time.time() - begin_time} seconds.")
