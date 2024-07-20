outcomes_names= ['outcome1_', 'outcome2_','outcome3_','outcome4_']
target_names = ['result_f5_101到500元_fold','result_f3_21到50元_fold','result_f4_51到100元_fold','result_f1_<=10元_fold']
cate_results_new = {} 
for outcome_name in outcomes_names:
    for target_name in target_names:
        target_covs_file_name = target_name
        hfuncs_each_fold = {} 
        debias_terms_each_fold = {} 
        cnt = 0 
        for data_file_name in data_file_names:
            if (target_covs_file_name in data_file_name) and (outcome_name in data_file_name):
                print(data_file_name)
                with open(f"prediction_result_new/{data_file_name}", 'rb') as ff:
                    LOADED_OBJECT = pickle.load(ff)
                hfuncs_, debias_terms_ = LOADED_OBJECT 
                hfuncs_each_fold[cnt] = hfuncs_ 
                debias_terms_each_fold[cnt] = debias_terms_
                cnt += 1 

        n_folds = len(hfuncs_each_fold.keys())
        samp_size = np.sum([len(hfuncs_each_fold[elm]) for elm in range(n_folds)])
        tau_hat_undebias = np.mean([ np.mean(hfuncs_each_fold[f])for f in range(n_folds)])
        tau_hat_debias = np.mean([ np.mean(hfuncs_each_fold[f] - debias_terms_each_fold[f])  for f in range(n_folds)])
        debias_point = tau_hat_debias
        debias_var = np.mean([debias_estimator_new(hfuncs_each_fold[f] ,debias_terms_each_fold[f], tau_hat_debias)[1] for f in range(n_folds)])
        undebias_var = np.mean([undebias_estimator_new(hfuncs_each_fold[f] ,tau_hat_undebias)[1] for f in range(n_folds)])
        undebias_point = tau_hat_undebias

        cate_results_new[(outcome_name, target_name)] = {'debias_point' : debias_point, 'debias_stderr': np.sqrt(debias_var / samp_size)}
    print("outcome ", outcome_name)