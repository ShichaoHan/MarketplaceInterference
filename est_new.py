dim_cate_results = {} 
for outcome_name in outcomes_names:
    for target_name in target_names:
        o_name = outcome_name.replace("_", "")
        cat_name =  target_name.replace("result_", "").replace("_fold", "")
        
        
        Q_seller = df_seller.shape[0]
        Q_twoside = df_twoside.shape[0]
        
        df_seller_cat = df_seller.loc[df_seller[cov_col]== cat_name, :]
        df_twoside_cat = df_twoside.loc[df_twoside[cov_col]==cat_name,:]
        
        T_twoside, C_twoside = np.array(df_twoside_cat.loc[df_twoside_cat['t_indicator']==1,o_name]), np.array(df_twoside_cat.loc[df_twoside_cat['t_indicator']==0,o_name])
        
        # T_sellerside, C_sellerside = np.array(df_seller_cat.loc[df_seller_cat['t_indicator']==1,o_name]), np.array(df_seller_cat.loc[df_seller_cat['t_indicator']==0,o_name])
        
        Y_t, Y_c = df_twoside.loc[df_twoside['t_indicator']==1, o_name],df_twoside.loc[df_twoside['t_indicator']==0, o_name]
        J_t, J_c = df_twoside.loc[df_twoside['t_indicator']==1, 'quota_level']==cat_name,df_twoside.loc[df_twoside['t_indicator']==0, 'quota_level'] == cat_name
        #dim_point_twoside, var_point_twoside = dim_est_IPW(T_twoside, C_twoside, 0.5, Q_twoside)
        dim_point_twoside, stderr_twoside = global_cate(Y_t, J_t, Y_c, J_c)
        dim_point_seller, var_point_seller = dim_est_IPW(T_sellerside, C_sellerside, 0.5, Q_seller)
        dim_cate_results[(outcome_name, target_name)] = {'twoside_point' : dim_point_twoside, 
                                                         'twoside_stderr': stderr_twoside,
                                                        'seller_point' : dim_point_seller, 
                                                        'seller_stderr':np.sqrt(var_point_seller / Q_seller)}