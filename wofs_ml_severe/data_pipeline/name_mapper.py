def name_mapping(previous_feature_names):
    """
    The goal of this new method is to convert feature names 
    saved with the ML model into the new feature names.
    """
    corrected_feature_names={}
    for f in previous_feature_names:
        if 'mb' in f and 'temperature' in f:
            new_f = f.replace('mb','')
        elif 'dewpoint' in f and 'mb' in f:
            new_f = f.replace('mb','').replace('dewpoint', 'td')
        elif 'geopotential' in f:
            new_f =f.replace('geopotential_height','geo_hgt').replace('mb', '')
        elif 'cloud_top_temp' in f:
            new_f =f.replace('cloud_top_temp','ctt')
        elif 'divergence' in f:
            new_f =f.replace('divergence_10m','div_10m')
        elif '10-m_bulk_shear' in f:
            new_f =f.replace('10-m_bulk_shear','10-500m_bulkshear')
        elif 'th_e_ml' in f:
            new_f =f.replace('th_e_ml','theta_e')
        elif '1to3km' in f or '3to5km' in f:
            new_f =f.replace('km_max','')
        elif 'bouyancy' in f:
            # Spelling error!
            new_f = f.replace('bouyancy','buoyancy')
        else:
            new_f = f
        
        # Additional '_' in the new format! 
        new_f = new_f.replace('_ens_mean', '__ens_mean')
        new_f = new_f.replace('_ens_std', '__ens_std')
        new_f = new_f.replace('_spatial_mean', '__spatial_mean')
        new_f = new_f.replace('_time_max', '__time_max')
        new_f = new_f.replace('_time_min', '__time_min')
        
        
        # New naming convention for amplitude stats! 
        new_f = new_f.replace('_ens_std_of_90th', '_amp_ens_std')
        new_f = new_f.replace('_ens_mean_of_90th', '_amp_ens_mean')
        new_f = new_f.replace('_ens_std_of_10th', '_amp_ens_std')
        new_f = new_f.replace('_ens_mean_of_10th', '_amp_ens_mean')
        
        #Baseline var
        #new_f = new_f.replace('uh_probs_>180_prob_max', 'uh_nmep_>180_3km__prob_max')
        #new_f = new_f.replace('hail_probs_>1.0_prob_max', 'hail_nmep_>1.0_3km__prob_max')
        #new_f = new_f.replace('wind_probs_>40_prob_max', 'wnd_nmep_>40_3km__prob_max')
        
        corrected_feature_names[f] = new_f

    return corrected_feature_names