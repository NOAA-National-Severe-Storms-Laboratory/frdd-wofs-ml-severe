MORPHOLOGICAL_FEATURES = [
                          'area',
                          'eccentricity',
                          'extent',
                          'orientation',
                          'minor_axis_length',
                          'major_axis_length',
                          'intensity_max',
                          'ens_track_prob'
                          ]



ENV_VARS =  [   'freezing_level',
              'theta_e',
              'u_10',
              'v_10',
              'temperature_850',
              'temperature_700',
              'temperature_500',
              'td_850',
              'td_700',
              'td_500',
              'mid_level_lapse_rate',
              'low_level_lapse_rate',
              'geo_hgt_850',
              'geo_hgt_500',
              'geo_hgt_700',
              'td_850',
              'td_700',
              'td_500',
              'QVAPOR_850',
              'QVAPOR_700',
              'QVAPOR_500',
              'qv_2',
             'srh_0to1',
             'srh_0to3',
             'cape_ml',
             'cin_ml',
             'shear_u_0to6',
             'shear_v_0to6',
             'shear_u_0to1',
             'shear_v_0to1',
             'shear_u_3to6',
             'shear_v_3to6',
             'stp', 
             'lcl_ml',
             'srh_0to500',
             'stp_srh0to500',
           ]

STORM_VARS  = [ 'uh_0to2_instant',
              'uh_2to5_instant',
              'wz_0to2_instant',
              'comp_dz',
              'ws_80',
              'w_up',
              'hailcast',
              'w_1km',
              'w_down',
              '10-500m_bulkshear',
              'div_10m',
              'buoyancy',
              'ctt',
              'dbz_3to5', 
              'dbz_1to3', 
              'dbz_1km', 
              'okubo_weiss',
]


map_to_readable_names={
                    'Run Date' : 'Run Date',
                    'Bias' : 'Bias',
                    'obj_centroid_x': 'X-comp of Object Centroid',
                    'obj_centroid_y': 'Y-comp of Object Centroid',
                    'area': 'Area',
                    'eccentricity': 'Eccentricity',
                    'extent' : 'Extent',
                    'orientation': 'Orientation',
                    'minor_axis_length': 'Minor Ax. Len.',
                    'major_axis_length': 'Major Ax. Len.',
                    'label': 'Object Label',
                    'ensemble_member': 'Ensemble Member',
                    'cape_0to3_ml':'0-3 km ML CAPE',
                    'cin_ml': 'ML CIN',
                    'cape_ml': 'ML CAPE',
                    'lcl_ml': 'ML LCL',
                    'u_10': '10-m U-wind comp.',
                    'v_10': '10-m V-wind comp.',
                    'th_e_ml': 'ML  $\\theta_{e}$',
                    'theta_e': 'ML $\\theta_{e}$',
                    'shear_u_3to6' : '3-6 km U-Shear',
                    'shear_v_3to6' : '3-6 km V-Shear',
                    'shear_u_0to6': '0-6 km U-Shear',
                    'shear_v_0to6': '0-6 km V-Shear',
                    'shear_u_0to1' : '0-1 km U-Shear',
                    'shear_v_0to1' : '0-1 km V-Shear',
                    'srh_0to1': '0-1 km SRH',
                    'srh_0to3': '0-3 km SRH',
                    'srh_0to500' : '0-500 m SRH',
                    'qv_2': '2-m Water Vapor',
                    't_2': '2-m Temp.',
                    'td_2': '2-m Dewpoint Temp.',
                    'bouyancy': 'Near-SFC Buoyant Forcing',
                    'buoyancy' : 'Near-SFC Buoyant Forcing',
                    'cp_bouy': 'SFC Buoyancy',
                    'rel_helicity_0to1': '0-1 km Relative Helicity',
                    '10-500m_bulk_shear': '10-500 m Bulk Shear',
                    '10-500m_bulkshear': '10-500 m Bulk Shear',
                    '10-m_bulk_shear' : '10-500 m Bulk Shear',
                    'mid_level_lapse_rate': '500-700 mb Lapse Rate',
                    'low_level_lapse_rate': '0-3 km Lapse Rate',
                    'temperature_850mb': 'T (850 mb)',
                    'temperature_700mb': 'T (700 mb)',
                    'temperature_500mb': 'T (500 mb)',
                    'temperature_850': 'T (850 mb)',
                    'temperature_700': 'T (700 mb)',
                    'temperature_500': 'T (500 mb)',
                    'geopotential_height_850mb': '850 mb Geop. Hgt.',
                    'geopotential_height_500mb': '500 mb Geop. Hgt.',
                    'geopotential_height_700mb': '700 mb Geop. Hgt.',
                    'geo_hgt_850': '$\Phi_{850}$',
                    'geo_hgt_500': '$\Phi_{500}$',
                    'geo_hgt_700': '$\Phi_{700}$',
                    'dewpoint_850mb': 'T$_{d, 850}$',
                    'dewpoint_700mb': 'T$_{d, 700}$',
                    'dewpoint_500mb': 'T$_{d, 500}$',
                    'td_850' :  'T$_{d, 850}$',
                    'td_700' :  'T$_{d, 700}$',
                    'td_500' :  'T$_{d, 500}$',
                    'cloud_top_temp': 'CTT',
                    'ctt' : 'CTT',
                    'dbz_1to3km_max': '1-3 km Max Refl.',
                    'dbz_1to3' : '1-3 km Max Refl.',
                    'dbz_3to5km_max': '3-5 km Max Refl.',
                    'dbz_3to5': '3-5 km Max Refl.',
                    'dbz_1km' : '1 km Refl.', 
                    'uh_0to2': '0-2 km UH',
                    'uh_2to5': '2-5 km UH',
                    'wz_0to2': '0-2 km Avg. Vert. Vort.',
                    'uh_0to2_instant': '0-2 km UH',
                    'uh_2to5_instant': '2-5 km UH',
                    'wz_0to2_instant': '0-2 km Avg. Vert. Vort.',
                    'comp_dz': 'Comp. Refl.',
                    'ws_80'  : '80-m Wind Speed',
                    'w_1km'  : 'Low-level $W$',
                    'w_down' : 'Downdraft',
                    'w_up': 'Updraft',
                    'hail': 'Hail',
                    'hailcast': 'Hail',
                    'Initialization Time' : 'Initialization Time',
                    'divergence_10m': '10-m Div.',
                    'div_10m': '10-m Div.',
                    'QVAPOR_850' : '850mb Water Vapor', 
                    'QVAPOR_700' : '700mb Water Vapor', 
                    'QVAPOR_500' : '500mb Water Vapor', 
                    'freezing_level' : 'Freezing Level', 
                    'stp' : 'STP', 
                    'stp_srh0to500' : 'STP_${SRH0TO500}$',
                    'okubo_weiss' : 'Okubo-Weiss Num.',
                    'ens_track_prob' : 'Max Ens. Prob', 
                    'area_ratio' : 'unitless', 
                    'avg_updraft_track_area' : "Ens. Avg. Updraft Track Area", 
                    }



map_to_units={
               'Bias' : 'unitless',
                'area': 'grid cells',
                'eccentricity': 'unitless',
                'extent' : 'unitless',
                'orientation': 'unitless',
                'minor_axis_length': 'grid length',
                'major_axis_length': 'grid length',
                'cin_ml': 'J kg$^{-1}$',
                'cape_ml': 'J kg$^{-1}$',
                'lcl_ml': 'm',
                'u_10': 'kts',
                'v_10': 'kts',
                'th_e_ml': 'K',
                'theta_e': 'K',
                'shear_u_0to6': 'kts',
                'shear_v_0to6': 'kts',
                'shear_u_3to6': 'kts',
                'shear_v_3to6': 'kts',
                'shear_u_0to1' : 'kts',
                'shear_v_0to1' : 'kts',
                'srh_0to1': 'm$^{2}$ s$^{-2}$',
                'srh_0to3': 'm$^{2}$ s$^{-2}$',
                'srh_0to500': 'm$^{2}$ s$^{-2}$',
                'qv_2': 'g kg$^{-1}$',
                't_2': '$^{\circ}$F',
                'td_2': '$^{\circ}$F',
                'bouyancy': 'm s$^{-2}$',
                'buoyancy' : 'm s$^{-2}$',
                'cp_bouy': 'm s$^{-2}$',
                '10-500m_bulk_shear': 'kts',
                '10-500m_bulkshear': 'kts',
                '10-m_bulk_shear' : 'kts',
                'mid_level_lapse_rate': '$^{\circ}$C / Km',
                'low_level_lapse_rate': '$^{\circ}$C / Km',
                    'temperature_850mb': '${^\circ}$F',
                    'temperature_700mb': '$^{\circ}$F',
                    'temperature_500mb': '${^\circ}$F',
                    'temperature_850': '$^{\circ}$F',
                    'temperature_700': '$^{\circ}$F',
                    'temperature_500': '$^{\circ}$F',
                    'geopotential_height_850mb': 'm',
                    'geopotential_height_500mb': 'm',
                    'geopotential_height_700mb': 'm',
                    'geo_hgt_850': 'm',
                    'geo_hgt_500': 'm',
                    'geo_hgt_700': 'm',
                    'dewpoint_850mb': '$^{\circ}$F',
                    'dewpoint_700mb': '$^{\circ}$F',
                    'dewpoint_500mb': '$^{\circ}$F',
                    'td_850' :  '$^{\circ}$F',
                    'td_700' :  '$^{\circ}$F',
                    'td_500' :  '$^{\circ}$F',
                    'cloud_top_temp': '$^{\circ}$C',
                    'ctt' : '$^{\circ}$C',
                    'dbz_1to3km_max': 'dBZ',
                    'dbz_1to3' : 'dBZ',
                    'dbz_1km' : 'dBZ',
                    'dbz_3to5km_max': 'dBZ',
                    'dbz_1to3' : 'dBZ',
                    'dbz_3to5' : 'dBZ',
                    'uh_0to2': 'm$^{2}$ s$^{-2}$',
                    'uh_2to5': 'm$^{2}$ s$^{-2}$',
                    'wz_0to2': 's$^{-1}$',
                    'uh_0to2_instant': 'm$^{2}$ s$^{-2}$',
                    'uh_2to5_instant': 'm$^{2}$ s$^{-2}$',
                    'wz_0to2_instant': 's$^{-1}$',
    
                    'comp_dz': 'dBZ',
                    'ws_80'  : 'kts',
                    'w_1km'  : 'm s$^{-1}$',
                    'w_down' : 'm s$^{-1}$',
                    'w_up': 'm s$^{-1}$',
                    'hail': 'in.',
                    'hailcast': 'in.',
                    'Initialization Time' : 'Hrs After Mignight',
                    'divergence_10m': 'kts',
                    'div_10m': 'kts',
                    'Run Date' : 'Run Date',
                    'okubo_weiss' : 'unitless', 
                    'forecast_time_index' : 'unitless',
                    'avg_updraft_track_area' : 'unitless',
                    'area_ratio' : 'unitless', 
                    'QVAPOR_850' : 'g/kg', 
                    'QVAPOR_700' : 'g/kg', 
                    'QVAPOR_500' : 'g/kg', 
                    'freezing_level' : 'm', 
                    'stp' : 'unitless', 
                    'stp_srh0to500' : 'unitless',
                    'okubo_weiss' : 'unitless',
                    'ens_track_prob' : 'unitless',
                    'obj_centroid_y' : 'unitless', 
                    'obj_centroid_x' : 'unitless', 
                    'label' : 'unitless', 
                    'avg_updraft_track_area' : "grid cells", 
                    }


def to_color(f):
    varname = f.split('__')[0]
    
    if varname in STORM_VARS:
        color = 'lightcoral'
    elif varname in ENV_VARS:
        color = 'lightblue'
    elif varname in MORPHOLOGICAL_FEATURES:
        color = 'peachpuff'
    else:
        color = 'lightgreen'
    
    return color

def get_units(feature):
    """ return units of a variable """
    varname = feature.split('_ens')[0].split('_time')[0]
    return map_to_units.get(varname, feature)


def to_readable_names(features):
    if not isinstance(features, list):
        features = [features]
    
    for f in features:
        old_f = f
        varname = f.split('_ens')[0].split('_time')[0]
    
        
        try:
            f = f.replace(varname, map_to_readable_names[varname])
        except:
            f = varname
        f = f.replace('_time_max', ' (time max)')
        f = f.replace('_time_min', ' (time min)')
        f = f.replace('_time_std', ' (time std)')
        f = f.replace('_ens_mean', ' (Ens. mean)')
        f = f.replace('_ens_std', ' (Ens. std)')
        f = f.replace('_spatial_mean', '')

        components = f.split('(')
        varname = components[0]
        ens_stat = components[-1].replace(')','').replace('std', 'stdev')

        if 'Ens.' not in f:
            official_name = f
        else:
            if '90' in ens_stat or '10' in ens_stat:
                p = 90 if '90' in ens_stat else 10
                ens_stat = '%s' % (ens_stat.split("_")[0])
                #ens_stat = ens_stat.replace('Ens. mean', '$\mu_{e,A}$')
                #ens_stat = ens_stat.replace('Ens. stdev', '$\sigma_{e,A}$')
            else:
                ens_stat = f'{ens_stat.split("_")[0]}'
                #ens_stat = ens_stat.replace('Ens. mean', '$\mu_{e,S}$')
                #ens_stat = ens_stat.replace('Ens. stdev', '$\sigma_{e,S}$')

            if len(components) == 3:
                time_stat = components[1]
                time_stat = time_stat.replace(') ', '').title().replace(' ', '-')
                time_stat = 'min' if 'Min' in time_stat else 'max'
                official_name = fr'{ens_stat} {varname}'
            else:
                official_name = fr'{ens_stat} {varname}'

    return official_name

def to_units(f):
    comps = f.split('__') 
    var = comps[0]
    return map_to_units[var] #.get(varname, '')


def to_display_name(f):
    stat_mapper = {'amp_ens_mean' : 'Ens. Amp. Mean', 
               'amp_ens_std' : 'Ens. Amp. Std.',
               'amp_ens_max' : 'Ens. Amp. Max.', 
               'ens_std' : 'Ens. Std.', 
               'ens_mean' : 'Ens. Mean',
               'ens_max' : 'Ens. Max',
               'ens_min' : 'Ens. Min',
              }

    comps = f.split('__') 
    var = comps[0]
    
    var = map_to_readable_names[var] #.get(var, var) 
    if len(comps)>1:
        if 'time' in comps[1]:
            # Intra-storm 
            ens_stat = stat_mapper[(comps[2]).split('_spatial')[0]]
            if comps[-1] == 'cond':
                ens_stat = f'Cond. {ens_stat}'
        else:
            # Env
            ens_stat = stat_mapper[comps[1]]
    
        display = f'{ens_stat} {var}'
    else:
        display = var
        
    return display

