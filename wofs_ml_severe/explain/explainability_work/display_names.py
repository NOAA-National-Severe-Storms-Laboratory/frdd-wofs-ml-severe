obj_props_for_learning = ['area',
                          'eccentricity',
                          'extent',
                          'orientation',
                          'minor_axis_length',
                          'major_axis_length',
                          'matched_to_tornado_warn_ploys_15km',
                          'matched_to_tornado_warn_ploys_30km',
                          'matched_to_severe_wx_warn_polys_15km'
                          'matched_to_severe_wx_warn_polys_30km'
                          'obj_centroid_x',
                          'obj_centroid_y']

morph_vars = ['area',
                    'eccentricity',
                          'extent',
                          'orientation',
                          'minor_axis_length',
                          'major_axis_length',
                          ]


env_vars_smryfiles = [
                       'srh_0to1',
                       'srh_0to3',
                       'cape_ml',
                       'cin_ml',
                       'shear_u_0to6',
                       'shear_v_0to6',
                       'shear_u_0to1',
                       'shear_v_0to1',
                       'lcl_ml',
                       'th_e_ml',
                       'u_10',
                       'v_10']

env_vars_wofsdata  = [
                        'mid_level_lapse_rate',
                        'low_level_lapse_rate',
                        'temperature_850mb',
                        'temperature_700mb',
                        'temperature_500mb',
                        'geopotential_height_850mb',
                        'geopotential_height_700mb',
                        'geopotential_height_500mb',
                        'dewpoint_850mb',
                        'dewpoint_700mb',
                        'dewpoint_500mb',
                        ]

storm_vars_smryfiles = [ 'uh_0to2',
                         'uh_2to5',
                         'wz_0to2',
                         'comp_dz',
                         'ws_80',
                         'w_up',
                         'hailcast' ]

storm_vars_wofsdata = [ 'w_1km',
                        'w_down',
                        '10-m_bulk_shear',
                        'divergence_10m',
                        'bouyancy',
                        'cloud_top_temp',
                        'dbz_1to3km_max',
                        'dbz_3to5km_max']



min_vars = [ 'cloud_top_temp',
             'bouyancy',
             'divergence_10m',
             'w_down'
             ]


smryfile_variables = {'ENS': storm_vars_smryfiles, 'ENV': env_vars_smryfiles}
wofsdata_variables = storm_vars_wofsdata + env_vars_wofsdata

storm_variables = storm_vars_smryfiles + storm_vars_wofsdata
#storm_variables = ['10-500m_bulk_shear' if x == '10-m_bulk_shear' else x for x in storm_variables]
environmental_variables = env_vars_wofsdata + env_vars_smryfiles

map_to_readable_names={
                    'Run Date' : 'Run Date',
                    'Bias' : 'Bias',
                    'MRMS Reflectivity @ 0 min':'MRMS Reflectivity @ Initialization',
                    'MRMS Reflectivity @ 15 min':'MRMS Reflectivity 15 min after Initialization',
                    'MRMS Reflectivity @ 30 min':'MRMS Reflectivity 30 min after Initialization',
                    'obj_centroid_x': 'X-comp of Object Centroid',
                    'obj_centroid_y': 'Y-comp of Object Centroid',
                    'area': 'Area',
                    'eccentricity': 'Eccentricity',
                    'extent' : 'Extent',
                    'orientation': 'Orientation',
                    'minor_axis_length': 'Minor Ax. Len.',
                    'major_axis_length': 'Major Ax. Len.',
                    'matched_to_tornado_warn_ploys_15km': 'Matched to Tornado Warning Polygon (15 km)',
                    'matched_to_tornado_warn_ploys_30km': 'Matched to Tornado Warning Polygon (30 km)',
                    'matched_to_severe_wx_warn_polys_15km': 'Matched to Severe Weather Warning Polygon (15 km)',
                    'matched_to_severe_wx_warn_polys_30km': 'Matched to Severe Weather Warning Polygon (30 km)',
                    'label': 'Object Label',
                    'ensemble_member': 'Ensemble Member',
                    'cape_0to3_ml':'0-3 km ML CAPE',
                    'cin_ml': 'ML CIN',
                    'cape_ml': 'ML CAPE',
                    'lcl_ml': 'ML LCL',
                    'u_10': '10-m U',
                    'v_10': '10-m V',
                    'th_e_ml': 'ML  $\\theta_{e}$',
                    'theta_e': 'ML $\\theta_{e}$',
                    'shear_u_0to6': 'U-Shear$_{0-6 km}$',
                    'shear_v_0to6': 'V-Shear$_{0-6 km}$',
                    'shear_u_0to1' : 'U-Shear$_{0-1 km}$',
                    'shear_v_0to1' : 'V-Shear$_{0-1 km}$',
                    'srh_0to1': 'SRH$_{0-1}$',
                    'srh_0to3': 'SRH$_{0-3}$',
                    'qv_2': '2-m Water Vapor',
                    't_2': '2-m Temp.',
                    'td_2': '2-m Dewpoint Temp.',
                    'bouyancy': 'Buoyancy$_{sfc}$',
                    'cp_bouy': 'Buoyancy$_{sfc}$',
                    'rel_helicity_0to1': '0-1 km Relative Helicity',
                    '10-500m_bulk_shear': '10-500 m Bulk Shear',
                    '10-500m_bulkshear': '10-500 m Bulk Shear',
                    '10-m_bulk_shear' : '10-500 m Bulk Shear',
                    'mid_level_lapse_rate': 'Lapse Rate$_{mid}$',
                    'low_level_lapse_rate': 'Lapse Rate$_{low}$',
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
                    'dbz_1to3km_max': 'Max Refl.$_{1-3 km}$',
                    'dbz_3to5km_max': 'Max Refl.$_{3-5 km}$',
                    'dbz_3to5': '3-5 km Max Refl.',
                    'uh_0to2': 'UH$_{0-2 km}$',
                    'uh_2to5': 'UH$_{2-5 km}$',
                    'wz_0to2': 'Vert. Vort.$_{0-2 km}$',
                    'comp_dz': 'Comp. Refl.',
                    'ws_80'  : '80-m Wnd Spd',
                    'w_1km'  : 'Low-level $W$',
                    'w_down' : 'Downdraft',
                    'w_up': 'Updraft',
                    'hail': 'Hail',
                    'hailcast': 'Hail',
                    'Initialization Time' : 'Initialization Time',
                    'divergence_10m': '10-m Div.',
                    'div_10m': '10-m Div.'
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
                'shear_u_0to1' : 'kts',
                'shear_v_0to1' : 'kts',
                'srh_0to1': 'm$^{2}$ s$^{-2}$',
                'srh_0to3': 'm$^{2}$ s$^{-2}$',
                'qv_2': 'g kg$^{-1}$',
                't_2': '$^{\circ}$F',
                'td_2': '$^{\circ}$F',
                'bouyancy': 'm s$^{-2}$',
                'cp_bouy': 'm s$^{-2}$',
                '10-500m_bulk_shear': 'm s$^{-1}$',
                '10-500m_bulkshear': 'm s$^{-1}$',
                '10-m_bulk_shear' : 'm s$^{-1}$',
                'mid_level_lapse_rate': '$^{\circ}$C / Km',
                'low_level_lapse_rate': '$^{\circ}$C / Km',
                    'temperature_850mb': '${^\circ}$C',
                    'temperature_700mb': '$^{\circ}$C',
                    'temperature_500mb': '${^\circ}$C',
                    'temperature_850': '$^{\circ}$C',
                    'temperature_700': '$^{\circ$}$C',
                    'temperature_500': '$^{\circ$}$C',
                    'geopotential_height_850mb': 'm',
                    'geopotential_height_500mb': 'm',
                    'geopotential_height_700mb': 'm',
                    'geo_hgt_850': 'm',
                    'geo_hgt_500': 'm',
                    'geo_hgt_700': 'm',
                    'dewpoint_850mb': '$^{\circ}$C',
                    'dewpoint_700mb': '$^{\circ}$C',
                    'dewpoint_500mb': '$^{\circ}$C',
                    'td_850' :  '$^{\circ}$C',
                    'td_700' :  '$^{\circ}$C',
                    'td_500' :  '$^{\circ}$C',
                    'cloud_top_temp': '$^{\circ}$C',
                    'ctt' : '$^{\circ}$C',
                    'dbz_1to3km_max': 'dBZ',
                    'dbz_3to5km_max': 'dBZ',
                    'dbz_1to3' : 'dBZ',
                    'dbz_3to5' : 'dBZ',
                    'uh_0to2': 'm$^{2}$ s$^{-2}$',
                    'uh_2to5': 'm$^{2}$ s$^{-2}$',
                    'wz_0to2': 's$^{-1}$',
                    'comp_dz': 'dBZ',
                    'ws_80'  : 'kts',
                    'w_1km'  : 'm s$^{-1}$',
                    'w_down' : 'm s$^{-1}$',
                    'w_up': 'm s$^{-1}$',
                    'hail': 'in.',
                    'hailcast': 'in.',
                    'Initialization Time' : 'Initialization Time',
                    'divergence_10m': 'kts',
                    'div_10m': 'kts',
                    'Run Date' : 'Run Date',
                    }


def to_color(f):
    varname = f.split('_ens')[0].split('_time')[0]
    if varname in storm_variables:
        color = 'lightgreen'
    elif varname in environmental_variables:
        color = 'lightblue'
    elif varname in obj_props_for_learning:
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
                ens_stat = ens_stat.replace('Ens. mean', '$\mu_{e,A}$')
                ens_stat = ens_stat.replace('Ens. stdev', '$\sigma_{e,A}$')
            else:
                ens_stat = f'{ens_stat.split("_")[0]}'
                ens_stat = ens_stat.replace('Ens. mean', '$\mu_{e,S}$')
                ens_stat = ens_stat.replace('Ens. stdev', '$\sigma_{e,S}$')

            if len(components) == 3:
                time_stat = components[1]
                time_stat = time_stat.replace(') ', '').title().replace(' ', '-')
                time_stat = 'min' if 'Min' in time_stat else 'max'
                official_name = fr'{varname} ({ens_stat})'
            else:
                official_name = fr'{varname} ({ens_stat})'

    return official_name