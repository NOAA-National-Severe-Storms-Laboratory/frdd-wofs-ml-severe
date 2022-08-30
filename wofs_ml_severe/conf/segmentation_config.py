#======================================================
# Configuration file for the ensemble storm track segmentation
# Contains the parameters used. 
#
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

param_set = [ {'min_thresh': 0,
                   'max_thresh': 100,
                   'data_increment': 1,
                   'delta': 0,
                   'area_threshold': 800,
                   'dist_btw_objects': 125 },
             
             {'min_thresh': 30,
                   'max_thresh': 100,
                   'data_increment': 1,
                   'delta': 0,
                   'area_threshold': 400,
                   'dist_btw_objects': 30 },
                 
              {'min_thresh': 50,
                   'max_thresh': 100,
                   'data_increment': 1,
                   'delta': 0,
                   'area_threshold': 250,
                   'dist_btw_objects': 30 },    
                 
            ]

config = {   
            'deterministic' : {
                'params' : ('single_threshold', {'bdry_thresh':10.0}),
                'qc_params' : [('min_area',10.)]
            },
            'ensemble' : {
                'params' : {'params' : param_set},
                'ensemble_size' : 18
            }
}
