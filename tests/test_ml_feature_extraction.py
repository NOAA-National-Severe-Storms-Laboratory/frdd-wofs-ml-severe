#######################################
# Tests the ML Feature Extraction Code.
#######################################

import numpy as np 
import xarray as xr 

# From the __init__.py 
from tests import TestFiles

# Import the wofs_ml_severe modules 
from wofs_ml_severe.data_pipeline.ensemble_track_segmentation import generate_ensemble_track_file
from wofs_ml_severe.data_pipeline.ml_data_generator import MLDataGenerator

import monte_python

class TestMLFeatureExtraction(TestFiles):
    """Test for the ML Feature Extraction"""
    
    def test_feature_extract(self):
        NV, NE, NY, NX = (3,5,300,300)
        
        # Generate the track file. 
        track_file = self.file_30M.replace('30M', 'ENSEMBLETRACKS')
        generate_ensemble_track_file(self.file_30M)

        # Create fake data (NV, NE, NY, NX)
        ens_data = np.random.normal(size=(NV,NE,NY,NX))
        
        ds = {}
        for typ in ['ens', 'env', 'svr']:
            data = {f'{typ}_{i+1}' : (['NE', 'NY', 'NX'], ens_data[i,:,:,:]) for i in range(NV)}
            if typ == 'ens':
                data['uh_2to5'] = (['NE', 'NY', 'NX'], ens_data[0,:,:,:])
                data['ws_80'] = (['NE', 'NY', 'NX'], ens_data[0,:,:,:])
                data['hailcast'] = (['NE', 'NY', 'NX'], ens_data[0,:,:,:])
            ds[typ] = xr.Dataset(data)
        
        ens_filenames = [f'wofs_ENS_{i:02d}' for i in range(6)]
        env_filename = ['wofs_ENV_06']
        svr_filename = ['wofs_SVR_06']
        
        all_filenames = ens_filenames + env_filename + svr_filename
        
        files_to_load = {'track_file' : track_file,
                         'ens_file' : ens_filenames,
                         'env_file' : env_filename[0],
                         'svr_file' : svr_filename[0],
                        }
                         
        for f in all_filenames:
            typ = f.split('_')[1].lower()
            print(ds.keys(), typ, f)
            ds[typ].to_netcdf(f) 
        
        data_generator = MLDataGenerator(files_to_load, TEMP=False, ml_config_path=self.ml_config_path)

        data_generator(runtype='rto') 
        
        # Checks that MLDATA file has been created. 
        # (optional) check the len(data) == n_objects. 
        self.assertIsFile(track_file.replace('ENSEMBLETRACKS', 'MLDATA').replace('.nc', '.feather'))