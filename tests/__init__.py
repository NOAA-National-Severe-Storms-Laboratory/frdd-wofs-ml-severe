#######################
# Get the filenames for the 
# unit testing. 
######################

import datetime
from os.path import join
import unittest
import os, sys, pathlib
from glob import glob

# Adding the parent directory to path so that 
# the package can be imported without being explicitly
path = os.path.dirname(os.getcwd())
sys.path.append(path)

# WoFS modules 
_base_module_path = '/home/monte.flora/python_packages/WoF_post'
import sys
sys.path.append(_base_module_path)

from wofs.common import get_ens_name, get_swath_file_name, INTERVAL, get_all_name
from wofs.common import zarr

class TestFiles(unittest.TestCase):
    def setUp(self):
        #self.file_30M = os.path.join(path, 'wofs_ml_severe', 'tests', 'test_data', 'wofs_30M_07_20210504_2205_2235.nc')
        self.ml_config_path = os.path.join('wofs_ml_severe', 'tests', 'test_data', 'ml_config.yml') 
        
        self.BASE_TEST_PATH = 'tests/test_data'
        ENS_NO = 3
        TOTAL_TIMESTEPS = 13
        START_TIME=datetime.datetime.strptime('202005142200', "%Y%m%d%H%M")

        MODEL_ID = 'WOFSRun20220914-153545'
        MODEL_DATE = '20220914'
        MODEL_INIT = '2200'
        MODEL_FCST = '2300'
        START_TIME=datetime.datetime.strptime(MODEL_DATE+MODEL_INIT, "%Y%m%d%H%M")
        
        DT = 5
        TIMESTEP = 12

        SUMMARY_FILE = f'{MODEL_ID}/{MODEL_DATE}/{MODEL_INIT}/wofs_ALL_{TIMESTEP}_{MODEL_DATE}_{MODEL_INIT}_{MODEL_FCST}.json'
        self.M30_FILE = f'{MODEL_ID}/{MODEL_DATE}/{MODEL_INIT}/wofs_30M_{TIMESTEP}_{MODEL_DATE}_2230_{MODEL_FCST}.json'
        
        path = f'{MODEL_ID}/{MODEL_DATE}/{MODEL_INIT}'
        delta_time_step = int(30/DT)
        
        ens_files = [f"{path}/{get_all_name(t, START_TIME, file_type='json')}" 
                     for t in range(TIMESTEP-delta_time_step, TIMESTEP+1)]
        
        svr_file = ens_files[0] 
        env_file = svr_file
        track_file = f'{path}/wofs_ENSEMBLETRACKS_{TIMESTEP}_{MODEL_DATE}_2230_{MODEL_FCST}.json'
        #print(f'{track_file=}')
        
        self.paths = {
                     'env_file'   : env_file,
                     'svr_file'   : svr_file, 
                     'ens_file'   : ens_files,
                     'track_file' : track_file, 
                }
        
   
    def assertIsFile(self, path):
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {str(path)}")    


