##################################
# Tests the WOFS-ML-Severe code! 
##################################
from os.path import join 
from glob import glob

# From the __init__.py 
from tests import TestFiles
from wofs_ml_severe import (generate_ensemble_track_file, 
                            MLDataGenerator) 

import sys, os
sys.path.insert(0, '/home/monte.flora/python_packages/wofs_ml_severe')
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')

from wofs.common import get_ens_name, get_swath_file_name, INTERVAL, get_all_name
from wofs.common import zarr
import wofs 

class TestWoFSMLSevere(TestFiles):
    """Test for the ensemble storm track segmentation code"""

    def test_ml_pipeline(self):
        # Generate the ensemble storm track file. 

        outdir = self.BASE_TEST_PATH

        save_filename = generate_ensemble_track_file(self.M30_FILE, outdir=outdir, debug=True)
        
        self.track_path = save_filename.replace('.json', '.nc')
        
        # Checks that ENSEMBLETRACK file has been created. 
        self.assertIsFile(self.track_path)
        
        self.paths['track_file'] = self.track_path
        
        # Generate the ML features from the tracks and summary files. 
        # For the moment, TEMP=True as the mid-level temps 
        # need to be converted from deg C to deg F. 
        generator = MLDataGenerator(TEMP=True, retro=False, 
                                    logger=print, outdir=outdir, debug=False) 
        
        # This will generate the MLPROB, MLDATA, and EXPLAIN datasets. 
        generator(self.paths, n_processors=1, realtime=True)
        
        # Generate the MLPROB plots
        mlprob_file = save_filename.replace('ENSEMBLETRACKS', 'MLPROB')

        ###mlprob_file = os.path.join(outdir, 'wofs_MLPROB_18_20230303_2300_2330.nc') 
        
        init_dt = self.START_TIME
        valid_dt = self.START_TIME
        
        wofs.plot_ml(mlprob_file, outdir, timestep_index=self.ts, dt=self.dt, init_dt=init_dt, 
                     valid_dt=valid_dt)

    

        
        
 