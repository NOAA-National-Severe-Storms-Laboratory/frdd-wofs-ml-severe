##################################
# Tests the WOFS-ML-Severe code! 
##################################
from os.path import join 
from glob import glob

# From the __init__.py 
from tests import TestFiles
from wofs_ml_severe import (generate_ensemble_track_file, 
                            MLDataGenerator) 

from wofs.common import get_ens_name, get_swath_file_name, INTERVAL, get_all_name
from wofs.common import zarr

class TestWoFSMLSevere(TestFiles):
    """Test for the ensemble storm track segmentation code"""
    
    def test_generate_ml_data(self):
        # Generate the ensemble storm track file. 
        outdir = self.BASE_TEST_PATH
        save_filename = generate_ensemble_track_file(self.M30_FILE, outdir=outdir, debug=True)
       
        self.track_path = save_filename.replace('30M', 'ENSEMBLETRACKS').replace('.json', '.nc')
        
        # Checks that ENSEMBLETRACK file has been created. 
        self.assertIsFile(self.track_path)
        
        self.paths['track_file'] = self.track_path
        
        # Generate the ML features from the tracks and summary files. 
        generator = MLDataGenerator(TEMP=False, retro=False, 
                                    logger=print, outdir=outdir, debug=True) 
        
        generator(self.paths, n_processors=1, realtime=True)
        
        # Checks that MLDATA and MLPROB files have been created. 
        # path = save_filename.replace('30M', 'ENSEMBLETRACKS')
        #self.assertIsFile(path)
        
        
 