#######################
# Get the filenames for the 
# unit testing. 
######################

import unittest
import os, sys, pathlib

# Adding the parent directory to path so that 
# the package can be imported without being explicitly
path = os.path.dirname(os.getcwd())
sys.path.append(path)

class TestFiles(unittest.TestCase):
    def setUp(self):
        self.file_30M = os.path.join(path, 'wofs_ml_severe', 'tests', 'test_data', 'wofs_30M_07_20210504_2205_2235.nc')
        self.ml_config_path = os.path.join('wofs_ml_severe', 'tests', 'test_data', 'ml_config.yml') 
        
    def assertIsFile(self, path):
        if not pathlib.Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {str(path)}")    


