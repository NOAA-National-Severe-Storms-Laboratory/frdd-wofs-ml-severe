##################################
# Tests the ensemble storm track 
# segmentation code.
##################################

# From the __init__.py 
from tests import TestFiles

from wofs_ml_severe.data_pipeline.ensemble_track_segmentation import generate_ensemble_track_file


class TestEnsembleTrackSeg(TestFiles):
    """Test for the ensemble storm track segmentation code"""
    
    def test_generate_track_file(self):
        generate_ensemble_track_file(self.file_30M)
        
        path = self.file_30M.replace('30M', 'ENSEMBLETRACKS')
        
        # Checks that ENSEMBLETRACK file has been created. 
        self.assertIsFile(path)