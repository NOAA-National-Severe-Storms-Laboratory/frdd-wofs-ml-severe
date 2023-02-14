#=================================
# RUNS THE DATAPIPELINE CLASS
#=================================
import pathlib, sys, os
path = pathlib.Path(os.getcwd()).parent.parent.resolve()
sys.path.append(str(path)+'/')

from wofs_ml_severe.data_pipeline.ml_data_pipeline import MLDataPipeline 

""" usage: stdbuf -oL python -u run_data_pipeline.py  2 > & log_data_pipeline & """

# Runs the datapipeline.

#delete_types = ['MLDATA', 'ENSEMBLETRACKS', 'MLTARGETS', 'FINAL']
#skip = []

delete_types = []
skip = ['get_ensemble_tracks']#, 'get_ml_features', 'match_to_storm_reports']

n_jobs = 30

# For debugging.
#dataPipeline = MLDataPipeline(dates=['20210526'], times=['2300'], n_jobs=30)(delete_types)

dataPipeline = MLDataPipeline(n_jobs=n_jobs)(skip, delete_types)
