#=================================
# RUNS THE DATAPIPELINE CLASS
#=================================
import sys 
sys.path.append('/home/monte.flora/python_packages/wofs_ml_severe')
from wofs_ml_severe import MLDataPipeline

""" usage: stdbuf -oL python -u run_data_pipeline.py  2 > & log_data_pipeline & """

# Runs the datapipeline.
dataPipeline = MLDataPipeline(n_jobs=30)()
