#######################################################################
# Author: Montgomery Flora
#
# DESCRIPTION: 
#       The storm data reports are converted from point-based products
#       into a grid-based products for each WoFS domain. 
#######################################################################

from data.get_storm_reports import StormReports
from utils.multiprocessing_utils import run_parallel

import xarray as xr 
import itertools 

def worker(date, time,):
    """Worker Function for Parallelization"""
    report = StormReports(initial_time, 
            forecast_length=30,
            err_window=15, 
            )

    # Load a summary file
    # fname = 
    # ds = xr.open_dataset(fname)

    #fname= 
    report.to_grid(ds=ds, fname=) 


dates = config.ml_dates
times = config.observation_times
run_parallel(
            func = worker,
            nprocs_to_use = 0.4,
            iterator = itertools.product(dates, times)
            )




