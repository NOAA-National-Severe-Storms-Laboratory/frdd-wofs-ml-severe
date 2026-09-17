#===================================================
# This script runs the StormReportDownloader 
# class to download local storm reports and warning 
# polygons to the desired directory.
#
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Date: Sept. 8, 2026
#===================================================
import pathlib, sys, os
path = pathlib.Path(os.getcwd()).parent.parent.resolve()
sys.path.append(str(path)+'/')

from wofs_ml_severe.data_pipeline.storm_report_downloader import StormReportDownloader 

""" usage: stdbuf -oL python -u run_report_downloader.py  2 > & log_report_download & """

outpath = '/work2/lucas.jones/LSRS/'
download_lsrs = False
download_warn = True

print("===================================================")

if download_lsrs:
    print("=============Beginning Report Download=============")

    # create a downloader object
    reportDownloader = StormReportDownloader(outdir = outpath)
    reports = reportDownloader.get_storm_events(years = ["2026"])

    print("============Report Download Completed==============")

if download_warn:
    print("=========Beginning Warning Polygon Download=========")

    #create downloader object
    warnDownloader = StormReportDownloader(outdir = outpath)
    warnings = warnDownloader.download_warnings(start_date = "2026-01-01T00:00Z", end_date = "2026-09-17T00:00Z")

    print("=========Warning Polygon Download Completed=========")