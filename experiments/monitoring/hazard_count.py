#======================================================
# To understand the occurence of each of the hazards 
# predicted by ML Severe, this script pulls in the training
# and testing data and plots counts of the hazards 
# present in each
# 
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Oct. 13, 2026
#======================================================

import sys
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')

from wofs_ml_severe.io.io import MLDataLoader

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

hazards = ['severe_hail', 'severe_wind', 'severe_torn']
lead_times = ["first_hour", "second_hour"]
modes = ["training", "testing"]

# read in the training and testing data and save counts of each hazard for each hour to 
# a pandas dataframe
counts = pd.DataFrame(columns = ["hazards", "lead_time", "mode", "count"])

for mode in modes:
    for name in hazards:
        for time in lead_times:

            loader = MLDataLoader(target_column = name, lead_time = time, mode = mode, 
                                  data_path = "/work2/lucas.jones/ml_data")
            x, data, metadata = loader.load()

            # get the count of that hazard occuring
            total = data.sum()
            
            counts.loc[len(counts)] = [name, time, mode, total]

training = counts[counts["mode"] == "training"]
testing = counts[counts["mode"] == "testing"]

fig, ax = plt.subplots(nrows = 2, ncols = 1, layout = "constrained")
res1 = ax[0].grouped_bar([training[training["hazards"] == "severe_hail"]["count"], 
                         training[training["hazards"] == "severe_wind"]["count"],
                         training[training["hazards"] == "severe_torn"]["count"]], 
                         tick_labels = lead_times, labels = hazards)
res2 = ax[1].grouped_bar([testing[testing["hazards"] == "severe_hail"]["count"],
                          testing[testing["hazards"] == "severe_wind"]["count"], 
                          testing[testing["hazards"] == "severe_torn"]["count"]],
                          tick_labels = lead_times, labels = hazards)
ax[0].set_title("Training Data")
ax[1].set_title("Testing Data")
fig.suptitle("Occurence of Hazards in the Data- Random Split", fontweight = "bold")
plt.savefig("hazard_counts_randomsplit.png", dpi = 400)