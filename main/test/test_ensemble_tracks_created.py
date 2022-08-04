import os
from os.path import join, exists
from glob import glob

############################################
#
# CHECKS IF A ENSEMBLETRACKS FILE WAS CREATE FOR 
# A CORRESPONDING 30M FILE
#
############################################

def are_they_all_there(files):
    track_files = [f.replace('30M', 'ENSEMBLETRACKS') for f in files]
    all_exist = [exists(f) for f in track_files]
    if not all(all_exist):
        inds = [i for i, x in enumerate(all_exist) if not x]
        for i in inds:
            print(track_files[i], 'does exist!') 
    


base_path = '/work/mflora/SummaryFiles/'

dates = [d for d in os.listdir(base_path) if '.txt' not in d]

for d in dates:
    times = os.listdir(join(base_path,d))
    for t in times:
        filepath = join(base_path,d,t) 
        files_30M = glob(join(filepath, 'wofs_30M*'))
        are_they_all_there(files_30M)

                    


