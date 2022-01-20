from WoF_post.wofs.post.wofs_cbook import  identify_deterministic_tracks, identify_ensemble_tracks, fill_with_prob

from WoF_post.wofs.post.utils import (
    save_dataset,
    generate_track_filename,
    generate_summary_file_name,
)
from os.path import join
import xarray as xr 

def worker(ncfile):
    """Worker function for parallelization"""
    var = 'w_up'
    ds = xr.open_dataset(ncfile, decode_times=False)
    data_to_label = ds[var].values

    deterministic_tracks = identify_deterministic_tracks(
                            data_to_label=data_to_label,
                            method=var_dict[var][0],
                            params=var_dict[var][1],
                            qc_params=qc_params[var]
                            )

    # Save the deterministic tracks?

    results = identify_ensemble_tracks(deterministic_tracks, ensemble_size=ensemble_size)
    field_names = [f'{var}__ensemble_tracks', f'{var}__filled_tracks', f'{var}__ensemble_probabilities']

    #data = {field: (['NY','NX'], result) for field, result in zip(field_names, results) }
    #save_filename = ncfile.replace('30M', 'ENSEMBLETRACKS')

    #print(f'Saving {save_filename}')
    #ds = xr.Dataset(data)
    #save_dataset(fname=save_filename, dataset=ds)

DEBUG = True

base_path = '/work/mflora/SummaryFiles'
if DEBUG:
    from glob import glob
    date = '20210504'
    init_time = '2200'
    fname = glob(join(base_path,date,init_time, 'wofs_30M*' ))[12]
    
    worker(fname)

else:
    base_path = '/work/mflora/SummaryFiles'
    dates = os.listdir(base_path)
    filenames = []
    for d in dates:
        filepath = join(base_path, d)
        times = os.listdir(filepath)
        for t in times:
            potential_files = os.listdir(join(filepath,t))
            filenames.extend([f for f in potential_files if '30M' in f])

# run parallel


