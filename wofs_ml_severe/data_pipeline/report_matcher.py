from monte_python.object_matching import match_to_lsrs, ObjectMatcher
from .storm_report_loader import StormReportLoader
from wofs_ml_severe.common.util import decompose_file_path
import xarray as xr 
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import traceback
from os.path import basename, dirname
from datetime import timedelta 

def get_init_time(filename):
    
    init_time = basename(dirname(filename))
    init_date = basename(dirname(dirname(filename)))
    
    return init_date+init_time
                         

def get_valid_time(filename, offset=6, dt=5):
    comps = decompose_file_path(filename)
    init_time = get_init_time(filename)

    valid_duration = int(comps['TIME_INDEX'])*dt - (offset*dt)
    start_time=(pd.to_datetime(init_time)+timedelta(minutes=valid_duration)).strftime('%Y%m%d%H%M')
    
    return start_time


class MatchReportsToTracks:
    """Produces the MLTARGETS dataframe."""
    def __init__(self, min_dists=[0,2,5]):
        self._min_dists = min_dists

    def get_reports(self, ncfile, 
                reports_path='/work/mflora/LSRS/STORM_EVENTS_2017-2022.csv', 
                report_type='NOAA'):
        """ Get the storm reports for the forecast period. """
        # The track files have the initial and end of a 30-min time period
        # so we want to use the valid date and init time for the matching. 
        init_time = get_valid_time(ncfile) 
        report = StormReportLoader(
            reports_path,
            report_type,
            init_time, 
            forecast_length=30,
            err_window=5, 
            )
 
        report_lsrs = StormReportLoader(
            '/work/mflora/LSRS/lsr_201703010000_202106090000.csv',
            'IOWA',
            init_time, 
            forecast_length=30,
            err_window=5, 
            )
 
        ds = xr.load_dataset(ncfile)
        
        try:
            grid_ds = report.to_grid(dataset=ds)
            lsr_points = report.get_points(dataset=ds)
            
            grid_ds_lsr = report_lsrs.to_grid(dataset=ds)
            lsr_points2 = report_lsrs.get_points(dataset=ds)
            
            new_vars = {var: f'{var}__IOWA' for var in grid_ds_lsr.data_vars}
            grid_ds_lsr = grid_ds_lsr.rename(new_vars)    
        
            grid_ds = xr.merge([grid_ds, grid_ds_lsr])
     
            new_points = {f'{var}__IOWA': lsr_points2[var] for var in lsr_points.keys()}
        
            lsr_points = {**lsr_points, **new_points}
    
        
        except Exception as e:
            print(traceback.format_exc())
            #self.logger('info', f'Unable to process storm reports for {ncfile}!')
            #self.logger('error', e, exc_info=True) 
    
        return grid_ds, lsr_points


    def match(self, track_file):
        """
        Match the ensemble storm tracks to the gridded LSRs. 
        Outputs a dataframe of targets for the MLDATA-based summary files.
    
        Multiple matching minimum matching distances are used. 
        """
        tracks_ds = xr.load_dataset(track_file, decode_times=False)
        tracks = tracks_ds['w_up__ensemble_tracks'].values
        ens_probs = tracks_ds['w_up__ensemble_probabilities'].values
    
        labels = np.unique(tracks)[1:]
        storm_data_ds, lsr_points = self.get_reports(track_file)
    
        target_vars = [v for v in storm_data_ds.data_vars if 'severe' in v] 
    
        if len(target_vars) == 0:
            return None 
    
        targets_data = {}
            
        for var in target_vars:
            target = storm_data_ds[var].values
            
            object_props = regionprops(tracks, tracks)  
            match_dict = match_to_lsrs(object_props, lsr_points[var], dist_to_lsr=1)
            
            for min_d in self._min_dists:
                obj_match = ObjectMatcher(min_dist_max=min_d,
                                      score_thresh = 0.2,
                                      time_max=0,
                                      one_to_one = False, match_to_reports=True)
                        
                matched_tracks, _ , _ = obj_match.match_objects(object_set_a=tracks, 
                                                            object_set_b=target, input_a=ens_probs)

                        
                # Create target column 
                targets_data[f"{var}_{min_d*3}km"] = [1 if label in matched_tracks 
                                                                     else 0 for label in labels]
                
            # Original target column 
            targets_data[f'{var}_original'] = [match_dict[label] for label in labels]
            
        df = pd.DataFrame(targets_data)
        target_file = track_file.replace('ENSEMBLETRACKS', 'MLTARGETS').replace('.nc', '.feather')
            
        #self.logger('debug', f'Saving {target_file}...')
        ####target_file = os.path.basename(target_file)
            
        df.to_feather(target_file)

        return target_file