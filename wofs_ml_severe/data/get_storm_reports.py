import pandas as pd 
from datetime import datetime, timedelta
import numpy as np 
from os.path import join
import pyresample
import itertools 
from scipy.ndimage import maximum_filter

class StormReports:
    """
    StormReports loads CSV files containing data about timing and locations of
    storm report data (e.g., hail, tornadoes).

    Attributes:
    -------------------------
        initial_time, string (format = YYYYMMDDHHmm)
            The beginning date and time of a forecast period.

        forecast_length, integer
            Forecast length (in minutes) (default=30)

        err_window, integer 
            Allowable reporting error (in minutes) (default=15) 
                If err_window > 0:
                    time window = begin_time-err_window to begin_time+(forecast_length+err_window)
                else:
                    time window = begin_time to begin_time + forecast_length


    """
    def __init__(self, initial_time, forecast_length=30, err_window=15): 
        
        path='/work/mflora/LSRS'
        if len(initial_time) != 12:
            raise ValueError('initial_time format needs to be YYYYMMDDHHmm!')
        
        self.forecast_length = forecast_length
        self.err_window = err_window

        if self.date[:4] == '2021'
            print('For 2021 dataset, loading local storm reports rather than Storm Data...')
            dtype = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'MAG':np.float64, 'TYPETEXT':object}
            cols = ['VALID', 'LAT', 'LON', 'MAG', 'TYPETEXT']
            fname = 'lsr_201703010000_202105250000.csv'
            self.event_type='TYPETEXT'
            self.hail = 'HAIL'
            self.wind = 'TSTM WND DMG'
            self.torn = 'TORNADO'
        else:
            dtype = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'MAG':np.float64, 'EVENT_TYPE':object, 'TOR_F_SCALE' :object}
            cols = ['VALID', 'LAT', 'LON', 'MAG', 'EVENT_TYPE', 'TOR_F_SCALE']
            fname = 'StormData_201704_202012.csv'
            self.event_type = 'EVENT_TYPE'
            self.hail = 'Hail'
            self.wind = 'Thunderstorm Wind'
            self.torn = 'Tornado'

        df = pd.read_csv(join(path,fname), usecols=cols, dtype=dtype, na_values = 'None')
        df['date'] = pd.to_datetime(df.VALID.astype(str), format='%Y%m%d%H%M')
        self.df = df 

        self.get_time_window(initial_time)

    def to_grid(self, dataset, fname, magnitude='both', hazard = 'all'):
        """
        Convert storm reports to a grid. Applies a maximum filter of 3 grid points.
        For a 3 km grid spacing, assumes that reports are potentially valid over a
        9 x 9 km region. 

        Parameters:
        ----------------------------
            dataset, xarray.Dataset

            magnitude, string 

            hazards, string 

        """
        if magnitude == 'both'
            mag_iterator = ['severe', 'sig_severe']
        else:
            mag_iterator = [magnitude]

        hazard_iterator = ['hail', 'wind', 'tornado'] if hazard=='all' else hazard
    
        data={}
        for magnitude, hazard in itertools.product(iterator, hazard_iterator):
            ll = getattr(self, f'get_{hazard}_reports')(magnitude)
            xy =  self.to_xy(ds, lats=ll[0], ll[1])   
            xy = list(zip(xy[1,:],xy[0,:]))
                    
            gridded_reports = self.points_to_grid(xy, np.max(dataset.NX))
            data[f'{hazard}_{magnitude}'] = (['y', 'x'], maximum_filter(gridded_reports, 3))

        ds = xr.Dataset(data)
        ds.to_netcdf(fname)

        ds.close()


    def points_to_grid(self, xy_pair, nx):
        """
        Convert points to gridded data
        """
        xy_pair = [ (x,y) for x,y in xy_pair if x < nx-1 and y < nx-1 and x > 0 and y > 0 ]
        gridded_lsr = np.zeros((nx, nx))
        for i, pair in enumerate(xy_pair):
            gridded_lsr[pair[0],pair[1]] = i+1

        return gridded_lsr


    def get_time_window(self, initial_time): 
        '''
        Get beginning and ending of the time window to search for LSRs
        '''
        # Convert the datetime string to a datetime object 
        initial_datetime = datetime.strptime(date, '%Y%m%d%H%M') 
        end_date = start_date + timedelta(minutes=self.forecast_length+self.err_window)
        start_date-= timedelta(minutes=err_window)
        
        self.start_date = start_date
        self.end_date = end_date

        self.time_mask = (self.df.date > self.start_date) & (self.df.date <= self.end_date)

        return self

    def get_hail_reports(self, magnitude='severe'): 
        '''
        Load hail reports. 

        Parameters:
        ---------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe',  >= 1 in hail size
                if 'sig_severe', >= 2 in hail size

        Returns:
        ---------------------
            lats, lons 

        '''
        mag_mask = (self.df.MAG >= magnitude)
        etype = self.event_type
        event_type_mask = (self.df.etype == self.hail) 
        severe_hail_reports = self.df.loc[self.time_mask & mag_mask & event_type_mask] 

        return ( severe_hail_reports['LAT'].values, severe_hail_reports['LON'].values)

    def get_tornado_reports(self, magnitude='severe'):
        '''
        Load the tornado reports.

        Parameters:
        ----------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe', then use all tornado reports
                if 'sig_severe', >= EF2 tornado reports

        Returns:
        ---------------------
            lats, lons
        '''
        etype = self.event_type 
        event_type_mask = (self.df.etype == self.torn)

        if magnitude == 'sig_severe':
            scales = [ 'EF2', 'EF3', 'EF4', 'EF5']
            mag_mask = df.TOR_F_SCALE.isin(scales)
            
            total_masks = self.time_mask & event_type_mask & mag_mask
        else:
            total_masks = self.time_mask & event_type_mask 

        tornado_reports = self.df.loc[total_masks]
        
        return (tornado_reports['LAT'].values, tornado_reports['LON'].values)       
   
    def get_wind_reports(self, magnitude='severe'):
        '''
        Load the wind reports.

        Parameters:
        ----------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe', >= 50 kts 
                if 'sig_severe', >= 65 kts

        Returns:
        ---------------------
            lats, lons
        '''
        etype = self.event_type 
        event_type_mask = (self.df.etype == self.wind)

        # TODO: Be prepared that magnitude does not work for the wind! 
        mag_mask = (self.df.MAG >= magnitude)
        wind_reports = self.df.loc[self.time_mask & mag_mask & event_type_mask]

        return (wind_reports['LAT'].values, wind_reports['LON'].values) 

    def to_xy(ds, lats, lons):
        """Uses a KD-tree approach to determine, which i,j index an
        lat/lon coordiante pair is closest to. Used to map storm reports to 
        the WoFS domain"""
        grid = pyresample.geometry.GridDefinition(lats=ds.xlat, lons=ds.xlon)
        swath = pyresample.geometry.SwathDefinition(lons=lons, lats=lats)

        # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
        _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
            source_geo_def=grid, target_geo_def=swath, radius_of_influence=50000,
            neighbours=1)

        # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
        # the 2D grid indices:
        x,y = np.unravel_index(index_array, grid.shape)

