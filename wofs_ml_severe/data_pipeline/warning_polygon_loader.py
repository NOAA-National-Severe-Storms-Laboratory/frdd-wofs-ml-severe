import geopandas as gpd
import pandas as pd
import numpy as np 
import xarray as xr 
from ..common.util import get_init_time, get_valid_time
from datetime import timedelta

class WarningPolygonLoader:
    def __init__(self, wofs_track_file=None, duration=30, err=10, file_path=None, start_time=None):
        if file_path is None:
            file_path = "/work/mflora/LSRS/2018-2023_nws_warnings_dt.feather"
        
        self.warn_df = gpd.read_feather(file_path)
        self.warn_df = self.warn_df.copy()
        
        # Convert the "issued_time" column to datetime format, if it's not already.
        #self.warn_df["ISSUED"] = pd.to_datetime(self.warn_df["ISSUED"])
        #self.warn_df["EXPIRED"] = pd.to_datetime(self.warn_df["EXPIRED"])
        
        # Limit the dataframe to tornado and severe weather storm-based polygons
        self.warn_df = self.get_storm_based_polygons(self.warn_df)
        
        if start_time is None:
            self.start_time = pd.to_datetime(get_valid_time(wofs_track_file))
        else:
            self.start_time = pd.to_datetime(start_time)
            
        self.end_time = self.start_time+timedelta(minutes=duration+err)

        self.start_time-=timedelta(minutes=err)
        
        if wofs_track_file is not None:
            with xr.open_dataset(wofs_track_file) as ds:
                self.lats, self.lons = ds['xlat'].values, ds['xlon'].values 
        
    def load(self, to_grid=True):
        # Get the valid warnings
        valid_df = self.get_valid_warnings(self.warn_df)

        # Split into separate dataframe for severe and tornado 
        # warnings polygons. 
        torn_df = self.get_polygons_by_type(valid_df, 'TO')
        sv_df = self.get_polygons_by_type(valid_df, 'SV')
        
        if not to_grid:
            return {'tornado': torn_df, 'severe' : sv_df}
        
        # Convert the polygons to a labeled array. 
        torn_label = self.geometries_to_labeled_array(torn_df, self.lons, self.lats)
        sv_label = self.geometries_to_labeled_array(sv_df, self.lons, self.lats)

        labels = ['tornado_warnings', 'severe_weather_warnings']
        
        data = {label : (['NE', 'NY'], arr) for label, arr in zip(labels, [torn_label, sv_label])}
        
        return xr.Dataset(data)
    
    def get_polygons_by_type(self, df, typ):
        return df[df.PHENOM == typ]
    
    def get_storm_based_polygons(self, df):
        """Retrieve only the storm-based, not count-based warning polygons"""
        return df[(df.GTYPE == 'P') & (df.PHENOM.isin(['TO', 'SV']))]

    def get_valid_warnings(self, df): 
        """Retrieve the warnings that are active overlaping the start and end time range 
           given. Typically, this will be the 30 min valid time for the 
           WoFS-ML-Severe tracks. """
        
        # Warning started before the beginning of the forecast period, but is valid during the 
        # forecast period. 
        overlaps_before = (df["ISSUED"] <= self.start_time) & (df["EXPIRED"] >= self.start_time)
        
        # Warning valid during the forecast period 
        overlaps_during = (df["ISSUED"] >= self.start_time) & (df["EXPIRED"] <= self.end_time)
        
        # Warning starts during the forecast period, but is valid after the end of the forecast period.
        overlaps_after = (df["ISSUED"] <= self.end_time) & (df["EXPIRED"] >= self.end_time)
        
        return df[(overlaps_before) | overlaps_during | overlaps_after]
                        
    def geometries_to_labeled_array(self, gdf, lons, lats):
        """
        Converts GeoDataFrame polygon geometrics to a labeled 2D array. 

        Parameters:
            - gdf (geopandas.GeoDataFrame): GeoDataFrame containing the geometries.
            - lons (2D array): WoFS longitude grid
            - lats (2D array): WoFS latitude grid

        Returns:
            numpy.ndarray: Labeled 2D array.
        """
        width, height = lons.shape
    
        gdf_cp = gdf.copy()
        
    
        gdf_cp['label'] = np.arange(1, len(gdf)+1)
    
        # Flatten the meshgrid and create a DataFrame with point geometries
        coords_df = pd.DataFrame({
            'X': lons.flatten(),
            'Y': lats.flatten()
        })
        geom_gdf = gpd.GeoDataFrame(coords_df, geometry=gpd.points_from_xy(coords_df.X, coords_df.Y))
    
        # Spatial join with the input gdf to get the labels for each point
        joined_gdf = gpd.sjoin(geom_gdf, gdf_cp, how='left', predicate='within')
    
        # ChatGPT fix for overlapping polygons. 
        joined_gdf = joined_gdf.drop_duplicates(subset=['X', 'Y']) 
    
        # Create a 2D array based on the labels of the points
        labels = joined_gdf['label'].values.reshape(height, width)
    
        # Replace NaNs with zeros
        labels[np.isnan(labels)] = 0 
    
        return labels
    