from bs4 import BeautifulSoup
import requests
import gzip
import shutil
from datetime import datetime, timedelta
import urllib.request
import os
import numpy as np
import pandas as pd
import re

class StormReportDownloader:
    """
    StormReportDownloader downloads STORM EVENT data from multiple years, 
    concatenates the data into a single dataframe, and then re-formats
    portions of the data (e.g., converting from local time to UTC time).
    
    Attribute
    ------------
    outdir : path-like, str
        Path to where the final dataset is stored and where temporary files
        are downloaded to. 
    
    """
    
    def __init__(self, outdir):
        self._outdir = outdir

    def get_storm_reports(self, start_date, end_date=None):
        pass
        
    def get_storm_events(self, years):
        paths = self.download_storm_event_files(years)
        df = self.format_data(paths)
        
        outpath = os.path.join(self._outdir, f'STORM_EVENTS_{years[0]}-{years[-1]}.csv' )
        df.to_csv(outpath)
    
    def get_http_href_path(self, url, comp='details'):
        """Get a list of url paths from a HTTP url"""
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return[url + '/' + node.get('href') for node in soup.find_all('a') 
            if comp in node.get('href')]

    def ungzip(self, path):
        with gzip.open(path, 'rb') as f_in:
            filename = os.path.basename(path).replace('.gz', '')
            outfile = os.path.join(self._outdir, filename)
            with open(outfile , 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        return outfile
                
    def download_storm_event_files(self, years):  
        """Downloads the STORM EVENT files
        
        :param years : list of str
            The years to download. 
        
        """
        base_url = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
        all_urls = self.get_http_href_path(base_url, comp='details')

        years_mod = [f'd{y}' for y in years]
        urls = [u for u in all_urls if any([y in u for y in years_mod])]

        paths = []
        for url in urls:
            print(f'Downloading {url}...')
            gzfile = os.path.join(self._outdir, f"{os.path.basename(url).split('_c')[0]}.csv.gz")
  
            # download
            urllib.request.urlretrieve(url, gzfile)
    
            # unzip
            path = self.ungzip(gzfile)
            paths.append(path)
        
            # Remove the gzip file.
            os.system(f'rm {gzfile}')
        
        return paths
    
    def download_lsr_warning_shapefiles(self, start_date, end_date=None):    
        lsr_zipfile = os.path.join(self._outdir, "latest_lsr.zip")

        # Convective day date (warning period will cover 12Z this day thru 12Z the next day)
        warning_date = datetime.strptime(start_date, "%Y%m%d" ) if type(start_date) == str else start_date
        
        next_date = datetime.strptime(end_date, "%Y%m%d" ) if type(end_date) == str else end_date

        str_date1 = '&year1=' + warning_date.strftime("%Y") + '&month1=' + warning_date.strftime("%m") + '&day1=' + warning_date.strftime("%d") + '&hour1=12&minute1=0'
        str_date2 = '&year2=' + next_date.strftime("%Y") + '&month2=' + next_date.strftime("%m") + '&day2=' + next_date.strftime("%d") + '&hour2=12&minute2=0'
    
        lsr_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo%5B%5D=ALL&state=_ALL" + str_date1 + str_date2 + "&fmt=shp"
    
        # download
        urllib.request.urlretrieve(lsr_url, lsr_zipfile)
    
        cmd = "unzip " + lsr_zipfile + " -d " + outdir 
        os.system(cmd) 
        
    def format_data(self, paths):
        """Combine and re-format the Storm Event CSVs"""
        DTYPE = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'BEGIN_YEARMONTH':object, 
         'BEGIN_DAY' : object, 'BEGIN_TIME' : object,
         'MAG':np.float64, 'EVENT_TYPE':object, 'TOR_F_SCALE' :object}
            
        COLS = ['BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 
        'BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE', 'EVENT_TYPE', 'TOR_F_SCALE', 'CZ_TIMEZONE', 'EVENT_ID']

        dfs = []
        for path in paths:
            df = pd.read_csv(path, usecols=COLS, dtype=DTYPE, na_values=None)
    
            # Format the dates and times and create the VALIDTIME
            dates = df['BEGIN_DAY']
            dates = [f'{(int(d)):02d}' for d in dates]

            times = df['BEGIN_TIME']
            new_times = []
            for t in times:
                if len(t)==1:
                    t = f'000{t}'
                elif len(t)==2:
                    t = f'00{t}'
                elif len(t)==3:
                    t = f'0{t}'
                new_times.append(t)

            df['VALID'] = df['BEGIN_YEARMONTH']+dates+new_times
            df = df.rename({'MAGNITUDE' : 'MAG', 
                   'BEGIN_LAT' : 'LAT', 
                   'BEGIN_LON' : 'LON',
                  }, 
                  axis='columns')

            df = df.drop(['BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME'], axis='columns')
            dfs.append(df)
    
        # Concatenate the dataframes together
        combined_df = pd.concat(dfs)   

        # Only keep the severe wx events. 
        df = combined_df[combined_df['EVENT_TYPE'].isin(['Thunderstorm Wind', 'Tornado', 'Hail', 'Flood'])]

        new_df = df.copy()

        # Convert from local time to UTC. 
        time_zone = df['CZ_TIMEZONE']
        time_zone = [int(re.findall(r'\d+', t)[0]) for t in time_zone]

        date = pd.to_datetime(df.VALID.astype(str), format='%Y%m%d%H%M')
        hrs = pd.to_timedelta(time_zone, 'h')

        new_date = date + hrs
        new_df['VALID'] = new_date.dt.strftime('%Y%m%d%H%M')
    
        return new_df