# LOAD WOFS SUMMARY FILES (MULTIPLE) 

from os.path import join, exists
from glob import glob

class LoadSummaryFiles:
    """
    Load the Warn-on-Forecast Summary Files 
    """
    def get_filenames(self, kind, time_steps):
        """Get the filenames based on time step(s)"""
        if not isinstance(time_steps, list):
            time_steps = [time_steps]
        
        kinds = ['ENS', 'ENV', 'SVR', 'SWT', '30M', '60M']
        if kind not in kinds:
            raise ValueError(f'{kind} is not a valid option!')

        if not exists(path)
            raise Exception(f'{path} does not exist!')

        # glob returns a list, so we only want the first and only element
        filenames = [ glob(join(path f'wofs_{kind}_{t:02d}*'))[0] for t in time_steps]
        
        return filenames

    def load(self, path, kind, vars_to_load, time_steps, drop_vars=None):
        """
        Load a single or multiple WoFS summary files. 
        
        Log: April 20, 2017. It was key to load the ncfiles separately
        rather than with load_mfdatset since load_mfdataset cause a MemoryError
        
        Parameters:
        --------------
            path , string
                File path to the summary files including date and time directory 
                E.g., path = /work/mflora/SummaryFiles/20210504/2330

            kind , 'ENS', 'ENV', 'SVR', 'SWT', '30M', or '60M'
                Kind of summary files to load

            vars_to_load, list of strings   
                Summary file variables to load 

            time_steps, integer or list thereof 
                Forecast time steps to load 

            drop_vars, list of strings
                Variables to not load

        Returns:
        ---------------
            multiple_datasets_dict : dict
                A dictionary where the keys are vars_to_load
                and the items are xr.Datasets concatenated
                if multiple time steps are provided.

                xr.Dataset shape = (NT, NE, NY, NX) 

        """
        nc_file_paths = self.get_filenames(time_steps, tag)
        drop_vars = [] if drop_vars is None else ['hgt', 'xlat', 'xlon']

        multiple_datasets = [ ]
        for i, ncfile in enumerate(nc_file_paths):
            dataset = xr.open_dataset(ncfile, drop_variables=drop_vars)
            dataset_loaded = [dataset[var].values for var in vars_to_load]
            multiple_datasets.append(dataset_loaded)
            dataset.close()
            del dataset, dataset_loaded

        multiple_datasets = np.array(multiple_datasets).squeeze()

        if len(nc_file_paths) == 1:
            multiple_datasets_dict = {var: multiple_datasets[i].squeeze() for i, var in enumerate(vars_to_load)}
        else:
            multiple_datasets_dict = {var: multiple_datasets[:,i].squeeze() for i, var in enumerate(vars_to_load)}

        return multiple_datasets_dict





