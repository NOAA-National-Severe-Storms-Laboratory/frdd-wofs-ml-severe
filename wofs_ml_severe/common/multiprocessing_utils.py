import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime
from tqdm import tqdm  
import traceback
from collections import ChainMap
import warnings
from copy import copy

from joblib._parallel_backends import SafeFunction
from joblib import delayed, Parallel

# Ignore the warning for joblib to set njobs=1 for
# models like RandomForest
warnings.simplefilter("ignore", UserWarning)

class LogExceptions(object):
    def __init__(self, func):
        self.func = func

    def error(self, msg, *args):
        """ Shortcut to multiprocessing's logger """
        return mp.get_logger().error(msg, *args)
    
    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
                    
        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

def to_iterator(*lists):
    """
    turn list
    """
    return itertools.product(*lists)


def run_parallel(
    func,
    args_iterator,
    nprocs_to_use,
    kwargs={}, 
):
    """
    Runs a series of python scripts in parallel. Scripts uses the tqdm to create a
    progress bar.
    Args:
    -------------------------
        func : callable
            python function, the function to be parallelized; can be a function which issues a series of python scripts
        args_iterator :  iterable, list,
            python iterator, the arguments of func to be iterated over
                             it can be the iterator itself or a series of list
        nprocs_to_use : int or float,
            if int, taken as the literal number of processors to use
            if float (between 0 and 1), taken as the percentage of available processors to use
        kwargs : dict
            keyword arguments to be passed to the func
    """
    iter_copy = copy(args_iterator)
    
    total = len(list(iter_copy))
    pbar = tqdm(total=total)
    def update(*a):
        pbar.update()
    
    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use * mp.cpu_count())
    else:
        nprocs_to_use = int(nprocs_to_use)

    if nprocs_to_use > mp.cpu_count():
        raise ValueError(
            f"User requested {nprocs_to_use} processors, but system only has {mp.cpu_count()}!"
        )
        
    pool = Pool(processes=nprocs_to_use)
    for args in args_iterator:
        if isinstance(args, str):
            args = (args,)
         
        pool.apply_async(LogExceptions(func), args=args, callback=update)
    pool.close()
    pool.join()

    ##return results 

