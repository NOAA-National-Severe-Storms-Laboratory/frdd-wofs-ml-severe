# Always prefer setuptools over distutils

import setuptools  # this is the "magic" import
import os
from numpy.distutils.core import setup, Extension

#from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

__key__ = 'PACKAGE_VERSION'
__version__= os.environ[__key__] if __key__ in os.environ else '0.0.3'

setup(
    name='wofs_ml_severe', 
    version=__version__,
    description='Official WoFS-ML-Severe Repository', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/WarnOnForecast/wofs_ml_severe', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists',
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'scikit-learn==1.0.2',
        'scikit-image>=0.19',
        'hyperopt==0.2.7', 
        'numpy',
        'scipy',
        'pandas', 
        'xarray>=0.18',
        'scikit-learn-intelex==2023.0.1',
        'xgboost==1.7.1',
        'imbalanced-learn==0.10.1',
    ],
    package_data={'wofs_ml_severe' : ['conf/default_ml_config.yml', 'conf/ml_config_2017.yml', 
                                      'conf/ml_config_2018-19.yml', 
                                      'conf/ml_config_2020-current.yml', 
                                      'conf/ml_config_realtime.yml', 
                                      'conf/ml_config_retro.yml', 'json/*']},
    packages=['wofs_ml_severe', 'wofs_ml_severe.common', 'wofs_ml_severe.data_pipeline', 'wofs_ml_severe.io', 
              'wofs_ml_severe.evaluate', 'wofs_ml_severe.conf'],  # Required
    python_requires='>=3.8, <4',
    package_dir={'wofs_ml_severe': 'wofs_ml_severe'},
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/WarnOnForecast/wofs_ml_severe/issues',
        'Source': 'https://github.com/WarnOnForecast/wofs_ml_severe',
    },
)
