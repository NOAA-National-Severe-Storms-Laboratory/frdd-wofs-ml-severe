# WoFS - ML - Severe

This is the official repository for the [Warn-on-Forecast System](https://www.nssl.noaa.gov/projects/wof/) machine learning (ML)-derived severe weather guidance. First, the codebase has the [data pipeline](https://github.com/monte-flora/wofs_ml_severe/blob/main/wofs_ml_severe/io/ml_data_pipeline.py) for building the ML dataset, which includes 

* Identifying the ensemble storm track objects ([Flora et al. 2018](https://journals.ametsoc.org/view/journals/wefo/34/6/waf-d-19-0094_1.xml?rskey=wn5MVr&result=6), [Flora et al. 2021](https://journals.ametsoc.org/view/journals/mwre/149/5/MWR-D-20-0194.1.xml?rskey=wn5MVr&result=3)), 
* Deriving the ML features from the WoFS summary files (diagnostic and prognostic variables derived from the WRFOUTs), and 
* Matching the ensemble storm tracks to local tornado, severe hail, and severe wind reports, 

Second, the codebase for fitting the model is found in [wofs_ml_severe.fit](https://github.com/monte-flora/wofs_ml_severe/tree/main/wofs_ml_severe/fit). The fitting relies on a package I developed for implementing pipelines, hyperparameter parameterization, cross-validation, and calibration into a single class ([ml_workflow](https://github.com/WarnOnForecast/ml_workflow)). 


Lastly, the codebase has extensive verification code including computing performance, roc, and attribute/reliability diagram curves. The verification output can then be plotted using the [PyMint](https://github.com/monte-flora/py-mint) package. 

