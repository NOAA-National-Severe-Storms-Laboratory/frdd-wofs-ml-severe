# WoFS - ML - Severe

This is the official repository for the [Warn-on-Forecast System](https://www.nssl.noaa.gov/projects/wof/)(WoFS) machine learning (ML)-derived severe weather guidance. The WoFS-ML-Severe products produce probabilistic guidance for individual severe weather threats (i.e., tornadoes, large hail, and strong near-surface straight-line winds). The guidance is available on the [cloud-based WoFS web viewer](https://cbwofs.nssl.noaa.gov/Forecast) (under ML Products) whenever the WoFS is running (NOTE: it is currently an experimental system and is largely ran in colloboration with NOAA's Hazardous Weather Testbed Spring Forecasting Experiment). The following discussion and figures come from [Flora et al. 2019](https://journals.ametsoc.org/view/journals/wefo/34/6/waf-d-19-0094_1.xml?rskey=wn5MVr&result=6) and [Flora et al. 2021](https://journals.ametsoc.org/view/journals/mwre/149/5/MWR-D-20-0194.1.xml?rskey=wn5MVr&result=3). 


<p align = "center">
<img src = "https://github.com/monte-flora/wofs_ml_severe/blob/main/images/baseline_severe_wind_20210504_2200.gif" width="300" height="300">
</p>
<p align = "center">
Example Severe Wind Forecast on May 4, 2021 @ 22:00-22:30 UTC
</p>


Probabilistic guidance for severe weather has largely been developed in the *spatial probability framework* (Fig. 1). For the WoFS-ML-Severe products, our goal is to a create an event-based probabilistic guidance where the goal is to highlight the likelihood of particular storms producing severe weather threats. 

<p align = "center">
<img src = "https://github.com/monte-flora/wofs_ml_severe/blob/main/images/event_v_spatial.png" width="600" height="300">
</p>
<p align = "center">
Fig. 1: Illustration of distinction between spatial and event reliability of probabilistic forecasts. Event reliability (a) measures the consistency of probabilistic forecasts associated with an individual thunderstorm within an anisotropic neighborhood determined by the forecast ensemble envelope (forecast probabilities [shown in red] are the likelihood of the event occurring). Spatial reliability (b) measures the consistency of probabilistic forecasts of an event occurring within some prescribed neighborhood of a point and are not associated with a specific convective storm (forecast probabilities [shown in red] are the likelihood of the event impacting a particular point)
</p>

The first step is to identify the ensemble storm tracks (Flora et al. 2019, 2021). The flowchart for the object identification method used is shown in Fig. 2. If you are interested in storm-based image segmentation, then check out Methods for Object-based and Neighborhood Threat Evaluation in Python ([MontePython](https://github.com/WarnOnForecast/MontePython)). 

<p align = "center">
<img src = "https://github.com/monte-flora/wofs_ml_severe/blob/main/images/object_id_flowchart.png" width="500" height="400">
</p>
<p align = "center">
Fig. 2: Ensemble Storm Track Identification Algorithm
</p>

Using the ensemble storm tracks, ML predictors are extracted from the WoFS forecast data and observed severe weather reports associated with the tracks are used as labels. The input predictors into WoFS-ML-Severe are based on intra-storm variables (e.g., 2-5 km AGL UH, column-maximum updraft, etc), environmental variables (e.g., CAPE, CIN, SRH, etc), and morphological properties of the track itself (area, minor axis length, etc). From intra-storm variables we extract spatial- and amplitude-based statistics while only spatial statistics are extracted for the environmental variables. The flow chart for the predictor creation is shown in Fig. 3. 

<p align = "center">
<img src = "https://github.com/monte-flora/wofs_ml_severe/blob/main/images/data_preprocessing.png" width="700" height="600">
</p>
<p align = "center">
Fig. 3: Intra-storm and Environmental Predictor Creation Flow Chart
</p>





First, the codebase has the [data pipeline](https://github.com/monte-flora/wofs_ml_severe/blob/main/wofs_ml_severe/data_pipeline/ml_data_generator.py) for building the ML dataset, which includes 

* Identifying the ensemble storm track objects , 
* Deriving the ML features from the WoFS summary files (diagnostic and prognostic variables derived from the WRFOUTs), and 
* Matching the ensemble storm tracks to local tornado, severe hail, and severe wind reports, 

Second, the codebase for fitting the model is found in [wofs_ml_severe.fit](https://github.com/monte-flora/wofs_ml_severe/tree/main/wofs_ml_severe/fit). The fitting relies on a package I developed for implementing pipelines, hyperparameter parameterization, cross-validation, and calibration into a single class ([ml_workflow](https://github.com/WarnOnForecast/ml_workflow)). 
