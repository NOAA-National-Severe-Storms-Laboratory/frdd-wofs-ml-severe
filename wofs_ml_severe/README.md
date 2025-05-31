# wofs_ml_severe

Core package containing all of the ML logic, data pipelines, and utilities for the WoFS-ML-Severe system.
The execution of these code are found in `frdd-wofs-ml-severe/experiments`. 


### common/
- `calibration.py` : Copy of sklearn's calibration.py; legacy and not used.
- `stacking_classifier.py` : script for running a stacked (ensemble) classifier
- `multiprocessing_utils.py`: Scripts for parallel processing 
- `util.py` : random, but useful utility functions 
- `emailer.py` : script for sending secure emails 

### conf/
- ML configuration YAML files for different model versions and time periods.
- Contains training, realtime, and retro config files used throughout the system.

### data_pipeline/

Contains the data processing pipelines and the primary code for 
the operational WoFS-ML-Severe (`ml_data_generator.py`)

- Data processing pipeline:
  - Object-based segmentation of ensemble tracks.
  - Feature extraction from storm objects.
  - Storm report matching.
  - Data ingestion and splitting.
- Primary scripts:
  - `ml_data_pipeline.py`: full pipeline for ML feature generation.
  - `ensemble_track_segmentation.py`: object identification from WoFS tracks.
  - `storm_based_feature_extracter.py`: compute storm object features.
  - `report_matcher.py`: associate storm objects with storm reports.
  - `ml_2to6_data_pipeline.py`: special 2-6hr dataset variant.
  - `storm_report_downloader.py`: automated downloading of SPC reports.

### evaluate/
- `metrics.py`: evaluation metrics and scoring functions (e.g., reliability, performance diagrams).

### explain/

Contains the notebook for generating the global and local explainability 
graphics for the cbWoFS (`cbwofs_explainability_graphics.ipynb`).
The latest graphics are stored in `new_graphics_2024_v2`. 

- Notebooks and scripts for model explainability:
  - SHAP analysis
  - Permutation importance
  - Coefficient analysis (for linear models)
  - Data drift monitoring

### fit/
- Model training routines:
  - `ml_trainer.py`: full ML training logic.
  - `ml_configuration.py`: reads configuration files and initializes training sessions.

### io/

- Loading and saving models, including TensorFlow models.
- File I/O utilities for reading forecast data.

Also contain jsons for the training-testing case splits in json files. 


### json/
- Contains details for the global and local explainability graphics on the cbwofs webviewer. Includes
the minimum and maximum value ranges for each input features and their rounding level. 

### viz/
- Visualization notebooks and output files related to ensemble tracks and ML predictions.

