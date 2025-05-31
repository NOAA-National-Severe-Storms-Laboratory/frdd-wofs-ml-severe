# experiments

Scripts and notebooks to conduct various experiments for dataset generation, model training, evaluation, and monitoring.

## build_the_dataset/
- Scripts to construct training datasets from raw WoFS data.
- Includes:
  - `official_run_data_pipeline.py`: full pipeline runner.
  - `train_test_splitter.ipynb`: splitting data into train/test sets.

Using MLDataPipeline, the dataset can be built in pieces
- Identifying storm tracks 
- Computing the ML features from the WoFS data 
- Matching tracks to storm reports, MESH, or warnings 
- Concatenating everything together 

## download_storm_events
Notebooks to download storm event data and warning polygons 

## explainability 
Random notebooks and scripts related to explainability 
- `MESH_swaths_with_reports.ipynb` : Produce Fig. 3 from Flora et al. (2025, WAF)

## fit_ml_models/
- Scripts to train ML models using different configurations.
- Includes:
  - `official_train_ml_models.py`: primary training script.
  - `fit_weighted_average_classifier.py`: ensemble weighting.
  - Hyperparameter tuning (`evaluate_hyperopt_results.ipynb`)

## fit_bl_models/
- Fit baseline models such as climatology or persistence models.

## evaluate_the_models/

- Evaluation notebooks for real-time and retrospective verification:
  - `official_evaluation.ipynb` : Pretty verification graphics for presentations
  - `regression_evaluator.ipynb` : Evaluating the hail size algorithm 
  - `mesh_probs_vs_lsr_probs.ipynb` : Produce Fig. 8 from Flora et al. (2025, WAF) 
  - `official_combined_verification.ipynb`: Produce Fig. 4,5,7 from Flora et al. (2025, WAF)

## monitoring/
- Monitor data pipeline health and track potential data drift or processing errors.
- Compare dataset versions.

## prune_dataset/
- Tools to prune datasets for training data cleanup.
