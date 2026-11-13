"""Shared code of the research/{board row}/experiment.py scripts.

common       paths (research/result/{experiment}), JSON-lines logs, official scoring, result files
table        df.parquet loading: causal feature ffill, training-label ffill, origin x 72-lead grids
calibration  prior/calibration split and the frozen per-lead offset
objectives   XGBoost custom objectives shared by more than one board row
continuation A900 + direct-S continuation boosting (scap / splus)
gbm          the 24-lead-block XGBoost runner every GBM board row uses (train, predict-only, smoke)
baseline     the ACE SWEPAM statistical baselines (naive, seasonal naive, mean reversion, SES, AutoARIMA)
"""
