# Use Cases in MLGEO


GEO-SMART project created a collection of JupyterBooks for [use cases](https://geo-smart.github.io/scm_geosmart_use_case/intro.html).

We also list below examples of final projects in the class.

* **Seasonal Ice Velocity Prediction** from Claire Jensen (UW):
    * Time series forecasting of surface ice velocity values on Zachariæ Isstrøm, Northeast Greenland. Ice velocity data are satellite-derived estimates collected through the [Greenland Ice Sheet Mapping Project (GrIMP)](https://nsidc.org/grimp). Three time series were aggregated from different points on the glacier and were trained and tested separately, finding different optimal parameters and models for each point. Various classic and deep learning models were explored, including LightGBM, RandomForest, LSTM, CNN, and transformers (Moirai).
    * GitHub [repository](https://github.com/cjense/seasonal-icevelocity-prediction)
    * [Report](https://github.com/cjense/seasonal-icevelocity-prediction/blob/main/report%20and%20presentation/ESS_569_Final_Report.pdf) (version tag 1cc81dd)
    * [Presentation](https://github.com/cjense/seasonal-icevelocity-prediction/blob/main/report%20and%20presentation/ESS-569%20Final%20Presenation.pdf) (version tag 1cc81dd)

![Glacier Velocity Time Series](./jensen.png)
The 12-month rolling average of the 6-day frequency velocity predictions from the CNN (train in purple, test in yellow) and LSTM (train in red, test in green) and the ground truth (blue).
