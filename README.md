# Wind Power Forecasting

The aim of this demo is to create a power generation prediction model and dashboard
hosted on the GCP App Engine for a wind turbine located in Turkey.

## Description

Wind energy is clean, inexhaustible, inexpensive and widely distributed. However,
the randomness and intermittency of wind lead to uncertainty in wind power generation,
which brings challenges to the corresponding energy management system and affects the
reliability of the entire power grid. Therefore, accurate estimates of wind power curves,
which show the non-linear relationship between wind speed and wind power, are required
for effective integration of wind power into the power system, as well as wind turbine
condition monitoring projects.

Accurate prediction of wind power is critical to increasing the utilization of wind
in the electricity grid. It also helps power system operators to plan unit commitment,
economic scheduling, and dispatch. In general, an accurate power curve is conducive
to wind power prediction. So, a suitable power curve results in more accurate power forecasts.

## Evaluation

MAAPE is used to evaluate intermittent production forecasts. That is, forecasts
for irregular levels of production. MAAPE is used over MAPE as MAPE shouldn't be
used to calculate the accuracy of an energy production forecast. A lower MAAPE
value indicates a more accurate model. 

In addition to MAAPE, RMSE and $R^2$ Score are also used for evaluation of the
regression models.

## Deployment

### GCP
This project has been deployed on GCP App Engine using the `app.yaml` config file.
This file acts as a deployment descriptor of the service version. Whenever new code is
pushed to a selected branch, GCP Cloud Build runs a build job as a container that
completes the CICD pipeline to deployment. The config file `cloudbuild.yaml` executes
the build from the selected repository on the GCP infrastructure. To automate this
process, a trigger needs to be created and this is where the defined repository and
branch are used as a source for deployment onto the App Engine. Once any code is
pushed to the selected branch, the build job will be triggered and the dashboard deployed.
The GCP deployment was done as a test case to gain familiarity and will be depreciated
once credits run out. The config files will be kept in the repository regardless.

### AWS
This project will be moved to an AWS deployment instance at a future date.

## Getting Started

### Installing

1. git clone/ download zip
2. cd into dir
3.
```
pip install -r /path/to/requirements.txt
```

### Executing program

- Local Dashboard
```
python main.py
```
- Notebooks
    - EDA: `root_dir/notebooks/eda.ipynb`
    - ARIMA Model: `root_dir/notebooks/time-series.ipynb`
    - Regression Models: `root_dir/notebooks/training.ipynb`
    - Inference: `root_dir/notebooks/inference.ipynb`

## Authors

[Jovinder Singh](https://github.com/jovi-s/)

## Acknowledgments

- [Wind Turbine Scada Dataset](https://www.kaggle.com/berkerisen/wind-turbine-scada-dataset/code?datasetId=133415&sortBy=voteCount)
