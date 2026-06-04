_data_helper_ is a python file designed to handle the underlying data. It is called almost everytime in the other notebooks.

_data_analysis_ contains the code used to evaluate our underlying data and validate our methods methods/param choices.

_weather_data_ contains the code snippet used to fetch the 2 years of weather data through Open-Meteo's ICON Seamless model, then the transformation into our needed population-weighted average.

_optuna_colab_ and Optuna_local contains the Optuna optimization code used to find our suitable hyperparameters for the models used.

_xgboost_ridge_ contains the code for running and saving the results of the best xgboost global and 24-expert models as well as the best ridge regression setup.

_xLinear_ contains the ablation study and the final full feature matrix setup's results with some additional error and result investigation.

_results_ we ingest all the previous result files here to construct plots and tables in an effort to explain our findings.
