# Analysing Time Series Data with the Seasonal Autoregressive Integrated Moving Average (SARIMA) Model

In this repository, we explore the SARIMA model, a powerful statistical technique for
modelling and forecasting time series data, in the form of a Jupyter notebook. Using air passenger traffic data, the notebook details how one can approach time series data in general, from analysing autocorrelation/partial autocorrelation, applying differencing to obtain stationary data, to performing a grid search to derive optimal model parameters.

We take this further by deploying the SARIMA model as a Streamlit web-app to analyse any time series data uploaded by public users. Check out the SARIMA tool here: https://sarima.streamlit.app/.

## Installation ##

The SARIMA model from the Statsmodel library is used. After cloning the github repository, install all dependencies as listed under *requirements.txt*:
```
pip install -r requirements.txt
```

## Running the Application ##

Start the Streamlit web-app locally by running the following command:

```
streamlit run SARIMA.py
```

## Quick Glance at the Tool ##

The user inputs the following:
1. A .csv file containing the time series data in the specified format
2. The number of time steps to forecast and the seasonal period
3. The performance indicator to optimise for the model parameter grid search, namely the Akaike Information Criterion (AIC) or the Root Mean Squared Error (RMSE). 

After the analysis is complete, the tool will display the optimal model parameters obtained from the grid search, as well as the performance results of the SARIMA model. This is accompanied by a line chart showing the original time series data, the forecasted data, and the confidence band (95% confidence). To extract the forecasted data, simply toggle between the tabs and download the tables shown.

<p align="center">
<img src="Analysis Results Sample.png" width="512" height="272">
</p>






