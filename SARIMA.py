import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
import warnings
import itertools
import math
import statsmodels.api as sm

### Intro ###

df_sample = pd.DataFrame(
    {'Date':[datetime.datetime(2025,x,1) for x in range(1,6)],
     'Value':[np.random.randint(1000,5000) for x in range(1,6)]})

st.title(":chart_with_upwards_trend: Forecasting Times Series With SARIMA ")
st.markdown(

"""

The Seasonal Autoregressive Integrated Moving Average (SARIMA) model is a powerful statistical technique for
modelling and forecasting time series data. It is based on four key components:

- **Autoregressive (AR)**, which uses past values of the time series to forecast future values
- **Moving Average (MA)**, which uses the model's past forecast errors to forecast future values 
- **Integrated (I)**, which indicates differencing the time series to make it stationary
- **Seasonal (S)**, which captures temporal variations that occur at regular intervals.
For every AR, MA, and I order of the model, there is a seasonal equivalent
"""
)

st.header("What Does This Tool Do?")
st.markdown(

"""

This tool uses the SARIMA model from the statsmodels library to perform time series forecasts. To optimise
the model parameters used for the forecast, it performs a grid search of model parameters within a pre-defined
range, and identifies the parameter combination with the best performance (either AIC or RMSE). The ranges used
for the grid search are as follows:

- **Autoregressive Order (p)**, range from 0 to 2
- **Differencing Order (d)**, range from 0 to 2
- **Moving Average Order (q)**, range from 0 to 2
- **Seasonal Autoregressive Order (P)**, range from 0 to 1
- **Seasonal Differencing Order (D)**, fixed at 0
- **Seasonal Moving Average Order (Q)**, range from 0 to 1

Note: The **seasonal period (m)** is an input by the user.
"""
)

st.header("How Does This Tool Work?")
st.markdown(
"""

1. The user uploads a .csv file containing the time series data. This file should contain two columns: the first column
should store the date (in datetime format), while the second column should store the values to be forecasted. The time step between each row
of data should be consistent (e.g. monthly data). For optimal performance, the time series data should contain no
more than 1000 rows of data. A sample of the input .csv file is shown below:
"""
)
st.dataframe(df_sample, use_container_width=True)

st.markdown(
"""
2. The user specifies the number of time steps to forecast, the seasonal period (e.g. for monthly data, if there is an annual cycle,
this would likely be 12; if there is a quarterly cycle, this would likely be 3).

3. The user selects the performance indicator to optimise for the model parameter grid search. Currently, two performance indicators
are supported:

    - **Akaike Information Criterion (AIC)**: The AIC is a statistical measure used to evaluate the quality of a model by balancing its
    goodness of fit against its complexity. It quantifies how well a model explains the observed data through a likelihood term, while
    imposing a penalty proportional to the number of parameters to discourage overfitting; the formula is defined as AIC = 2 × (number of parameters)
    \- 2 × (log-likelihood). A lower AIC value indicates a preferable model, reflecting an optimal trade-off between predictive accuracy and simplicity.
    
    - **Root Mean Square Error (RMSE)**: The RMSE is a metric that assesses the accuracy of the model’s predictions by calculating the square root of
    the average squared differences between predicted and actual values. It emphasises larger errors due to the squaring operation and provides a result
    in the same units as the data, offering a direct interpretation of prediction error magnitude.

4. After the analysis is complete, the tool will display the optimal model parameters obtained from the grid search, as well as the performance
results of the SARIMA model. This is accompanied by a line chart showing the original time series data, the forecasted data, and the confidence band (95% confidence) for
the forecasted data. To extract the forecasted data, simply toggle between the tabs and download the tables shown.

"""
)

st.write("")
st.image("Analysis Results Sample.png", caption = 'Analysis Results Sample')

### User Inputs ###

st.header("Input Data")
df = None
uploaded_file = st.file_uploader("Upload a csv file:", type = "csv")

try:
    df = pd.read_csv(uploaded_file)
except:
    st.warning("Please upload a csv file.")

st.write("")
steps = st.number_input(
    "Enter the number of time steps to forecast:", value = None, placeholder = "Enter an integer higher than 0..."
)

try:
    steps = int(steps)
except:
    st.warning("Please enter the number of steps to forecast.")

st.write("")
seasonal_period = st.number_input(
    "Enter the seasonal period:", value = None, placeholder = "Enter an integer higher than 0..."
)

try:
    seasonal_period = int(seasonal_period)
except:
    st.warning("Please enter the seasonal period of the time series.")

st.write("")
perf_indicator = st.radio(
    "Select performance indicator to optimise for parameter grid search:",
    ["Akaike Information Criterion (AIC)","Root Mean Square Error (RMSE)"],
    captions=[
        "AIC scores how well the model fits the data while penalising it for being too complicated, with lower scores meaning a better balance of accuracy and simplicity.",
        "RMSE calculates the average distance between the model’s predictions and real values, showing prediction error in the same unit and in a way that is easy to interpret.",
    ],
)

ind =  None
if perf_indicator == "Akaike Information Criterion (AIC)":
    ind = "AIC"
elif perf_indicator == "Root Mean Square Error (RMSE)":
    ind = "RMSE"

### Run Forecast ###

if isinstance(df, pd.DataFrame) and steps and seasonal_period and ind:

    if st.button("Run forecast", type = "primary"):

        with st.spinner("Running forecast..."):

            col1, col2 = df.columns
            df[col1] = pd.to_datetime(df[col1])

            ### Perform Optimisation ###

            train_len = int(len(df) * 0.8)
            train, test = df[:train_len], df[train_len:]

            # Define the range of values for p, d, q, P, D, Q, and m
            p_values = range(0, 3)          # Autoregressive order
            d_values = range(0, 3)          # Differencing order
            q_values = range(0, 3)          # Moving average order
            P_values = range(0, 2)          # Seasonal autoregressive order
            D_values = range(0, 1)          # Seasonal differencing order
            Q_values = range(0, 2)          # Seasonal moving average order
            m_values = [seasonal_period]    # Seasonal period

            param_combinations = list(itertools.product(p_values, 
                                                        d_values, 
                                                        q_values, 
                                                        P_values, 
                                                        D_values, 
                                                        Q_values, 
                                                        m_values))

            best_aic = float("inf")  
            best_params_aic = None

            best_rmse = float("inf")
            best_params_rmse = None

            train_rmse_results = {}

            for params in param_combinations:

                order = params[:3]
                seasonal_order = params[3:]
    
                try:
                    model = sm.tsa.SARIMAX(train[col2], 
                                        order=order, 
                                        seasonal_order=seasonal_order)
                    result = model.fit(disp=False)
                    aic = result.aic

                    forecast = result.get_forecast(steps=len(test))
                    rmse = np.sqrt(np.mean((forecast.predicted_mean - test[col2])**2))
                    
                    train_rmse_results[params] = rmse

                    # Ensure the convergence of the model
                    if not math.isinf(result.zvalues.mean()):
                        
                        print('Order: {} Seasonal Order: {} AIC: {:.2f} RMSE: {:.2f}'.format(order, seasonal_order, aic, rmse))
                    
                        if aic < best_aic:
                            best_aic = aic
                            best_params_aic = params

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params_rmse = params
                            
                    else:
                        print('Order: {} Seasonal Order: {} Unable to converge.'.format(order, seasonal_order))
                        print(order, seasonal_order, 'not converged')

                except:
                    continue

            if ind == "AIC":
                best_params = best_params_aic
            
            elif ind == "RMSE":
                best_params = best_params_rmse

            model = sm.tsa.statespace.SARIMAX(df[col2],order = best_params[:3],seasonal_order = best_params[3:])
            model_fit = model.fit()

            results_table1 = pd.read_html(model_fit.summary().tables[0].as_html(), header=None)
            no_observations = results_table1[0][3][0]
            log_likelihood = results_table1[0][3][1]
            AIC = results_table1[0][3][2]
            BIC = results_table1[0][3][3]
            HQIC = results_table1[0][3][4]

            p, d, q, P, D, Q, m = best_params_aic

            forecast = model_fit.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()

            step_delta = (df[col1] - df[col1].shift(1)).dropna().mean()
            forecast_period = [df[col1].iloc[-1] + step_delta * i for i in list(range(1,steps+1))]
            confidence_intervals[col1] = forecast_period
            confidence_intervals['mean '+ col2] = forecast_values

            df['Predicted'] = False
            forecast_values_df = pd.DataFrame({col1:forecast_period,col2:forecast_values})

            forecast_values_df['Predicted'] = True
            df_viz = pd.concat([df, forecast_values_df], ignore_index = True)

            df_params = pd.DataFrame(
                {
                    "parameter": ["Autoregressive Order",
                                "Differencing Order",
                                "Moving Average Order",
                                "Seasonal Autoregressive Order",
                                "Seasonal Differencing Order",
                                "Seasonal Moving Average Order",
                                "Seasonal Period"],
                    "value": [p, d, q, P, D, Q, m]
                }

            )

            df_results = pd.DataFrame(
                {
                    "metric": ["No. of Observations",
                            "Log Likelihood",
                            "AIC",
                            "RMSE (on training data)",
                            "BIC",
                            "HQIC"],
                    "value": [str(np.round(no_observations,0)),str(np.round(log_likelihood,0)),str(np.round(AIC,0)),str(np.round(train_rmse_results[best_params],0)),str(np.round(BIC,0)),str(np.round(HQIC,0))]
                }
            )

            ### Visualisation ###

            y_min = df[col2].min() * 0.95
            y_max = df[col2].max() * 1.05

            chart = alt.Chart(df_viz).mark_line().encode(
                x=col1,
                y = alt.Y(col2, scale = alt.Scale(domain=[y_min,y_max])),
                strokeDash='Predicted:N'
            ).interactive()

            cb_label1 = 'lower ' + col2
            cb_label2 = 'upper ' + col2

            confidence_band = alt.Chart(confidence_intervals).mark_area(opacity=0.3,color='#FFA500').encode(
                alt.X(col1).title(col1),
                alt.Y(cb_label1).title(col2),
                alt.Y2(cb_label2)
            ).interactive()

            st.header("Analysis Results")

            tab1, tab2, tab3 = st.tabs(["Chart", "Dataframe", "Confidence Band"])

            with tab1:
                st.altair_chart(chart + confidence_band, theme='streamlit', use_container_width=True)

            with tab2:
                st.dataframe(df_viz, use_container_width=True)

            with tab3:
                st.dataframe(confidence_intervals[[col1, 'mean '+col2, 'lower '+col2, 'upper '+col2]], use_container_width=True)

            st.subheader('Model Parameters')
            st.dataframe(
                df_params,
                column_config = {
                    "parameter": "Parameter",
                    "value": "Value"
                },
                hide_index = True,
                use_container_width = True,
            )

            st.subheader('Model Results')
            st.dataframe(
                df_results,
                column_config = {
                    "metric": "Metric",
                    "value": "Value"
                },
                hide_index = True,
                use_container_width = True,
            )