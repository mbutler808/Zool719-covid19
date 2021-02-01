import csv
import numpy as np
import pandas as pd
import datetime
from datetime import datetime

death = pd.read_csv('https://www.cdc.gov/coronavirus/2019-ncov/downloads/covid-data/2021-01-11-model-data.csv')
death.rename(columns={'target_week_end_date': 'forecast_end_date'}, inplace=True)

cases = pd.read_csv('https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/2021-01-11-all-forecasted-cases-model-data.csv')
cases.rename(columns={'target_end_date': 'forecast_end_date'}, inplace=True)

# hospitalizations = pd.read_csv('https://www.cdc.gov/coronavirus/2019-ncov/downloads/cases-updates/2020-10-12-hospitalizations-model-data.csv')
# hospitalizations.rename(columns={'target_end_date': 'forecast_end_date'}, inplace=True)

def filter_and_clean_df(df, metric, timeframe):

    df_hawaii = df[df['location_name'] == 'Hawaii']

    if metric == 'death':
        df_hawaii_wk = df_hawaii[df_hawaii['target'] == f'{timeframe} wk ahead cum death']
    if metric == 'cases':
        df_hawaii_wk = df_hawaii[df_hawaii['target'] == f'{timeframe} wk ahead inc case']
    if metric == 'hospitalizations':
        df_hawaii_wk = df_hawaii[df_hawaii['target'] == f'{timeframe * 7} day ahead inc hosp']

    df_hawaii_wk = df_hawaii_wk.reset_index(drop=True)

    df_hawaii_wk['forecast_date'] = pd.to_datetime(df_hawaii_wk['forecast_date'])
    df_hawaii_wk['forecast_end_date'] = pd.to_datetime(df_hawaii_wk['forecast_end_date'])

    # List from CDC website
    assume_NPI_change = ['Columbia', 'Covid19Sim', 'Google-HSPH', 'IHME', 'JCB', 'JHU-IDD', 'NotreDame-FRED', 'PSI', 'UCLA', 'YYG']

    df_hawaii_wk['Social_Distance_Assumption'] = ['Changes' if i in assume_NPI_change else 'No_Changes' for i in df_hawaii_wk['model']]

    df_hawaii_wk = df_hawaii_wk[['forecast_date', 'forecast_end_date', 'model', 'point', 'Social_Distance_Assumption']]

    df_hawaii_wk.rename(columns={'forecast_date' : 'forecast_start_date',
                                 'point' : df_hawaii_wk['forecast_end_date'][0].strftime("%-m/%-d/%Y")},
                                 inplace=True)

    return df_hawaii_wk

death_1 = filter_and_clean_df(death, 'death', 1)
cases_1 = filter_and_clean_df(cases, 'cases', 1)
# hospitalizations_1 = filter_and_clean_df(hospitalizations, 'hospitalizations', 1)

death_2 = filter_and_clean_df(death, 'death', 2)
cases_2 = filter_and_clean_df(cases, 'cases', 2)
# hospitalizations_2 = filter_and_clean_df(hospitalizations, 'hospitalizations', 2)

death_3 = filter_and_clean_df(death, 'death', 3)
cases_3 = filter_and_clean_df(cases, 'cases', 3)
# hospitalizations_3 = filter_and_clean_df(hospitalizations, 'hospitalizations', 3)

death_4 = filter_and_clean_df(death, 'death', 4)
cases_4 = filter_and_clean_df(cases, 'cases', 4)
# hospitalizations_4 = filter_and_clean_df(hospitalizations, 'hospitalizations', 4)

forecast_start = cases['forecast_date'].iloc[0]
forecast_end = cases['forecast_end_date'].iloc[0]

deaths_by_week = pd.merge(pd.merge(pd.merge(death_1.iloc[:, [2,3]], death_2.iloc[:, [2,3]]), death_3.iloc[:, [2,3]]), death_4.iloc[:, [2,3]]).set_index('model').T
cases_by_week = pd.merge(pd.merge(pd.merge(cases_1.iloc[:, [2,3]], cases_2.iloc[:, [2,3]]), cases_3.iloc[:, [2,3]]), cases_4.iloc[:, [2,3]]).set_index('model').T


del deaths_by_week.index.name
del cases_by_week.index.name

# change to reflect daily numbers
cases_by_week = (cases_by_week / 7)

datelist = [i.strftime("%-m/%-d/%Y") for i in pd.date_range(death_1['forecast_start_date'][0], periods=28).tolist()]

def create_metric_df(metric):
    dummyarray = np.empty((len(datelist), len(metric.columns)))
    dummyarray[:] = np.nan

    df = pd.DataFrame(dummyarray)
    df.index = datelist
    df.columns = metric.columns

    counter = 0
    for i in df.index:
        if i == metric.iloc[0].name:
            break
        else:
            counter += 1

    df.iloc[counter] = metric.iloc[0]
    df.iloc[counter+7] = metric.iloc[1]
    df.iloc[counter+14] = metric.iloc[2]
    df.iloc[counter+21] = metric.iloc[3]

    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    return df

death_forecast = create_metric_df(deaths_by_week)
case_forecast = create_metric_df(cases_by_week)

death_forecast.to_csv('./cdc_forecasts/death_forecast.csv')
case_forecast.to_csv('./cdc_forecasts/case_forecast.csv')
