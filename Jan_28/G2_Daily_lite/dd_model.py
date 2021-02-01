#########################################################
############### ~ Import Libraries ~ ####################
#########################################################

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from datetime import datetime, timedelta
from dd_model.model import run_scenario
import requests
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from datetime import datetime, date
from dateutil.parser import parse
from bokeh.models import HoverTool
from bokeh.models.widgets import Tabs, Panel
import csv
from api.covid_api import get_state_data
from functions.ndays import ndays
from sheets_api.sheets import get_daily_positive_cases
import json
import time
from bokeh.resources import CDN
from bokeh.embed import file_html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from hdc_api.sheets import get_test_positivity, get_rt, get_cases

#########################################################
################### ~ Load Dfs ~ ########################
#########################################################

# Formats dates to reflect the following example: 9/7/2020 or 2020-9-7 (# or - represents removing 0s)
format_date_str = "%#m/%#d/%Y" # PC
#format_date_str = "%-m/%-d/%Y" # Mac/Linux

# Load historical COVID data from Coivdtracking.org (Updates ~13:00 HST)
state_df = get_state_data('HI', format_date_str)

# Load daily new cases from https://docs.google.com/spreadsheets/d/1Q4y14VoTvawco3cI4Qm-zP9FJSXa79CVlUSTybubANY/edit#gid=2124927884
daily_new_cases = get_daily_positive_cases()

# test_case_df = get_cases()
# test_case_df = test_case_df[test_case_df['Region'] == 'State']
# test_case_df = test_case_df[['Date', 'Cases_New']]

#########################################################
##################### ~ Set Dates ~ #####################
#########################################################

# Used for JSON update due to potential lag from today's date to model begin (resulting from covidtracking.org not updating until 1pm)
todays_date = str(datetime.now().strftime(format_date_str))

# Use for CSV creation
todays_date_f_string = todays_date.replace('/', '.')

# Set's 'today' (when the forecast output begins based on available historic data) - used to initialize start of the forecast
data_lag = 0

if list(state_df['date'])[-1] != todays_date:
    # sets the 'today' to the beginning of historic data start to ensure no data gap
    data_lag = (pd.to_datetime(todays_date) - list(state_df['date'])[-1]).days
else:
    data_lag = 0

today = str((datetime.now() - timedelta(days = data_lag)).strftime(format_date_str))

# Set initialization days and length of the forecast (recommend keeping consistent and only change situationally for specific scenarios)
initialization_days = 15
forecast_length = 13

Model_Begin = str((datetime.now() - timedelta(days = initialization_days)).strftime(format_date_str))
Model_End = str((datetime.now() - timedelta(days = -forecast_length)).strftime(format_date_str))

# Calculates time difference between model start and current date (used in data cleaning function)
ndays_today_Model_Begin = (ndays(Model_Begin, today))

#########################################################
################# ~ Set Parameters ~ ####################
#########################################################

# Model parameters used to move from Reported New Cases to Estimated Number of Initial Infections
shift_days = 7
cases_scale = 7

# Populations:
oahu = 953207
all_islands = 1415872

# Set Rt values for initalization, pessimistic, and expected scenarios
rt_initialization = 1.045
rt_estimate_pessimistic = 1.05
rt_estimate_expected = 1.0

# Set parameters
incubation = 3
infectious_duration = 6
delay = 3
hosp_stay = 7
ICU_stay = 10
hospitalization_rate = 0.0135
hospitalization_ICU_rate = 0.156
ICU_hosp_rate = 0.3

# Set [Exposed, Infected] travelers for each day in respective range of dates
travel_values  = [[4, 0], # rt_initialization - rt_estimate
                  [3, 0]] # rt_estimate - Model_End

# Set how much historical data is included in df & number of rolling days for reported new cases average
historical_days = 30
rolling_mean_days = 7

# Set how many days of Reported New Cases are summed to get the Active Cases for Quarantine
rolling_sum_days = 14

#########################################################
##### ~ Get Values for Initial Compartment Vector ~ #####
#########################################################

# To start calculation for Estimated Number of Initial Infections,
# get the first day in day range equal to range of duration of infectious period,
# which when summed will account for total persons in the I compartment (infected) based on the Model Begin date
start_index = [e for e, i in enumerate(daily_new_cases['Date']) if i == pd.to_datetime(Model_Begin) + timedelta(shift_days - infectious_duration)][0]

# Sum Reported New Cases for duration of infection,
# then scale by the cases_scale factor to estimate true number of infected.
initial_infections = daily_new_cases[start_index : start_index + (infectious_duration + 1)]['Cases'].sum() * cases_scale

# Get initial values from historical data for hospitalizations, ICU, and deaths
initial_hospitalizations = int(state_df['hospitalizedCurrently'][state_df['date'] == Model_Begin])
initial_ICU = int(state_df['inIcuCurrently'][state_df['date'] == Model_Begin])
initial_Fatal = int(state_df['death'][state_df['date'] == Model_Begin]) + 12

#########################################################
########### ~ Initialize Code for Model Run ~ ###########
#########################################################

def run_model(island, rt_change_dates, rt_change_values, travel_values):
    """
    Runs model based from Model Begin to Model End with Rt changing at stated points throughout the run.
    Outputs a dataframe from today's date to Model End.
    """

    def rt_ndays(rt_change_dates):
        """
        Get the number of days between each inputted period and the Model Begin
        """
        rt_change_ndays = []

        for i in rt_change_dates:
            rt_change_ndays.append(ndays(Model_Begin, i))

        return rt_change_ndays

    rt_change_ndays = rt_ndays(rt_change_dates)

    # Zip dates and Rt values together for future processing
    zipped_rt_ndays = list(zip(rt_change_ndays, rt_change_values))


    def infected_travelers():
        """
        Creates a list of the number of travelers expected to be traveling to Hawaii
        that are exposed or infected with COVID that will enter undetected
        """
        # Calculate number of days for each period
        travel_dates = rt_change_dates + [Model_End]
        travel_dates = [ndays(travel_dates[i], travel_dates[i + 1]) for i in range(len(travel_dates) - 1)]

        # Create Travel array
        Travel_expected = []
        for e, dates in enumerate(travel_dates):
            for i in range(dates):
                Travel_expected.append(travel_values[e])

        Travel_expected = [[e] + value for e, value in enumerate(Travel_expected)]

        return Travel_expected

    Travel_expected = infected_travelers()

    # Runs model code in ./dd_model/model.py with previously stated parameters
    data = run_scenario(Model_Begin,
                        Model_End,
                        initial_hospitalizations,
                        initial_ICU,
                        initial_Fatal,
                        island, # Size of population
                        initial_infections,   # Number of initial infections
                        incubation,       # Length of incubation period (days) # !age distributable
                        infectious_duration,       # Duration patient is infectious (days)
                        delay,       # Time delay for severe cases (non-icu) to be hospitalized
                        hosp_stay,       # Length of hospital stay (recovery time for severe cases)
                        ICU_stay,      # Length of hospital stay for ICU cases
                        hospitalization_rate,   # Percent of infectious people who go to hospital
                        hospitalization_ICU_rate,    # Percent of Hospitalizations that flow to ICU (remainder go to Recovered (R))
                        ICU_hosp_rate,     # Percent of ICU that flow to back to Hospitalizations (remainder go to Fatal (F))
                        zipped_rt_ndays,
                        Travel_expected)

    # Create df from model output
    data_columns = ['Date', 'Cases', 'Hospitalized', 'ICU', 'Fatal', 'Rt', 'Infected_Travels', 'Exposed_Travelers', 'Susceptible', 'Total_Infected', 'New_Hospitalizations']
    df = pd.DataFrame(data, columns = data_columns)
    df = df[['Date', 'Cases', 'Hospitalized', 'ICU', 'Fatal', 'Rt', 'Susceptible', 'Total_Infected', 'New_Hospitalizations']]

    df['New_Hospitalizations'] = [i - df['New_Hospitalizations'].iloc[e-1] if e >= 1 else 0 for e, i in enumerate(df['New_Hospitalizations'])]

    return df

#########################################################
#################### ~ Run Model ~ ######################
#########################################################

# Date Rt for pessimistic / expected begins. Starts ~1 week prior to today's date to smooth curve
rt_estimate_start = str((datetime.now() - timedelta(days = 9)).strftime(format_date_str))

# Creates new variable from Model Begin date to better align with verbiage in function
initialization_start = Model_Begin

# Run pessimistic & expected scenarios
pessimistic_14 = run_model(all_islands, # Select which population to use in simulation
                    [initialization_start, rt_estimate_start], # Dates for Rt changes
                    [rt_initialization, rt_estimate_pessimistic], # Rt values beginning on above dates
                    travel_values)
expected_14 = run_model(all_islands,
                       [initialization_start, rt_estimate_start],
                       [rt_initialization, rt_estimate_expected],
                       travel_values)

#########################################################
############# ~ Add Reported New Cases ~ ################
#########################################################

# Initialize cleaning function
def add_reported_new_cases(scenario, df):
    """
    Calculates (using total infected and scaling back down according to cases_scale)
    and adds 'Reported_New_Cases' to each scenario
    """
    # Change date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Create 'active cases' from infected individuals
    df['Cases'] = df['Cases'] / cases_scale

    # Create 'daily new cases' from 'total infected'
    df['Reported_New_Cases'] = np.concatenate((np.zeros(1+shift_days),np.diff(df['Total_Infected'] / cases_scale)))[:len(df['Date'])]

    # Create column labeling scenario
    df['Scenario'] = scenario

    # Start output at today's date
    df = df.loc[ndays_today_Model_Begin:]

    return df

# Run add_reported_new_cases for both scenarios
pessimistic_14 = add_reported_new_cases('Pessimistic', pessimistic_14)
expected_14 = add_reported_new_cases('Expected', expected_14)

#########################################################
################# ~ Add Active Cases ~ ##################
#########################################################

# Initialize function that calculates active cases
def get_active_cases(model_output, daily_reported_cases, scenario):
    """
    Concatenates model output data with reported cases, then applies a rolling sum (rolling_sum_days)
    reported cases to create 'Active_Cases'
    """
    model_output = model_output[['Date', 'Reported_New_Cases']]
    model_output.columns = ['Date', 'Cases']

    daily_reported_cases = daily_reported_cases[['Date', 'Cases']][:-1]

    active_df = pd.concat([daily_reported_cases, model_output])
    active_df['Cases'] = active_df['Cases'].round()
    active_df['Active_Cases'] = active_df['Cases'].rolling(rolling_sum_days).sum().astype(float)
    active_df = active_df.reset_index(drop=True)
    return active_df

# Run get_active_cases for both scenarios
pessimistic_active = get_active_cases(pessimistic_14, daily_new_cases, 'Pessimistic')
expected_active = get_active_cases(expected_14, daily_new_cases, 'Expected')

#########################################################
############## ~ Create Graphs (Bokeh) ~ ################
#########################################################

# Create 7 day moving average for Daily Reported New Cases
daily_new_cases['Rolling_Average_Cases'] = daily_new_cases['Cases'].rolling(rolling_mean_days).mean()

# Reduce daily new cases df to length of selected period
daily_new_cases_len = len(daily_new_cases.index)
daily_new_cases_historical = daily_new_cases.loc[(daily_new_cases_len - (historical_days + (rolling_mean_days - 1))):]

daily_new_cases_historical_graph = daily_new_cases_historical.rename(columns={'Rolling_Average_Cases': 'Reported_New_Cases'})
daily_new_cases_historical_graph['Scenario'] = 'Historical'

# Reduce active cases df to length of selected period
pessimistic_active_len = len(pessimistic_active.index)
active_historical = pessimistic_active.loc[daily_new_cases_historical_graph.index[0]:daily_new_cases_historical_graph.index[-1]]

# Reduce state df to length of selected period
state_df_len = len(state_df.index)
state_df = state_df.reset_index(drop=True)
state_df_historical = state_df.loc[(state_df_len - historical_days):]

state_df_historical_graph = state_df_historical[['date', 'hospitalizedCurrently', 'death', 'inIcuCurrently']]
state_df_historical_graph.columns = ['Date', 'Hospitalized', 'Fatal', 'ICU']
state_df_historical_graph['Scenario'] = 'Historical'

# Set Y axis max
max_hosp = pd.concat([pessimistic_14['Hospitalized'], state_df_historical_graph['Hospitalized']]).astype(int).max() * 1.1
max_ICU = pd.concat([pessimistic_14['ICU'], state_df_historical_graph['ICU']]).astype(int).max() * 1.1
max_Fatal = pd.concat([pessimistic_14['Fatal'], state_df_historical_graph['Fatal']]).astype(int).max() * 1.5
max_Reported_New_Cases = pd.concat([pessimistic_14['Reported_New_Cases'], daily_new_cases_historical_graph['Reported_New_Cases']]).astype(int).max() * 1.1
max_Active_Cases = pd.concat([pessimistic_active['Active_Cases'][-15:], active_historical['Active_Cases']]).astype(int).max() * 1.1

# Initialize function to display forecast graphs
def forecast_graph():
    """
    Creates tabs for each metric and displays output to html
    """

    def initialize_plotting_function(y_metric):
        """
        Initializing function for plotting historical data + model output data
        """
        # Set data sources
        source_pessimistic_14 = ColumnDataSource(pessimistic_14)
        source_expected_14 = ColumnDataSource(expected_14)

        source_daily_new_cases_historical = ColumnDataSource(daily_new_cases_historical_graph)
        source_state_df_historical = ColumnDataSource(state_df_historical_graph)

        # Creates interactive hover
        tooltips = [
                ('Scenario', '@Scenario'),
                (f'{y_metric}',f'@{y_metric}'),
                ('Date', '@Date{%F}')
               ]

        y_max = 0

        if y_metric == 'Hospitalized':
            y_max = int(max_hosp)
        if y_metric == 'ICU':
            y_max = int(max_ICU)
        if y_metric == 'Fatal':
            y_max = int(max_Fatal)
        if y_metric == 'Reported_New_Cases':
            y_max = int(max_Reported_New_Cases)

        # Initalize plot foundation
        p = figure(x_axis_type = "datetime", y_range=(0, y_max))

        # Add historical lines
        if y_metric == 'Hospitalized':
            historical_hosp_line = p.line(x='Date', y=f'{y_metric}',
                 source=source_state_df_historical,
                 line_width=2, color = 'grey')
            p.add_tools(HoverTool(renderers=[historical_hosp_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
        if y_metric == 'Fatal':
            historical_death_line = p.line(x='Date', y=f'{y_metric}',
                 source=source_state_df_historical,
                 line_width=2, color = 'grey')
            p.add_tools(HoverTool(renderers=[historical_death_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
        if y_metric == 'ICU':
            historical_ICU_line = p.line(x='Date', y=f'{y_metric}',
                 source=source_state_df_historical,
                 line_width=2, color = 'grey')
            p.add_tools(HoverTool(renderers=[historical_ICU_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
        if y_metric == 'Reported_New_Cases':
            historical_daily_new_cases_line = p.line(x='Date', y=f'{y_metric}',
                source=source_daily_new_cases_historical,
                line_width=2, color = 'grey')
            p.add_tools(HoverTool(renderers=[historical_daily_new_cases_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))

        # Add forecast lines
        pessimistic_14_line = p.line(x='Date', y=f'{y_metric}',
                 source=source_pessimistic_14,
                 line_width=2, color='firebrick', legend='Pessimistic')
        expected_14_line = p.line(x='Date', y=f'{y_metric}',
                 source=source_expected_14,
                 line_width=2, color='steelblue', legend='Expected')

        p.add_tools(HoverTool(renderers=[expected_14_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))
        p.add_tools(HoverTool(renderers=[pessimistic_14_line], tooltips=tooltips, mode='vline', formatters={'Date': 'datetime'}))

        # Add Graph details
        p.title.text = f'Number of {y_metric}'
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = f'{y_metric}'

        # Sets graph size
        p.plot_width = 1200
        p.plot_height = 700

        # Sets legend
        p.legend.location = "top_left"
        p.legend.click_policy="hide"

        return p

    # Create panels for each tab
    cases_tab = Panel(child=initialize_plotting_function('Reported_New_Cases'), title='Reported New Cases')
    hospitalized_tab = Panel(child=initialize_plotting_function('Hospitalized'), title='Hospitalized')
    ICU_tab = Panel(child=initialize_plotting_function('ICU'), title='ICU')
    Fatal_tab = Panel(child=initialize_plotting_function('Fatal'), title='Fatal')
    Susceptible_tab = Panel(child=initialize_plotting_function('Susceptible'), title='Susceptible')

    # Assign the panels to Tabs
    tabs = Tabs(tabs=[Susceptible_tab, cases_tab, hospitalized_tab, ICU_tab, Fatal_tab])

    return tabs

# Display forecast graphs
show(forecast_graph())
