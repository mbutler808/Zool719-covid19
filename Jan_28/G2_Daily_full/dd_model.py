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
import chart_studio.plotly as py
import chart_studio
from hdc_api.sheets import get_test_positivity, get_rt, get_cases

#########################################################
################### ~ Load Dfs ~ ########################
#########################################################

# Formats dates to reflect the following example: 9/7/2020 or 2020-9-7 (# or - represents removing 0s)
# format_date_str = "%#m/%#d/%Y" # PC
format_date_str = "%-m/%-d/%Y" # Mac/Linux

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

#########################################################
############## ~ Create Graphs (Plotly) ~ ###############
#########################################################

####################### ~ Test ~ #######################

# Load CDC forecasts
def create_forecast_graphs(cdc_metric, df, df_column, legend_name, max_metric, chart_studio_name):
    if (cdc_metric == 'case') or (cdc_metric == 'death'):

        cdc_forecast = pd.read_csv(f'{cdc_metric}_forecast.csv')
        cdc_forecast.rename(columns={cdc_forecast.iloc[:, 0].name : 'Date'}, inplace=True)
        cdc_forecast.set_index('Date', inplace=True)
        del cdc_forecast.index.name
        cdc_forecast.index = pd.to_datetime(cdc_forecast.index)

        # Filter CDC forecast to correct dates
        start_counter = 0
        for i in cdc_forecast.index:
            if i != pessimistic_14['Date'].iloc[0]:
                start_counter += 1
            else:
                break

        cdc_forecast = cdc_forecast.iloc[start_counter:start_counter+15]

        # Add bools for ensemble buttons
        show_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed
        hide_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed

        for i in range(0, len(cdc_forecast.columns)):
            show_ensemble_lines.append(True)
            hide_ensemble_lines.append(False)

    # Function to create plotly figure & push to Chart Studio
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14[df_column], mode='lines', name=legend_name,
                             line=dict(color='lightblue', width=4)))
    if (cdc_metric == 'case') or (cdc_metric == 'death'):
        for i in range(0, len(cdc_cases.columns)):
            fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=cdc_forecast.iloc[:, i], mode='lines', name=legend_name,
                                     line=dict(color='lightgray', width=1)))
    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)'
                        ))
    fig.update_yaxes(range=[0, max_metric],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    if (cdc_metric == 'case') or (cdc_metric == 'death'):
        fig.update_layout(autosize=False,
                          width=1500,
                          height=1000,
                          showlegend=False,
                          plot_bgcolor='white',
                          margin=dict(
                            autoexpand=False,
                            l=80,
                            r=80,
                            t=50
                            ),
                          title={'y':1},
                            updatemenus=[
                                    dict(
                                        type="buttons",
                                        bgcolor = 'rgb(205, 205, 205)',
                                        bordercolor = 'rgb(84, 84, 84)',
                                        font = dict(color='rgb(84, 84, 84)'),
                                        direction="right",
                                        active=-1,
                                        x=0.57,
                                        y=1.05,
                                        buttons=list([
                                            dict(label="Show Ensemble",
                                                 method="update",
                                                 args=[{"visible": show_ensemble_lines},
                                                       {"annotations": []}]),
                                            dict(label="Hide Ensemble",
                                                 method="update",
                                                 args=[{"visible": hide_ensemble_lines},
                                                       {"annotations": []}])
                                        ]),

                                    )
                                ])
    else:
            fig.update_layout(autosize=False,
                              width=1500,
                              height=1000,
                              showlegend=False,
                              plot_bgcolor='white',
                              margin=dict(
                                autoexpand=False,
                                l=80,
                                r=80,
                                t=50
                                ),
                              title={'y':1}
                              )
    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = f'hipam_forecast_{chart_studio_name}', auto_open=True)

    fig.write_html(f"./file_{chart_studio_name}.html")

# create_forecast_graphs(cdc_metric, df, df_column, legend_name, max_metric, chart_studio_name)
create_forecast_graphs('case', daily_new_cases_historical_graph, 'Reported_New_Cases', 'Cases', max_Reported_New_Cases, 'cases')
create_forecast_graphs('death', state_df_historical_graph, 'Fatal', 'Deaths', max_Fatal, 'death')
create_forecast_graphs('active_cases', active_historical, 'Active_Cases', 'Active Cases', max_Active_Cases, 'active_cases')

####################### ~ Cases ~ #######################

# Load CDC forecasts
cdc_cases = pd.read_csv('./cdc_forecasts/case_forecast.csv')
cdc_cases.rename(columns={cdc_cases.iloc[:, 0].name : 'Date'}, inplace=True)
cdc_cases.set_index('Date', inplace=True)
del cdc_cases.index.name
cdc_cases.index = pd.to_datetime(cdc_cases.index)
cdc_cases
# Filter CDC forecasts to correct dates
start_counter = 0
for i in cdc_cases.index:
    if i != pessimistic_14['Date'].iloc[0]:
        start_counter += 1
    else:
        break

cdc_cases = cdc_cases.iloc[start_counter:start_counter+15]
# cdc_cases = cdc_cases.iloc[0:15]
# Add bools for ensemble buttons
show_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed
hide_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed

for i in range(0, len(cdc_cases.columns)):
    show_ensemble_lines.append(True)
    hide_ensemble_lines.append(False)

# Function to create plotly figure & push to Chart Studio
def create_case_forecast_graph():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=daily_new_cases_historical_graph['Date'], y=daily_new_cases_historical_graph['Reported_New_Cases'], mode='lines', name='Cases',
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14['Reported_New_Cases'], mode='lines', name='Cases',
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14['Reported_New_Cases'], mode='lines', name='Cases',
                             line=dict(color='lightblue', width=4)))
    for i in range(0, len(cdc_cases.columns)):
        fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=cdc_cases.iloc[:, i], mode='lines', name='Cases',
                                 line=dict(color='lightgray', width=1)))
    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)'
                        ))
    fig.update_yaxes(range=[0, max_Reported_New_Cases],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                        updatemenus=[
                                dict(
                                    type="buttons",
                                    bgcolor = 'rgb(205, 205, 205)',
                                    bordercolor = 'rgb(84, 84, 84)',
                                    font = dict(color='rgb(84, 84, 84)', size=12),
                                    direction="right",
                                    active=-1,
                                    x=0.57,
                                    y=1.05,
                                    buttons=list([
                                        dict(label="Show Forecasts",
                                             method="update",
                                             args=[{"visible": show_ensemble_lines},
                                                   {"annotations": []}]),
                                        dict(label="Hide Forecasts",
                                             method="update",
                                             args=[{"visible": hide_ensemble_lines},
                                                   {"annotations": []}])
                                    ]),

                                )
                            ])

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'hipam_forecast_cases', auto_open=True)

    fig.write_html("./chart_studio/file_cases.html")

create_case_forecast_graph()

################## ~ Active Cases ~ ####################

# Function to create plotly figure & push to Chart Studio
def create_active_case_forecast_graph():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=daily_new_cases_historical_graph['Date'], y=active_historical['Active_Cases'], mode='lines', name='Active Cases',
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_active['Active_Cases'][-15:], mode='lines', name='Active Cases',
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_active['Active_Cases'][-15:], mode='lines', name='Active Cases',
                             line=dict(color='lightblue', width=4)))

    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)'
                        ))
    fig.update_yaxes(range=[0, max_Active_Cases],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1})


    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'hipam_forecast_active_cases', auto_open=True)

    fig.write_html("./cdc_forecasts/file_active_cases.html")

create_active_case_forecast_graph()

###################### ~ Deaths ~ ######################

# Load CDC forecasts
cdc_deaths = pd.read_csv('./cdc_forecasts/death_forecast.csv')
cdc_deaths.rename(columns={cdc_deaths.iloc[:, 0].name : 'Date'}, inplace=True)
cdc_deaths.set_index('Date', inplace=True)
del cdc_deaths.index.name
cdc_deaths.index = pd.to_datetime(cdc_deaths.index)

# Filter CDC forecasts to correct dates
start_counter = 0
for i in cdc_deaths.index:
    if i != pessimistic_14['Date'].iloc[0]:
        start_counter += 1
    else:
        break

# cdc_deaths = cdc_deaths.iloc[start_counter:start_counter+15]
cdc_deaths = cdc_deaths.iloc[0:15]

# Add bools for ensemble buttons
show_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed
hide_ensemble_lines = [True, True, True] # True's represent HIPAM lines and will always be displayed

for i in range(0, len(cdc_deaths.columns)):
    show_ensemble_lines.append(True)
    hide_ensemble_lines.append(False)

# Function to create plotly figure & push to Chart Studio
def create_death_forecast_graph():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=state_df_historical_graph['Date'], y=state_df_historical_graph['Fatal'], mode='lines', name='Deaths',
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14['Fatal'], mode='lines', name='Deaths',
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14['Fatal'], mode='lines', name='Deaths',
                             line=dict(color='lightblue', width=4)))
    for i in range(0, len(cdc_deaths.columns)):
        fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=cdc_deaths.iloc[:, i], mode='lines', name='Deaths',
                                 line=dict(color='lightgray', width=1)))

    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, max_Fatal],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                        updatemenus=[
                                dict(
                                    type="buttons",
                                    bgcolor = 'rgb(205, 205, 205)',
                                    bordercolor = 'rgb(84, 84, 84)',
                                    font = dict(color='rgb(84, 84, 84)', size=12),
                                    direction="right",
                                    active=-1,
                                    x=0.57,
                                    y=1.05,
                                    buttons=list([
                                        dict(label="Show Forecasts",
                                             method="update",
                                             args=[{"visible": show_ensemble_lines},
                                                   {"annotations": []}]),
                                        dict(label="Hide Forecasts",
                                             method="update",
                                             args=[{"visible": hide_ensemble_lines},
                                                   {"annotations": []}])
                                    ]),

                                )
                            ])

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'hipam_forecast_death', auto_open=True)

    fig.write_html("./cdc_forecasts/file_death.html")

create_death_forecast_graph()

################## ~ Hospitalizations ~ ##################
# Function to create plotly figure & push to Chart Studio
def create_hospitalization_forecast_graph():

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=state_df_historical_graph['Date'], y=state_df_historical_graph['Hospitalized'], mode='lines', name='Cases',
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14['Hospitalized'], mode='lines', name='Cases',
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14['Hospitalized'], mode='lines', name='Cases',
                             line=dict(color='lightblue', width=4)))

    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, max_hosp],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1}
                      )

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'hipam_forecast_hospitalizations', auto_open=True)

    fig.write_html("./cdc_forecasts/file_hospitalizations.html")

create_hospitalization_forecast_graph()

######################### ~ ICU ~ ########################
def create_ICU_forecast_graph():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=state_df_historical_graph['Date'], y=state_df_historical_graph['ICU'], mode='lines', name='Cases',
                             line=dict(color='lightgray', width=4)))
    fig.add_trace(go.Scatter(x=pessimistic_14['Date'], y=pessimistic_14['ICU'], mode='lines', name='Cases',
                             line=dict(color='lightcoral', width=4)))
    fig.add_trace(go.Scatter(x=expected_14['Date'], y=expected_14['ICU'], mode='lines', name='Cases',
                             line=dict(color='lightblue', width=4)))

    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, max_ICU],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1}
                      )
    #####

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'hipam_forecast_ICU', auto_open=True)

    fig.write_html("./cdc_forecasts/file_ICU.html")

create_ICU_forecast_graph()
#########################################################
############### ~ Create Final DataFrames ~ #############
#########################################################

def insert_active_cases(data_df, active_cases_df):
    """
    Inserts active cases into model output dataframes by matching dates between the model output df (data_df) & active cases df
    """
    active_cases_list = []
    for i in data_df['Date']:
        active_cases_list.append(float(active_cases_df['Active_Cases'][active_cases_df['Date'] == i].values[0]))

    data_df.insert(2, 'Active_Cases', active_cases_list)

# Create historical df
historic_df = state_df_historical[['date', 'positiveIncrease', 'hospitalizedCurrently', 'inIcuCurrently', 'death']]
historic_df['Scenario'] = 'Historical'
historic_df.columns = ['Date', 'Reported_New_Cases', 'Hospitalized', 'ICU', 'Fatal', 'Scenario']
historic_df = historic_df.reset_index(drop=True)
historic_df['Reported_New_Cases'] = daily_new_cases_historical[-30:]['Rolling_Average_Cases'].reset_index(drop=True)
insert_active_cases(historic_df, expected_active)

# Create pessimistic df
pessimistic_14_data = pessimistic_14[['Date', 'Reported_New_Cases', 'Hospitalized', 'ICU', 'Fatal', 'Scenario']].reset_index(drop=True)
insert_active_cases(pessimistic_14_data, pessimistic_active)

# Create expected df
expected_14_data = expected_14[['Date', 'Reported_New_Cases', 'Hospitalized', 'ICU', 'Fatal', 'Scenario']].reset_index(drop=True)
insert_active_cases(expected_14_data, expected_active)

#########################################################
########## ~ Create Oahu Graph (Plotly) ~ ###############
#########################################################

oahu_stats = {'7 Day Avg. Cases' : [93, 73, 68.7, 80, 49, 71, 81, 71, 84, 60, 72, 89, 83, 62, 88, 130, 86],
              'Test Positivity Rate' : [0.04, 0.032, 0.034, 0.023, 0.02, 0.027, 0.031, 0.027, 0.025, 0.021, 0.022, 0.031, 0.028, 0.029, 0.042, 0.040, 0.031]
              }
oahu_df = pd.DataFrame(oahu_stats)
oahu_df.index = ['9/30', '10/7', '10/14', '10/21', '10/28', '11/04', '11/11', '11/18', '11/25', '12/02', '12/09', '12/16', '12/23', '12/30', '1/06', '1/13', '1/20']

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=oahu_df.index, y=oahu_df['7 Day Avg. Cases'], name = 'Cases', marker_color = ['orange', 'orange', 'orange', 'orange', 'gold', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']),
              secondary_y=False)
fig.add_trace(go.Scatter(x=oahu_df.index, y=oahu_df['Test Positivity Rate'], mode='markers', marker=dict(size=50, color=['orange', 'orange', 'orange', 'gold', 'gold', 'orange', 'orange', 'orange', 'orange', 'gold', 'gold', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']), marker_symbol='cross-dot', name='Test Positivity'),
              secondary_y=True)

fig.update_traces(marker_line_color='rgb(84,84,84)', marker_line_width=3, opacity=0.8)

fig.update_xaxes(showline=True,
                 showgrid=False,
                 showticklabels=True,
                 linecolor='rgb(140, 140, 140)',
                 linewidth=2,
                 ticks='outside',
                 tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgb(140, 140, 140)',
                    ))
fig.update_yaxes(range = [0, 140],
                 title_text='7 Day Avg. Cases',
                 title_font = {"size": 20, "color": 'rgb(140, 140, 140)'},
                 secondary_y=False,
                 showline=True,
                 showgrid=False,
                 showticklabels=True,
                 linecolor='rgb(140, 140, 140)',
                 linewidth=2,
                 ticks='outside',
                 tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgb(140, 140, 140)',
                    ))
fig.update_yaxes(range = [0, 0.0625],
                 title_text='7 Day Avg. Test Positivity',
                 title_font = {"size": 20, "color": 'rgb(140, 140, 140)'},
                 secondary_y=True,
                 showline=True,
                 showgrid=False,
                 showticklabels=True,
                 linecolor='rgb(140, 140, 140)',
                 linewidth=2,
                 tickformat='.1%',
                 ticks='outside',
                 tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgb(140, 140, 140)',
                    ))
fig.update_layout(autosize=False,
                  width=1500,
                  height=1000,
                  showlegend=False,
                  plot_bgcolor='white',
                  margin=dict(
                    autoexpand=False,
                    l=80,
                    r=80,
                    t=50
                    ),
                  title={'y':1},
                  shapes=[
                        dict(
                            type="rect",
                            # x-reference is assigned to the x-values
                            xref="paper",
                            # y-reference is assigned to the plot paper [0,1]
                            yref="paper",
                            x0=0,
                            y0=0,
                            x1=0.29,
                            y1=1,
                            fillcolor="FireBrick",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ),
                        dict(
                            type="rect",
                            # x-reference is assigned to the x-values
                            xref="paper",
                            # y-reference is assigned to the plot paper [0,1]
                            yref="paper",
                            x0=0.29,
                            y0=0,
                            x1=0.92,
                            y1=1,
                            fillcolor="orange",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ),
                        {
                            'type': 'line',
                            'xref': 'paper',
                            'x0': 0.94,
                            'y0': 100, # use absolute value or variable here
                            'x1': 0,
                            'y1': 100, # ditto
                            'line': {
                                'color': 'FireBrick',
                                'width': 4,
                                'dash': 'dash',
                            },
                        },
                        {
                            'type': 'line',
                            'xref': 'paper',
                            'x0': 0.94,
                            'y0': 50, # use absolute value or variable here
                            'x1': 0,
                            'y1': 50, # ditto
                            'line': {
                                'color': 'Orange',
                                'width': 4,
                                'dash': 'dash',
                            },
                        },
                        {
                            'type': 'line',
                            'xref': 'paper',
                            'x0': 0.94,
                            'y0': 25, # use absolute value or variable here
                            'x1': 0,
                            'y1': 25, # ditto
                            'line': {
                                'color': 'Gold',
                                'width': 4,
                                'dash': 'dash',
                            },
                        },

                    ],
                  )

fig.show()
username = 'ldantzin' # your username
api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
py.plot(fig, filename = 'current_oahu', auto_open=True)

fig.write_html("./chart_studio/file_current_oahu.html")

#########################################################
################ ~ Current Situation ~ ##################
#########################################################

# df cleaning and prep
hdc_cases = get_cases()
hdc_positivity = get_test_positivity()
# hdc_rt = get_rt()

hdc_positivity = hdc_positivity[(hdc_positivity['Region'] == 'Hawaii County') | (hdc_positivity['Region'] == 'Honolulu County') | (hdc_positivity['Region'] == 'Kauai County') | (hdc_positivity['Region'] == 'Maui County') | (hdc_positivity['Region'] == 'State')]
hdc_cases = hdc_cases[(hdc_cases['Region'] == 'Hawaii County') | (hdc_cases['Region'] == 'Honolulu County') | (hdc_cases['Region'] == 'Kauai County') | (hdc_cases['Region'] == 'Maui County') | (hdc_cases['Region'] == 'State')]

# hdc_rt['date'] = pd.to_datetime(hdc_rt['date'])
hdc_positivity['Date'] = pd.to_datetime(hdc_positivity['Date'][630:])
hdc_cases['Date'] = pd.to_datetime(hdc_cases['Date'])

def get_start_date_index(df_date_series):
    start_date_index = 0
    for e, i in enumerate(df_date_series):
        if i == state_df_historical_graph['Date'].iloc[0]:
            start_date_index = e
            break
    return start_date_index

# rt_start_date_index = get_start_date_index(hdc_rt['date'])
positivity_start_date_index = get_start_date_index(hdc_positivity['Date'])
cases_start_date_index = get_start_date_index(hdc_cases['Date'])

hdc_cases = hdc_cases[cases_start_date_index:]
hdc_positivity = hdc_positivity[positivity_start_date_index:]
# hdc_rt = hdc_rt[rt_start_date_index:]


def create_case_situation_graph():
    hdc_cases['NewCases_Rate'] = hdc_cases['NewCases_Rate'].astype(float)
    hdc_cases['Region'] = ['All County' if i == 'State' else i for i in hdc_cases['Region']]

    low = 1
    medium = 10
    high = 25
    critical = 38

    def get_threshold_data(df, metric, region, threshold, ceiling):
        threshold_list = []
        for e, i in enumerate(df[f'{metric}'][df['Region'] == f'{region} County']):
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2) & (i >= threshold) & (i <= ceiling):
                threshold_list.append(i)
                break
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2):
                threshold_list.append(np.nan)
                break
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e+1] > threshold:
                threshold_list.append(i)
                continue
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e] > threshold:
                threshold_list.append(i)
                continue
            if (e != 0) & (df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e-1] > threshold) & (i <= threshold):
                threshold_list.append(i)
            else:
                threshold_list.append(np.nan)
        return threshold_list

    hawaii_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', low, medium)
    hawaii_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', medium, high)
    hawaii_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Hawaii', high, critical)

    Honolulu_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', low, medium)
    Honolulu_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', medium, high)
    Honolulu_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Honolulu', high, critical)

    Kauai_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', low, medium)
    Kauai_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', medium, high)
    Kauai_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Kauai', high, critical)

    Maui_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', low, medium)
    Maui_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', medium, high)
    Maui_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'Maui', high, critical)

    all_medium_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', low, medium)
    all_high_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', medium, high)
    all_critical_cases = get_threshold_data(hdc_cases, 'NewCases_Rate', 'All', high, critical)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'], mode='lines', name='Hawaii County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'] <= low), mode='lines', name='Hawaii County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_medium_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_high_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Hawaii County'], y=hawaii_critical_cases, mode='lines', name='Hawaii County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'], mode='lines', name='Honolulu County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'] <= low), mode='lines', name='Honolulu County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_medium_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_high_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Honolulu County'], y=Honolulu_critical_cases, mode='lines', name='Honolulu County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'], mode='lines', name='Kauai County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'] <= low), mode='lines', name='Kauai County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_medium_cases, mode='lines', name='Kauai County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_high_cases, mode='lines', name='Kauai County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Kauai County'], y=Kauai_critical_cases, mode='lines', name='Kauai County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'], mode='lines', name='Maui County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'] <= low), mode='lines', name='Maui County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_medium_cases, mode='lines', name='Maui County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_high_cases, mode='lines', name='Maui County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'Maui County'], y=Maui_critical_cases, mode='lines', name='Maui County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'], mode='lines', name='All County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'].where(hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'] <= low), mode='lines', name='All County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_medium_cases, mode='lines', name='All County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_high_cases, mode='lines', name='All County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_cases['Date'][hdc_cases['Region'] == 'All County'], y=all_critical_cases, mode='lines', name='All County',
                             line=dict(color='firebrick', width=4)))


    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, hdc_cases['NewCases_Rate'].max()+1],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    annotations = []
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Hawaii County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Hawaii'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Honolulu County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Honolulu'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Kauai County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Kauai'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'Maui County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Maui'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_cases['NewCases_Rate'][hdc_cases['Region'] == 'All County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='All'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                      annotations=annotations,
                      shapes=[
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 10, # use absolute value or variable here
                                'x1': 0,
                                'y1': 10, # ditto
                                'line': {
                                    'color': '#f38181',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 1, # use absolute value or variable here
                                'x1': 0,
                                'y1': 1, # ditto
                                'line': {
                                    'color': '#fce38a',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            }
                        ]
                      )

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'all_counties_cases', auto_open=True)

    fig.write_html("./chart_studio/file_all_counties_cases.html")

def create_positivity_situation_graph():
    hdc_positivity['% Pos'] = hdc_positivity['% Pos'].astype(float)
    hdc_positivity['Region'] = ['All County' if i == 'State' else i for i in hdc_positivity['Region']]

    low = 0.03
    medium = 0.1
    high = 0.2
    critical = 0.31

    def get_threshold_data(df, metric, region, threshold, ceiling):
        threshold_list = []
        for e, i in enumerate(df[f'{metric}'][df['Region'] == f'{region} County']):
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2) & (i >= threshold) & (i <= ceiling):
                threshold_list.append(i)
            if (e > len((df[f'{metric}'][df['Region'] == f'{region} County']))-2):
                threshold_list.append(np.nan)
                break
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e+1] > threshold:
                threshold_list.append(i)
                continue
            if df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e] > threshold:
                threshold_list.append(i)
                continue
            if (e != 0) & (df[f'{metric}'][df['Region'] == f'{region} County'].iloc[e-1] > threshold) & (i <= threshold):
                threshold_list.append(i)
            else:
                threshold_list.append(np.nan)
        return threshold_list

    hawaii_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', low, medium)
    hawaii_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', medium, high)
    hawaii_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Hawaii', high, critical)

    Honolulu_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', low, medium)
    Honolulu_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', medium, high)
    Honolulu_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Honolulu', high, critical)

    Kauai_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', low, medium)
    Kauai_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', medium, high)
    Kauai_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Kauai', high, critical)

    Maui_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', low, medium)
    Maui_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', medium, high)
    Maui_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'Maui', high, critical)

    all_medium_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', low, medium)
    all_high_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', medium, high)
    all_critical_cases = get_threshold_data(hdc_positivity, '% Pos', 'All', high, critical)


    hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].where((hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] > low) & (hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] <= medium))


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'], mode='lines', name='Hawaii County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'] <= low), mode='lines', name='Hawaii County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_medium_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_high_cases, mode='lines', name='Hawaii County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Hawaii County'], y=hawaii_critical_cases, mode='lines', name='Hawaii County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'], mode='lines', name='Honolulu County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'] <= low), mode='lines', name='Honolulu County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_medium_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_high_cases, mode='lines', name='Honolulu County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Honolulu County'], y=Honolulu_critical_cases, mode='lines', name='Honolulu County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'], mode='lines', name='Kauai County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'] <= low), mode='lines', name='Kauai County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_medium_cases, mode='lines', name='Kauai County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_high_cases, mode='lines', name='Kauai County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Kauai County'], y=Kauai_critical_cases, mode='lines', name='Kauai County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'], mode='lines', name='Maui County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'] <= low), mode='lines', name='Maui County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_medium_cases, mode='lines', name='Maui County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_high_cases, mode='lines', name='Maui County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'Maui County'], y=Maui_critical_cases, mode='lines', name='Maui County',
                             line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'], mode='lines', name='All County',
                             line=dict(color='white', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'].where(hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'] <= low), mode='lines', name='All County',
                             line=dict(color='lightgreen', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_medium_cases, mode='lines', name='All County',
                             line=dict(color='#fce38a', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_high_cases, mode='lines', name='All County',
                             line=dict(color='#f38181', width=4)))
    fig.add_trace(go.Scatter(x=hdc_positivity['Date'][hdc_positivity['Region'] == 'All County'], y=all_critical_cases, mode='lines', name='All County',
                             line=dict(color='firebrick', width=4)))


    fig.update_xaxes(showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))
    fig.update_yaxes(range=[0, hdc_positivity['% Pos'].max()+.01],
                     showline=True,
                     showgrid=False,
                     showticklabels=True,
                     linecolor='rgb(140, 140, 140)',
                     linewidth=2,
                     tickformat='.1%',
                     ticks='outside',
                     tickfont=dict(
                        family='Arial',
                        size=16,
                        color='rgb(140, 140, 140)',
                        ))

    annotations = []
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Hawaii County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Hawaii'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Honolulu County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Honolulu'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Kauai County'].iloc[-1],
                                  xanchor='left', yanchor='middle',
                                  text='Kauai'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'Maui County'].iloc[-2],
                                  xanchor='left', yanchor='middle',
                                  text='Maui'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    annotations.append(dict(xref='paper', x=1.01, y=hdc_positivity['% Pos'][hdc_positivity['Region'] == 'All County'].iloc[-2],
                                  xanchor='left', yanchor='middle',
                                  text='All'.format(color='rgb(140, 140, 140)'),
                                  font=dict(family='Arial',
                                            size=16,
                                            color='rgb(140, 140, 140)'),
                                  showarrow=False))
    fig.update_layout(autosize=False,
                      width=1500,
                      height=1000,
                      showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(
                        autoexpand=False,
                        l=80,
                        r=80,
                        t=50
                        ),
                      title={'y':1},
                      annotations=annotations,
                      shapes=[
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': .1, # use absolute value or variable here
                                'x1': 0,
                                'y1': .1, # ditto
                                'line': {
                                    'color': '#f38181',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            },
                            {
                                'type': 'line',
                                'xref': 'paper',
                                'x0': 1,
                                'y0': 0.03, # use absolute value or variable here
                                'x1': 0,
                                'y1': 0.03, # ditto
                                'line': {
                                    'color': '#fce38a',
                                    'width': 4,
                                    'dash': 'dash',
                                },
                            }
                        ]
                      )

    username = 'ldantzin' # your username
    api_key = 'VHoCUagvK4RmEvDTqG3F' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename = 'all_counties_positivity', auto_open=True)

    fig.write_html("./chart_studio/file_all_counties_positivity.html")

create_case_situation_graph()
create_positivity_situation_graph()

#########################################################
#################### ~ Create CSV ~ #####################
#########################################################

# Create CSV
data_df = pd.concat([pessimistic_14_data, expected_14_data, historic_df]) # Create final df for CSV
data_df.to_csv(f'./Model_Outputs/model_output_{todays_date_f_string}.csv')

#########################################################
###### ~ Create Historical Parameters Dictionary ~ ######
#########################################################

# Create historical record of model inputs
todays_input_dict = {'Todays_Date' : todays_date,
              'Initialization_Start' : initialization_start,
              'Rt_Initialization' : rt_initialization,
              'Rt_Estimate_Start' : rt_estimate_start,
              'Rt_Estimate_Worst' : rt_estimate_pessimistic,
              'Rt_Estimate_Expected' : rt_estimate_expected,
              'Incubation' : incubation,
              'Infectious_Duration' : infectious_duration,
              'Delay' : delay,
              'Hosp_Stay' : hosp_stay,
              'ICU_stay' : ICU_stay,
              'Hospitalization_Rate' : hospitalization_rate,
              'Hospitalization_ICU_Rate' : hospitalization_ICU_rate,
              'ICU_Hosp_Rate' : ICU_hosp_rate}

nest_todays_input_dict = {todays_date : todays_input_dict}

# Initialize nested dict - ONLY USED TO CREATE INITIAL DICTIONARY
# with open('historical_input_dict.json', 'w') as fp:
#     json.dump(historical_input_dict, fp)

with open('historical_input_dict.json') as f:
  historical_input_dict_data = json.load(f)

# Add today's inputs to dict
historical_input_dict_data.update(nest_todays_input_dict)

with open('historical_input_dict.json', 'w') as fp:
    json.dump(historical_input_dict_data, fp)


with open('historical_input_dict.json') as f:
  input_dict = json.load(f)

rt_e = ''
