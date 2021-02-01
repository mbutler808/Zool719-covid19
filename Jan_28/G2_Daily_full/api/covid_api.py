import requests
import pandas as pd
import numpy as np
import datetime

def get_state_data(state, format_date_str):

    response = requests.get('https://api.covidtracking.com/v1/states/daily.csv')
    states_list = response.content.decode("utf-8").splitlines()
    states_list_columns = states_list[0].split(',')

    def create_columns(states_list):
        new_states_list = []

        for row in range(1, len(states_list)):
            new_row = states_list[row].split(',')
            new_states_list.append(new_row)

        return new_states_list

    states_list_split = create_columns(states_list)

    states_df = pd.DataFrame(states_list_split, columns = states_list_columns)

    # Select Hawaii
    state_df = states_df[states_df['state'] == f'{state}']
    # Sort from oldest to newest
    state_df = state_df.iloc[::-1]
    # Reset index
    state_df = state_df.reset_index(drop = True)


    # Select relevant Columns
    state_df = state_df[['date',
              'positive',
              'positiveIncrease',
              'negative',
              'negativeIncrease',
              'hospitalizedCurrently',
              'hospitalizedCumulative',
              'inIcuCurrently',
              'onVentilatorCurrently',
              'recovered',
              'death',
              'deathIncrease'
              ]]

    # Change to datetime object
    state_df['date'] = pd.to_datetime(state_df['date'])

    return state_df
