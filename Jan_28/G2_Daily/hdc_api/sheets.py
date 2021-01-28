import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os

def get_test_positivity():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('Test Positivity Rate')

    date_reported = worksheet.col_values(1)
    county = worksheet.col_values(2)
    positivity_rate = worksheet.col_values(3)

    df = pd.DataFrame([date_reported, county, positivity_rate]).T
    df.columns = df.iloc[0]
    df = df[1:]

    df['% Pos'] = [float(i.strip('%'))*.01 for i in df['% Pos']]

    return  df



def get_rt():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('Rate of Transmission')

    date_reported = worksheet.col_values(1)
    rt = worksheet.col_values(4)
    rt_lower = worksheet.col_values(6)
    rt_upper = worksheet.col_values(7)

    df = pd.DataFrame([date_reported, rt, rt_lower, rt_upper]).T
    df.columns = df.iloc[0]
    df = df[1:]

    return  df

def get_cases():

    path = os.getcwd()
    path_sheets_api = path + '/hdc_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1sd-L317Je9ZhiQh3_uH9jTkl3ckc_o3sgrVauShcwCk/edit#gid=0')

    worksheet = sheet.worksheet('COVID Data')

    date_reported = worksheet.col_values(2)
    county = worksheet.col_values(1)
    cases_100k = worksheet.col_values(7)
    cases_new = worksheet.col_values(4)

    df = pd.DataFrame([date_reported, county, cases_100k, cases_new]).T
    df.columns = df.iloc[0]
    df = df[1:]

    return  df
