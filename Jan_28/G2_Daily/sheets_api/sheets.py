import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os

def get_daily_positive_cases():

    path = os.getcwd()
    path_sheets_api = path + '/sheets_api/'

    scope = ["https://spreadsheets.google.com/feeds",
             'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name((path_sheets_api + "creds.json"), scope)

    client = gspread.authorize(creds)

    sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1Q4y14VoTvawco3cI4Qm-zP9FJSXa79CVlUSTybubANY/edit#gid=2124927884')

    worksheet = sheet.worksheet('dailyconfirm')

    Date_Reported = worksheet.col_values(3)
    Positive_Cases = worksheet.col_values(4)

    Positive_Cases_Df = pd.DataFrame(Positive_Cases, Date_Reported)[1:]
    Positive_Cases_Df = Positive_Cases_Df.reset_index()
    Positive_Cases_Df.columns = ['Date', 'Cases']
    Positive_Cases_Df['Date'] = pd.to_datetime(Positive_Cases_Df['Date'])
    Positive_Cases_Df['Date'] = [i.replace(hour=0, minute=0, second=0) for i in Positive_Cases_Df['Date']]
    Positive_Cases_Df['Cases'] = Positive_Cases_Df['Cases'].astype(int)

    return Positive_Cases_Df
