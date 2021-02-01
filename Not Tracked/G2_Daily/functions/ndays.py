from datetime import datetime, timedelta

def ndays(date1, date2):
    date_format = "%m/%d/%Y"
    date1 = datetime.strptime(date1, date_format)
    date2 = datetime.strptime(date2, date_format)
    delta = date2 - date1
    return delta.days
