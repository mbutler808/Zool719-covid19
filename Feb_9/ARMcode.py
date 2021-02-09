import pandas as pd

# lambda function that converts a string into datetime
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')

# reads the csv into
df = pd.read_csv(r"C:\TBLAMEY\UH\SP2021\Covid-19\Reported_hono.csv", parse_dates=['Date'], date_parser=dateparse)
ts = df['Reported']

# sets the date column as the index
df = df.set_index('Date')
df.plot()




import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Ho: It is non-stationary vs H1: It is stationary
result = adfuller(ts.dropna())
print('p-value: %f' % result[1])

# Original Series plot
fig, axes = plt.subplots(2, 2, sharex=True)
axes[0, 0].plot(ts); axes[0, 0].set_title('Original Series')
plot_acf(ts, ax=axes[0, 1])

result = adfuller(ts.diff().dropna())
print('p-value: %f' % result[1])

# Differencing stabilizes the mean (Reducing Trend)
# 1st Order Differencing plot
axes[1, 0].plot(ts.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(ts.diff().dropna(), ax=axes[1, 1])



import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Original Series plot
fig, axes = plt.subplots(2, 1, sharex=True)
plot_acf(ts.diff().dropna(), ax=axes[0])
plot_pacf(ts.diff().dropna(), ax=axes[1])



import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# 1,1,1 ARIMA Model
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()



import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

ts = df['Reported']

# 1,1,1 ARIMA Model
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit(disp=0)

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()




from statsmodels.tsa.stattools import acf
# Create Training and Test
s = 270; d = 307
train = ts[s:d]; test = ts[d:]
# Build Model
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit(disp=-1)
# Forecast
fc, se, conf = fitted.forecast(len(ts)-d, alpha=0.01)  # 99% conf
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
# Plot
#plt.figure(figsize=(10,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()




import pmdarima as pm
model = pm.auto_arima(ts, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model.summary())



from statsmodels.tsa.stattools import acf
# Create Training and Test
s = 270; d = 307
train = ts[s:d]; test = ts[d:]
# Build Model
model = ARIMA(ts, order=(2, 1, 3))
fitted = model.fit(disp=-1)
# Forecast
fc, se, conf = fitted.forecast(len(ts)-d, alpha=0.01)  # 99% conf
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
# Plot
#plt.figure(figsize=(10,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()




# Forecast
n_periods = 14
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(ts), len(ts)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.figure(figsize=(10,5), dpi=100)

plt.plot(ts)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("Final Forecast")
plt.show()
