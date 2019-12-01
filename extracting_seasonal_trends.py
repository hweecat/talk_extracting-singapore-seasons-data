'''
Extracting Seasonal Insights from Air Temperature and Rainfall Data from Data.gov.sg v1
Developer: Ong Chin Hwee (@hweecat)
Language: Python 3.7.3

This data exploration uses meteorological data scraped from Data.gov.sg APIs.
Objective: To explore trend and seasonality of air temperature and rainfall data
from Singapore Weather Station
Timeframe for analysis: 2 December 2016 to 30 November 2019 (~3 years)

Air Temperature: measured in degree Celsius
Rainfall: measured in millimetres (mm)
Timestamp: GMT +8 Singapore Time
'''
# %%
import numpy as np
import pandas as pd
import datetime
import pickle
import glob

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# timezone
import pytz

# %% [markdown]

# Obtain list of air temperature data CSV files from Dec 2016 to Nov 2019

data_filelist_airtemp = glob.glob("air-temperature/*.csv")

data_filelist_airtemp

# %% [markdown]

# Obtain list of rainfall data CSV files from Dec 2016 to Nov 2019

data_filelist_rain = glob.glob("rainfall/*.csv")

data_filelist_rain

# %% [markdown]
# Concatenate data tables from air temperature data CSV files

df_airtemp = pd.concat([pd.read_csv(file, index_col=0) for file in data_filelist_airtemp])

df_airtemp.info()

#%% [markdown]
# Convert timestamp to datetime[ns] format with UTC offset

df_airtemp['timestamp'] = pd.to_datetime(df_airtemp['timestamp'], infer_datetime_format=True, utc=True)

df_airtemp.info()

# %% [markdown]
# Drop redundant columns from air temperature dataframe + set Timestamp as Index

df_airtemp_data = df_airtemp.drop(columns=['index', 'station_id']).set_index('timestamp')

df_airtemp_data

# %% [markdown]
# Convert timestamp to UTC +8 (Singapore) offset

df_airtemp_data = df_airtemp_data.tz_convert('Asia/Singapore')

df_airtemp_data

#%% [markdown]
# Rename value column to 'air temperature'

df_airtemp_data.rename(columns={'value':'air temperature'}, inplace=True)

df_airtemp_data

# %% [markdown]
# Concatenate data tables from rainfall data CSV files

df_rain = pd.concat([pd.read_csv(file, index_col=0) for file in data_filelist_rain])

df_rain.info()

#%%
# Convert timestamp to datetime[ns] format with UTC offset

df_rain['timestamp'] = pd.to_datetime(df_rain['timestamp'], infer_datetime_format=True, utc=True)

df_rain.info()

# %%
# Drop redundant columns from rainfall dataframe + set Timestamp as Index

df_rain_data = df_rain.drop(columns=['index', 'station_id']).set_index('timestamp')

df_rain_data

# %%
# Convert timestamp to UTC +8 (Singapore) offset

df_rain_data.index = df_rain_data.index.tz_convert('Asia/Singapore')

df_rain_data.index

#%%
# Rename value column to 'rainfall'

df_rain_data.rename(columns={'value':'rainfall'}, inplace=True)

df_rain_data

#%%
# Concatenate air temperature and rainfall data along column axis
# Since time period between data points not uniform across timeframe,
# resampling is needed to space out data points evenly.

df_weather_data_min = pd.concat([
    df_airtemp_data.resample('5T').mean().ffill(),
    df_rain_data.resample('5T').fillna(method='ffill')],
    axis=1)

df_weather_data_min

#%% [markdown]
# Check for number of missing values

df_weather_data_min.isnull().sum()

#%% [markdown]
# Downsample data points from 5-minute frequency to hourly frequency,
# taking mean value within each time period as data to resample

df_weather_data_hour = df_weather_data_min.resample('H').mean().ffill()

df_weather_data_hour

#%%

plt.figure(figsize=(20,10))
df_weather_data_hour['air temperature'].plot(marker='o', linestyle='none')
plt.xlabel('Hour')
plt.ylabel('Air Temperature in degree Celsius')
plt.title('Mean hourly sampling of air temperature at Changi Weather Station')

#%% [markdown]
# Downsample data points from minute-by-minute to weekly frequency,
# taking mean value within each time period as data to resample

df_weather_data_week = df_weather_data_min.resample('W').mean()

df_weather_data_week

#%%

plt.figure(figsize=(20,10))
df_weather_data_week['air temperature'].plot(marker='.', linestyle='-')
plt.xlabel('Week')
plt.ylabel('Air Temperature in degree Celsius')
plt.title('Mean weekly sampling of air temperature at Changi Weather Station')

#%%

plt.figure(figsize=(20,10))
df_weather_data_week['rainfall'].plot(marker='.', linestyle='-')
plt.xlabel('Week')
plt.ylabel('Rainfall in mm')
plt.title('Mean weekly sampling of rainfall at Changi Weather Station')

#%% [markdown]
# Downsample data points from minute-by-minute to monthly frequency,
# taking mean value within each time period as data to resample

df_weather_data_month = df_weather_data_min.resample('M').mean()

df_weather_data_month

#%%

plt.figure(figsize=(20,10))
df_weather_data_week['air temperature'].plot(marker='.', linestyle=':', label='Weekly')
df_weather_data_month['air temperature'].plot(marker='o', linestyle='-', label='Monthly')
plt.xlabel('Month')
plt.ylabel('Air Temperature in degree Celsius')
plt.title('Mean sampling of air temperature at Changi Weather Station')
plt.legend()

#%%

plt.figure(figsize=(20,10))
df_weather_data_week['rainfall'].plot(marker='.', linestyle=':', label='Weekly')
df_weather_data_month['rainfall'].plot(marker='o', linestyle='-', label='Monthly')
plt.xlabel('Month')
plt.ylabel('Rainfall in mm')
plt.title('Mean sampling of rainfall at Changi Weather Station')
plt.legend()

#%% [markdown]
# Prepare data for trend and seasonality analysis

df_weather_data_min['Year'] = df_weather_data_min.index.year

df_weather_data_min['Month'] = df_weather_data_min.index.month

df_weather_data_min.sample(5, random_state=0)

#%% [markdown]
## Extracting Seasonality from Weather Dta
# Box plot of month-wise distribution of weather data

fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['air temperature', 'rainfall'], axes):
    sns.boxplot(data=df_weather_data_min, x='Month', y=name, ax=ax)
    ax.set_title(name)
# Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
# Set Title
axes[0].set_title('Year-wise Box Plot\n(Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(Seasonality)', fontsize=18)
plt.show()

#%% [markdown]
# Box plot of year-wise distribution of weather data

fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['air temperature', 'rainfall'], axes):
    sns.boxplot(data=df_weather_data_min[df_weather_data_min['Year'] > 2016], x='Year', y=name, ax=ax)
    ax.set_title(name)
# Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
# Set Title
axes[0].set_title('Year-wise Box Plot\n(Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(Seasonality)', fontsize=18)
plt.show()

# %% [markdown]
## Associate rainfall occurrence with temperature

df_weather_data_min['rain?'] = df_weather_data_min['rainfall'].apply(lambda x: 'Rain' if x > 0 else 'No Rain')

df_weather_data_min['rain?']

# %%


sns.boxplot(data=df_weather_data_min, x='rain?', y='air temperature')


# %%

df_weather_rainy_days = df_weather_data_hour['rainfall'].resample('D').apply(
    lambda x: 1 if x.any() > 0 else 0 if x.all() == 0 else None)

df_weather_rainy_days

# %%

df_weather_monthly_rainy_days = df_weather_rainy_days.resample('M').sum()

df_weather_monthly_rainy_days

# %%

plt.figure(figsize=(20,10))
df_weather_monthly_rainy_days.plot()
plt.title('Monthly number of rain days from Dec 2016 to Nov 2019', fontsize=36)
plt.xlabel('Month', fontsize=24)
plt.ylabel('Days', fontsize=24)
plt.xticks(fontsize=16)

# %% [markdown]
# Determining Weather Trends by Rolling Windows

df_weather_data_daily = df_weather_data_min[['air temperature','rainfall']].resample('D').mean()

df_weather_data_daily

# %%

df_weather_data_7days = df_weather_data_daily[['air temperature','rainfall']].rolling(window=7,center=True).mean()

df_weather_data_7days

# %%

df_weather_data_30days = df_weather_data_daily[['air temperature','rainfall']].rolling(window=30,center=True).mean()

df_weather_data_30days

# %%

plt.figure(figsize=(20,10))
df_weather_data_7days['air temperature'].plot(marker='.', linestyle=':', label='7-day rolling mean')
df_weather_data_30days['air temperature'].plot(marker='o', linestyle='-', label='30-day rolling mean')
plt.xlabel('timestamp')
plt.ylabel('Air temperature in degree Celsius')
plt.title('Rolling window of air temperature at Changi Weather Station')
plt.legend()

# %%

plt.figure(figsize=(20,10))
df_weather_data_7days['rainfall'].plot(marker='.', linestyle=':', label='7-day rolling mean')
df_weather_data_30days['rainfall'].plot(marker='o', linestyle='-', label='30-day rolling mean')
plt.xlabel('timestamp')
plt.ylabel('Rainfall in mm')
plt.title('Rolling window of rainfall at Changi Weather Station')
plt.legend()


# %%
