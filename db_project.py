# Correct import of required libraries
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

# Load the dataset
file_path = '/Users/idohaber/Desktop/Git-Projects/08_climate_db/climate_hist.csv'  
df = pd.read_csv(file_path)

print(df)

# Convert the DATE column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Set the DATE column as the index of the dataframe
df.set_index('DATE', inplace=True)

# Fill missing values if necessary
df.fillna(method='ffill', inplace=True)

# Calculate rolling mean
window_size = 365  # Use a window of 365 days for a rolling average
df['TMAX_roll_avg'] = df['TMAX'].rolling(window=window_size).mean()
df['PRCP_roll_avg'] = df['PRCP'].rolling(window=window_size).mean()
df['SNOW_roll_avg'] = df['SNOW'].rolling(window=window_size).mean()
df['TMIN_roll_avg'] = df['TMIN'].rolling(window=window_size).mean()

# Convert dates to ordinal numbers for linear regression
df['date_ordinal'] = pd.to_datetime(df.index).map(datetime.toordinal)

# Updated Linear regression function to return slope and intercept
def add_trendline(x, y, ax, color):
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        print("Not enough data points to plot a trendline.")
        return None, None  # Return None if not enough data
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trendline = intercept + slope * x
    ax.plot(df.index[mask], trendline, color=color, linestyle='--', linewidth=2)
    return slope, intercept

# Plotting 
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Temperature Max Trend
ax1 = plt.subplot(4, 1, 1)
plt.plot(df.index, df['TMAX_roll_avg'], label='TMAX 1yr Rolling Average', color='r')
slope_tmax, intercept_tmax = add_trendline(df['date_ordinal'], df['TMAX_roll_avg'], ax1, 'darkred')
equation_tmax = f"TMAX: y = {slope_tmax:.5f}x + {intercept_tmax:.2f}"
plt.title(f'Temperature Max Trend Over Time\n{equation_tmax}')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()

# Temperature Min Trend
ax2 = plt.subplot(4, 1, 2)
plt.plot(df.index, df['TMIN_roll_avg'], label='TMIN 1yr Rolling Average', color='b')
slope_tmin, intercept_tmin = add_trendline(df['date_ordinal'], df['TMIN_roll_avg'], ax2, 'darkred')
equation_tmin = f"TMIN: y = {slope_tmin:.5f}x + {intercept_tmin:.2f}"
plt.title(f'Temperature Min Trend Over Time\n{equation_tmin}')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()

# Precipitation Trend
ax3 = plt.subplot(4, 1, 3)
plt.plot(df.index, df['PRCP_roll_avg'], label='PRCP 1yr Rolling Average', color='brown')
slope_prcp, intercept_prcp = add_trendline(df['date_ordinal'], df['PRCP_roll_avg'], ax3, 'darkred')
equation_prcp = f"PRCP: y = {slope_prcp:.5f}x + {intercept_prcp:.2f}"
plt.title(f'Precipitation Trend Over Time\n{equation_prcp}')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.legend()

# Snowfall Trend
# Snowfall Trend
ax4 = plt.subplot(4, 1, 4)
plt.plot(df.index, df['SNOW_roll_avg'], label='SNOW 1yr Rolling Average', color='green')
slope_snow, intercept_snow = add_trendline(df['date_ordinal'], df['SNOW_roll_avg'], ax4, 'darkgreen')  # Fixed variable names
equation_snow = f"SNOW: y = {slope_snow:.5f}x + {intercept_snow:.2f}"  # Fixed equation using correct variables
plt.title(f'Snowfall Trend Over Time\n{equation_snow}')  # Include equation in the title
plt.xlabel('Year')
plt.ylabel('Snowfall (mm)')
plt.legend()


plt.tight_layout()
plt.show()
