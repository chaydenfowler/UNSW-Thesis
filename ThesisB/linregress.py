import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

matplotlib.rcParams.update({'font.size': 12,
                           'figure.figsize': [4.5, 3],
                           'lines.markeredgewidth': 0,
                           'lines.markersize': 2
                           })

'''
np.random.seed(12345678)
x = np.random.random(10)
y = 1.6*x + np.random.random(10)
'''

pd.plotting.register_matplotlib_converters()

file_name = '84-Site_12-BP-Solar.csv'

df = pd.read_csv(file_name)
try:
    df.columns = [col.decode('utf-8') for col in df.columns]
except AttributeError:
    pass  # Python 3 strings are already unicode literals
df = df.rename(columns = {
    u'12 BP Solar - Active Power (kW)':'power',
    u'12 BP Solar - Wind Speed (m/s)': 'wind',
    u'12 BP Solar - Weather Temperature Celsius (\xb0C)': 'Tamb',
    u'12 BP Solar - Global Horizontal Radiation (W/m\xb2)': 'ghi',
    u'12 BP Solar - Diffuse Horizontal Radiation (W/m\xb2)': 'dhi'
})


# Specify the Metadata
meta = {"latitude": -23.762028,
        "longitude": 133.874886,
        "timezone": 'Australia/North',
        "tempco": -0.005,
        "azimuth": 0,
        "tilt": 20,
        "pdc": 5100.0,
        "temp_model": 'open_rack_cell_polymerback'}


df.index = pd.to_datetime(df.Timestamp)
# TZ is required for irradiance transposition
df.index = df.index.tz_localize(meta['timezone'], ambiguous = 'infer')

# Explicitly trim the dates so that runs of this example notebook
# are comparable when the sourec dataset has been downloaded at different times
df = df['2008-11-11':'2017-05-15']

# Chage power from kilowatts to watts
df['power'] = df.power * 1000.0
# There is some missing data, but we can infer the frequency from the first several data points
freq = pd.infer_freq(df.index[:10])

# And then set the frequency of the dataframe
df = df.resample(freq).median()

# Calculate energy yield in Wh
df['energy'] = df.power * pd.to_timedelta(df.power.index.freq).total_seconds()/(3600.0)

y = df['energy'].to_numpy()
# y = df.index.to_numpy()
x = (df.index - df.index[0])/np.timedelta64(1, 'm')

mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])

print("slope: %f    intercept: %f" % (slope, intercept))
print("R-squared: %f" % r_value**2)

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(df.index, df.energy, 'o', label='original data', alpha = 0.01)
ax.plot(df.index, intercept + slope*x, 'r', label='fitted line')
ax.set_ylim(0,500)
fig.autofmt_xdate()
plt.legend()
plt.show()
