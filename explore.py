# Read and explore data
#%% Import packages

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
os.chdir("/Users/andour/Google Drive/projects/turtle_forecast")
#%% Read data
# Since the goal is to forcast for 2019, let us separate the sets of 2017 and 2018
data = pd.read_csv("train.csv")
data["dt_caught"] = pd.to_datetime(data.Date_TimeCaught)
validation_set = data.loc[data.dt_caught > "2017-01-01"]
train_set = data.loc[data.dt_caught < "2017-01-01"]

validation_set.to_csv("validation_set_2017_2018.csv")

# How many turtles were caught per week?
train_set = train_set.sort_values(by = "dt_caught")

train_set["week_capture"] = train_set.dt_caught.dt.weekofyear
train_set["year_capture"] = train_set.dt_caught.dt.year

plot_data = train_set.groupby(["year_capture",
                        "week_capture"]).count()["Date_TimeCaught"].reset_index()

plot_data["Date_TimeCaught"].plot()


train_set.head()
plot_data
# How many turtles were caught per year?

year_count = train_set.\
groupby("year_capture").\
size().\
reset_index().\
rename(columns = {0 : "capture_count"})

sns.set_palette("Set2")
fig, ax = plt.subplots()
ax.set_title("Early turtle capture")
ax.plot("year_capture", "capture_count", data = year_count)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set(xlabel = "Year", ylabel = "Turtle count")
ax.grid()
# Clear increase in the yearly tutle captures - we can also look at the pct change :
year_count["capture_count"].pct_change() # volatility but increases are much higher than occasional decreases

# Is there seasonality in captures - probability of capturing a turtle per month
train_set["month_capture"] = train_set.dt_caught.dt.month
month_count = train_set.groupby("month_capture").\
size().\
to_frame().\
reset_index().\
rename({0 : "month_count"}, axis = 1)

month_count["total"] = month_count["month_count"].sum()
month_count["prob_month_capture"] = month_count["month_count"]/month_count["total"]

# Higher probabilities that turtles are caught in the last 3 months of the yearly
# Need a variable to capture this asymmetry

# Do some researchers catch more than others (per year)? 
# Are some fishers more likely to rescue? Moral hazard
# Are some species more likely to be captured?
# Are there many recaptures?
# Do some some turtles get rescued more?
# Are some sites more likely to find turtles?
# What variables should go in our model
train_set.head()
