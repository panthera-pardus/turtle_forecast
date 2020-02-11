# Read and explore data
#%% Import packages

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import calendar
from sklearn.linear_model import LinearRegression
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
sns.set_palette("Set2")
plot_data["Date_TimeCaught"].plot()


train_set.head()
plot_data
# How many turtles were caught per year?

year_count = train_set.\
groupby("year_capture").\
size().\
reset_index().\
rename(columns = {0 : "capture_count"})


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
# Quick addition of month name for graph

month_count["month"] = month_count["month_capture"].apply(lambda x: calendar.month_abbr[x])
month_count.plot.bar(x = "month", y = "prob_month_capture")

# Higher probabilities that turtles are caught in the last 3 months of the yearly
# Need a variable to capture this asymmetry
# It seems that there are more catches at the end of the year

# Do some researchers catch more than others (per year)?
train_set.head()
researcher_capture = train_set.groupby("Researcher").\
size()

# There are very strong differences between researchers catches
# It may be interesting to know if there are more catpures on specific days and do these days correspond to researchers
# Perhaps logging captures under one or a few ids or there is more work on some days more than others?

# 1) Barchart to check if some days are more likely than others
train_set["day_capture"] = train_set.dt_caught.dt.dayofweek



day_capture = train_set.groupby("day_capture").\
size()

np.std(day_capture)
# The day of the week does not seem to have a big influence.
# However the difference can be approx 10% between Monday and Tuesday
# Is this due to researcher high count? In any case a variable for day might be useful

# 2) Regression on days vs researcher count?
research_day = train_set.groupby(["day_capture", "Researcher"]).\
size().\
reset_index().\
rename(columns = {0 : "capture_count"})

research_day.plot.scatter(x = "day_capture", y = "capture_count")

# From this graph, it seems that the day/researcher relation is not in itself a big influence on capture count
# New variable not needed

# Are some fishers more likely to rescue? Moral hazard
train_set.columns
fisher_count = train_set.groupby("Fisher").\
size().\
reset_index().\
rename({0 : "count"}, axis = 1)

fisher_count.describe()
# The fisher id contains outliers but generally the variable is "clean"

# Are some species more likely to be captured?

train_set.groupby("Species").\
size()

# Are there many recaptures? Do some some turtles get rescued more?
# train_set.groupby("T_Number").\
# size()
# Not really

# Are some sites more likely to find turtles?
train_set.groupby("CaptureSite").\
size()

# Yes there is a huge variance between sites and even males vs females - females tend to be captured more
# What variables should go in our model - cf feature engineering script
