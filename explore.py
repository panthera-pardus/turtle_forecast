# Read and explore data
#%% Import packages

import pandas as pd
import numpy as np

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
