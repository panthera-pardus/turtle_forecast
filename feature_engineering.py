# Goal is to prepare the train.csv for modelling - here we can create a def/class
# to apply to the different dataframes.
# We want the dataframe of the following shape given the exploration :
# target : number_capture
# features : site_id, quarter, month, year, day_of_week, number_unique, likely_site, m_f_ratio

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import calendar

os.chdir("/Users/panthera_pardus/Documents/ds_projects/turtle_forcast")

#%% Read data
# Since the goal is to forcast for 2019, let us separate the sets of 2017 and 2018
data = pd.read_csv("data/train.csv")
data["dt_caught"] = pd.to_datetime(data["Date_TimeCaught"])
data.columns
# Target : number_capture_day
df_target = data.groupby(["CaptureSite", "Date_TimeCaught"]).\
count()["Rescue_ID"].\
reset_index()

df_target["Site_Date"] = df_target["CaptureSite"] + "_" + df_target["Date_TimeCaught"]
df_target["Site_Date"] = df_target["Site_Date"].str.replace("-", "")
df_target["target"] = df_target["Rescue_ID"]

df_target = df_target[["Site_Date", "CaptureSite", "Date_TimeCaught", "target"]]

# Time dimension variables : quarter, month, year, day_of_week
data["quarter"] = data.dt_caught.dt.quarter
data["month"] = data.dt_caught.dt.month
data["day_of_week"] = data.dt_caught.dt.dayofweek

df_time = data[["CaptureSite", "Date_TimeCaught", "quarter", "month", "day_of_week"]]
df_time["Site_Date"] = df_time["CaptureSite"] + "_" + df_time["Date_TimeCaught"]
df_time["Site_Date"] = df_time["Site_Date"].str.replace("-", "")

df_time = df_time[["Site_Date", "quarter", "month", "day_of_week"]]

# Site dimension variables : number_unique_fishermen, likely_site, m_f_ratio
data.groupby("CaptureSite").\
count()
