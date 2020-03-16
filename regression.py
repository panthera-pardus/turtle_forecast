#%% Build a linear regression and a Xcat/XGBoost model and compare the two

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import calendar
from sklearn.linear_model import LinearRegression

os.chdir("/Users/panthera_pardus/Documents/ds_projects/turtle_forcast")

#%% Read data
# Since the goal is to forcast for 2019, let us separate the sets of 2017 and 2018
data = pd.read_csv("data/train.csv")
data["dt_caught"] = pd.to_datetime(data.Date_TimeCaught)
validation_set = data.loc[data.dt_caught > "2017-01-01"]
train_set = data.loc[data.dt_caught < "2017-01-01"]

validation_set.to_csv("validation_set_2017_2018.csv")
