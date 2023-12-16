import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns

covid_level = pd.read_csv("data/covidlevel.csv", encoding='latin-1')

print("Shape of covid level data: {}".format(covid_level.shape))  # show shape of dataset including number of data
# points and features
print("Feature names: \n{}".format(covid_level.keys()))  # print all feature names of dataset
