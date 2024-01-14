import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from moepy import lowess
from sklearn.metrics import mean_squared_error

'''
covid_data = pd.read_csv("JH_data/covid19_deaths_US.csv")
print(covid_data)
# convert to date


# extract each row data, convert to columns and save to file
# filename = covid_data.Admin2[0] + ".csv"
print(filename)
# covid_data.iloc[0][11:1155].to_csv(filename)  # "filename.csv")
print(covid_data.iloc[0][11:1155])
# print(covid_data.iloc[0]['Population'][11:1155])
'''
covid_data = pd.read_csv("JH_data/Autauga.csv")
print(covid_data.date)
'''
for i in range(3):
    filename = "JH_data/" + covid_data.Admin2[i] + ".csv"
    print(filename)
    covid_data.iloc[[i]].to_csv(filename)  # "filename.csv")
'''
'''
covid_data['date'] = pd.to_datetime(covid_data['date'])
covid_data_date_short = covid_data.sort_values(by='date')
print(covid_data_date_short)
covid_data_date_short.to_csv('data/covid_29019_Bo.csv')
'''