import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from moepy import lowess
from sklearn.metrics import mean_squared_error

i = 1
filename = "01-0" + str(i) + "-2021.csv"
covid_data = pd.read_csv(filename)
'''
print(covid_data.FIPS[648:3922])
covid_data.FIPS[648:3922].to_csv("filename.csv", mode='a', index=False)
covid_data.Admin2[648:3922].to_csv("filename.csv", mode='a', index=False)
covid_data.Confirmed[648:3922].to_csv("filename.csv", mode='a', index=False)
covid_data.Deaths[648:3922].to_csv("filename.csv", mode='a', index=False)
'''
# Write only selected columns to csv file
header = ["FIPS", "Admin2", "Confirmed", "Deaths"]
covid_data[648:3922].to_csv("filename.csv", columns=header, index=False)  # , mode='a',
print(covid_data.FIPS[648:3922])

# Read the next data file
# i = 2
filename = "01-0" + str(i+1) + "-2021.csv"
covid_data2 = pd.read_csv(filename)
print(covid_data2.FIPS[648:3922])

# Adding data as column to CSV
df_new = pd.read_csv("filename.csv")
print(df_new)
#          age state  point
# name
# Alice     24    NY     64
# Bob       42    CA     92
# Charlie   18    CA     70

df_new.insert(loc=4, column='FIPS_' + str(i), value=covid_data2.FIPS[648:3922].tolist())
df_new.insert(loc=5, column='Admin2_' + str(i), value=covid_data2.Admin2[648:3922].tolist())
df_new.insert(loc=6, column='Confirmed' + str(i), value=covid_data2.Confirmed[648:3922].tolist())
df_new.insert(loc=7, column='Deaths_' + str(i), value=covid_data2.Deaths[648:3922].tolist())


# df_new.FIPS2[0:3274] = covid_data2.FIPS[648:3922]
# df_new["FIPS2"]["Deaths2"] = df_new.FIPS
# df_new["Admin22"] = df_new.Admin2
# df_new[["FIPS2"], ["Admin22"], ["Confirmed2"], ["Deaths2"]] = [[df_new.FIPS], [df_new.Admin2], [df_new.Confirmed],
# [df_new.Deaths]]
print(df_new)
#          age state  point   new_col
# name
# Alice     24    NY     64  new data
# Bob       42    CA     92  new data
# Charlie   18    CA     70  new data

df_new.to_csv("filename.csv", index=False)
