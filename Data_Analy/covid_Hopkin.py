import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from moepy import lowess
from sklearn.metrics import mean_squared_error

# Step 1: Read data for each region and save to temp1
covid_data = pd.read_csv("JH_data/covid19_deaths_US.csv")
# print(covid_data.head())
# print(covid_data.iloc[0])
# 6037 4013 17031 36047 36081
# 48215 36119 18089 5007 17201
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in [110, 123, 215, 642, 729, 776, 1925, 1943, 1963, 2807]:  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)
    filename = "JH_data/" + covid_data.Admin2[x] + "_" + str(covid_data.FIPS[x]) + ".csv"
    # filename = "JH_data/" + covid_data.Admin2[x] + "_" + str(covid_data.FIPS[x]) + ".xlsx"

    # Step 2: Read temp1, prepare data to calculate R, add 'deaths_by_pop' column and save to temp2
    covid_data_temp = pd.read_csv("JH_data/temp1.csv", names=['dates', 'I', 'deaths_by_pop'])
    # print(covid_data.head())
    covid_data_temp.to_csv("JH_data/temp2.csv", index=False)

    # Step 3: Read temp2, format dates, calculate deaths by pop, save to regions files
    covid_data_temp = pd.read_csv("JH_data/temp2.csv")
    # print(covid_data.dates)
    # convert to date
    covid_data_temp.dates[12:1155] = pd.to_datetime(covid_data_temp.dates[12:1155]).dt.date
    # Only keep date, remove time
    # print(covid_data.dates[12:16])
    # print(covid_data.I[11])
    if float(covid_data_temp.I[11]) > 0:
        for i in range(12, 1155):
            covid_data_temp.deaths_by_pop[i] = float(covid_data_temp.I[i]) * 100000/float(covid_data_temp.I[11])

        # print(filename)
        covid_data_temp.to_csv(filename, index=False)
        # covid_data_temp.to_excel(filename, index=False)
print("done")


