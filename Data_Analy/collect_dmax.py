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

# Choose data sheet 1
# 51640 48311 21201 53033 6087 15007
#

# Choose data sheet 2
# 6037 4013 17031 39041 51179 36109
# 215, 110, 642, 2144, 3128, 1957


# Choose data sheet 3
# 2180 2188 48247 29165 32011 32029
#

# Choose data sheet 4
# 6037 12086 17031 48447 32011 31165
# 215, xxx, 642,

# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"
'''
for x in [215, 110, 642, 2144, 3128, 1957]:  # 3342 # Choose data sheet 1
for x in [215, 110, 642, 2144, 3128, 1957]:  # 3342 # Choose data sheet 2
for x in [215, 110, 642, 2144, 3128, 1957]:  # 3342 # Choose data sheet 3
for x in [215, 110, 642, 2144, 3128, 1957]:  # 3342 # Choose data sheet 4
'''
for x in [215, 110, 642, 2144, 3128, 1957]:  # 3342 # Choose data sheet 2

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
            covid_data_temp.deaths_by_pop[i] = float(covid_data_temp.I[i]) * 100000 / float(covid_data_temp.I[11])

        # print(filename)
        covid_data_temp.to_csv(filename, index=False)
        # covid_data_temp.to_excel(filename, index=False)
print("done")
