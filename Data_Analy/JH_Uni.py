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
filename = "JH_data/" + covid_data.Admin2[0] + ".csv"
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in range(500, 1000):  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)

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
        covid_data_temp.to_csv(filename, mode='a', index=False)

print("done")

filename = "JH_data/4.csv"
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in range(1500, 2000):  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)

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
        covid_data_temp.to_csv(filename, mode='a', index=False)

print("done")

filename = "JH_data/5.csv"
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in range(2000, 2500):  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)

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
        covid_data_temp.to_csv(filename, mode='a', index=False)

print("done")

filename = "JH_data/6.csv"
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in range(2500, 3000):  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)

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
        covid_data_temp.to_csv(filename, mode='a', index=False)

print("done")

filename = "JH_data/7.csv"
# filename = "JH_data/" + covid_data.Admin2[0] + ".xlsx"

for x in range(3000, 3342):  # 3342

    covid_data.iloc[x].to_csv("JH_data/temp1.csv", header=None)

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
        covid_data_temp.to_csv(filename, mode='a', index=False)

print("done")

# covid_data_date_short = covid_data_remove.sort_values(by='sample_collect_date')
'''
csv_input = pd.read_csv('input.csv')
csv_input['Berries'] = csv_input['Name']
csv_input.to_csv('output.csv', index=False)
print(covid_data.iloc[0][12:1155])
filename = "JH_data/" + covid_data.Admin2[0] + ".csv"
covid_data.iloc[0][11:1155].to_csv(filename, header=['color'])
'''
'''
# extract each row data, convert to columns and save to file
filename = "JH_data/" + covid_data.Admin2[0] + ".csv"
print(filename)
covid_data.iloc[0][11:1155].to_csv(filename)  # "filename.csv")
print(covid_data.iloc[0][11:1155])
# print(covid_data.iloc[0]['Population'][11:1155])

covid_data = pd.read_csv("JH_data/Autauga.csv")
print(covid_data.date)
'''
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
