import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns

'''
covid_data = pd.read_csv("data/cdcdataset.csv")

print("Shape of covid level data: {}".format(covid_data.shape))  # show shape of dataset including number of data
# points and features
print("Feature names: \n{}".format(covid_data.keys()))  # print all feature names of dataset

covid_data_short = pd.DataFrame(columns=['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin', 'pcr_target',
                                         'pcr_gene_target_agg'])
covid_data_short[['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin', 'pcr_target', 'pcr_gene_target_agg']] \
    = covid_data[['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin', 'pcr_target', 'pcr_gene_target_agg']]

print(covid_data_short[0:3])

covid_data_short.to_csv('data/coviddata.csv')

covid_data = pd.read_csv("data/coviddata.csv")
print(covid_data)
covid_data_remove = covid_data.dropna()  # remove/drop the rows where at least one element is missing.
print(covid_data_remove.isnull().any())  # check whether there were any rows with nulls
print(covid_data_remove)
# convert to date
covid_data_remove['sample_collect_date'] = pd.to_datetime(covid_data_remove['sample_collect_date'])
covid_data_date_short = covid_data_remove.sort_values(by='sample_collect_date')
print(covid_data_date_short)

# covid_data_sum = covid_data_date_short.groupby('sample_collect_date').sum()
# print(covid_data_sum)
covid_data_date_short.to_csv('data/coviddatafinal.csv')
'''

covid_data = pd.read_csv("data/coviddatafinal.csv")
print(covid_data)





