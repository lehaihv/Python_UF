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
# print(covid_data)
covid_data_remove = covid_data.dropna()  # remove/drop the rows where at least one element is missing.
# print(covid_data_remove.isnull().any())  # check whether there were any rows with nulls
# print(covid_data_remove)
# convert to date
covid_data_remove['sample_collect_date'] = pd.to_datetime(covid_data_remove['sample_collect_date'])
covid_data_date_short = covid_data_remove.sort_values(by='sample_collect_date')
print(covid_data_date_short)

# covid_data_sum = covid_data_date_short.groupby('sample_collect_date').sum()
# print(covid_data_sum)
covid_data_date_short.to_csv('data/coviddatafinal.csv')
'''

pd.set_option('display.max_columns', None)  # force pandas to display any/all number of columns.
# '''
covid_data = pd.read_csv("data/coviddatafinal.csv")
print(covid_data)

print(covid_data.groupby('wwtp_name').sum())

covid_data_wtp_name = covid_data.groupby('wwtp_name').sum()
covid_data_wtp_name.to_csv('data/coviddatafinal1.csv')
# '''

covid_data1 = pd.read_csv("data/coviddatafinal1.csv")[0:50]
print(covid_data1)

# Multiple lines using pyplot
# red dashes, blue squares and green triangles
plt.plot(covid_data1.wwtp_name, covid_data1.pcr_target_flowpop_lin, 'r--')

plt.xlabel("Date")
plt.ylabel("Number of cases")
plt.legend(['covid_inpatient_bed_utilization', 'covid_hospital_admissions_per_100k', 'covid_cases_per_100k'],
           ncol=1, loc='upper right')

plt.title('Covid Level in the US', fontsize=20, fontname='Times New Roman')

'''# integrate LaTeX expressions (insert mathematical expressions within the chart)
plt.text(covid_level_date_short.date_updated[42], 750000, r'$y = x^2$', fontsize=20, bbox={'facecolor': 'yellow',
                                                                                           'alpha': 0.2})
'''

sns.set()
plt.show()



