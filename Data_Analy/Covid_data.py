import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from moepy import lowess

'''
covid_data = pd.read_csv("data/cdcdataset.csv")

print("Shape of covid level data: {}".format(covid_data.shape))  # show shape of dataset including number of data
# points and features
print("Feature names: \n{}".format(covid_data.keys()))  # print all feature names of dataset

# create new covid_data_short data_frame to store interested data
covid_data_short = pd.DataFrame(columns=['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin', 'pcr_target',
                                         'pcr_gene_target_agg', 'county_names'])
# copy interested data from dataset to new data_frame
covid_data_short[['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin', 'pcr_target', 'pcr_gene_target_agg',
                  'county_names']] = covid_data[['sample_collect_date', 'wwtp_name', 'pcr_target_flowpop_lin',
                                                 'pcr_target', 'pcr_gene_target_agg', 'county_names']]

print(covid_data_short[0:3])

covid_data_short.to_csv('data/coviddata.csv')


covid_data = pd.read_csv("data/coviddata.csv")
print(covid_data)

# create new covid_data_short data_frame to store interested data
covid_data_short = pd.DataFrame(columns=['sample_collect_date', 'pcr_target_flowpop_lin'])
# copy interested data from dataset to new data_frame
covid_data_short[['sample_collect_date', 'pcr_target_flowpop_lin']] = covid_data[['sample_collect_date',
                                                                                  'pcr_target_flowpop_lin']]

covid_data_short.to_csv('data/coviddata1.csv')


covid_data = pd.read_csv("data/coviddata1.csv")
print(covid_data)

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


pd.set_option('display.max_columns', None)  # force pandas to display any/all number of columns.
'''
'''
covid_data = pd.read_csv("data/coviddatafinal.csv")
print(covid_data)

print(covid_data.groupby('wwtp_name').sum())

covid_data_wtp_name = covid_data.groupby('wwtp_name').sum()
covid_data_wtp_name.to_csv('data/coviddatafinal1.csv')


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

integrate LaTeX expressions (insert mathematical expressions within the chart)
plt.text(covid_level_date_short.date_updated[42], 750000, r'$y = x^2$', fontsize=20, bbox={'facecolor': 'yellow',
                                                                                           'alpha': 0.2})


sns.set()
plt.show()

'''

'''
covid_data = pd.read_excel("data/covid_29019.xlsx")
print(covid_data)
# covid_data_remove = covid_data.dropna()  # remove/drop the rows where at least one element is missing.
# covid_data_remove.to_excel('data/covid_29019.xlsx')


# convert to date
covid_data['date'] = pd.to_datetime(covid_data['date'])
#covid_data_date_short = covid_data_remove.sort_values(by='date')
#print(covid_data_date_short)

# covid_data['date'] = pd.to_datetime(covid_data['date']).dt.date  # Only keep date, remove time
# print(covid_data['date'])

# covid_data_date_short.to_excel('data/covid_29019.xlsx')
'''
'''
# Sample date&time variable
date_time_var = covid_data['date'][0]

# Convert to Pandas datetime object
date_time_obj = pd.to_datetime(date_time_var)

# Remove time from the datetime object
date_var = date_time_obj.date()

# Convert the date object to a string
date_str = date_var.strftime('%Y-%m-%d')
print(date_time_var)
print(date_str)
'''

covid_data = pd.read_csv("data/covid_29019.csv")
print(covid_data)
# convert to date
# covid_data['date'] = pd.to_datetime(covid_data['date'])
# covid_data_date_short = covid_data.sort_values(by='date')
# print(covid_data_date_short)
# covid_data_date_short.to_csv('data/covid_29019_Bo.csv')

# Data generation
x = np.linspace(0, 5, num=74)
y = np.array(covid_data.ww_data[0:74]).reshape(-1)  # np.sin(x) + (np.random.normal(size=len(x)))/10

# Model fitting
lowess_model = lowess.Lowess()
lowess_model.fit(x, y)

# Model prediction
x_pred = np.linspace(0, 5, 513)
y_pred = lowess_model.predict(x_pred)

# Plotting
plt.plot(x_pred, y_pred, '--', label='LOWESS', color='k', zorder=3)
plt.scatter(x, y, label='Noisy Virus Concentration', color='C1', s=5, zorder=1)
plt.legend(frameon=False)
plt.show()
print(y)
print(y_pred)
covid_data.concentration = pd.Series(y_pred)
pd.DataFrame(y_pred).to_csv('data/covid_29019_2.csv')


'''
covid_data = pd.read_excel("data/Community_covid.xlsx")
# print(covid_data)
covid_data['date_updated'] = pd.to_datetime(covid_data['date_updated']).dt.date  # Only keep date, remove time
# print(covid_data['date_updated'])
# covid_data_date_short = covid_data.sort_values(by='date_updated')
# print(covid_data_date_short)
covid_data.to_excel('data/Community_covid1.xlsx')
'''

