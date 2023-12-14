import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted

# citi_bikes = pd.read_csv("data/citibike.csv")

# citi_bikes = pd.DataFrame(pd.read_csv("data/citibike.csv"))
covid_level = pd.read_csv("data/covidlevel.csv", encoding='latin-1')

pd.set_option('display.max_columns', None)  # force pandas to display any/all number of columns.
# pd.set_option('display.max_columns', 6)  # force pandas to display 6 columns.
# pd.set_option('display.max_rows', None)  # force pandas to display any/all number of rows.

# X_train, X_test, y_train, y_test = train_test_split(citi_bikes.tripduration, citi_bikes.bikeid,
#                                                    random_state=1)

'''
# print(citi_bikes.shape)  # show shape of dataset including number of data points and features
print("Shape of citibike data: {}".format(citi_bikes.shape))  # show shape of dataset including number of data points
# and features
# print("citibike.keys(): \n{}".format(citi_bikes.keys()))  # print all feature names of dataset
print("Feature names: \n{}".format(citi_bikes.keys()))  # print all feature names of dataset

# print(X_train.shape)
# print(X_test.shape)

print(citi_bikes[0:5])  # Print 5 first row of dataset
print(citi_bikes.tripduration[0:5])  # Print 5 first row of feature "tripduration" of dataset
print(citi_bikes[["tripduration", "bikeid"]][0:5])  # Print 5 first row feature "tripduration" and "bikeid" of dataset

print(citi_bikes.starttime[0:5])
print(citi_bikes.head())  # Print first 5 row (head) of dataset
print(citi_bikes.tail())  # Print last 5 row (tail) of dataset

print(citi_bikes.isnull().any())  # check whether there were any rows with nulls
print(citi_bikes[citi_bikes["birth year"].isnull()].head())  # Print first 5 row (head) with nulls of dataset
print(citi_bikes[citi_bikes["birth year"].isnull()].tail())  # Print last 5 row (tail) with nulls of dataset
'''
'''
print("Shape of covid level data: {}".format(covid_level.shape))  # show shape of dataset including number of data
# points and features
print("Feature names: \n{}".format(covid_level.keys()))  # print all feature names of dataset

covid_level_remove = covid_level.dropna()  # remove/drop the rows where at least one element is missing.
print("Shape of covid level data remove: {}".format(covid_level_remove.shape))  # show shape of dataset including
# number of data points and features

print(covid_level.head())  # Print first 5 row (head) of dataset
print(covid_level.tail())  # Print last 5 row (tail) of dataset

print(covid_level.isnull().any())  # check whether there were any rows with nulls
# print(covid_level[covid_level["county_population"].isnull()].head())  # Print first 5 row (head) with
# nulls of dataset
# print(covid_level[covid_level["covid_hospital_admissions_per_100k"].isnull()].tail())  # Print last 5 row (tail) with
# nulls of dataset
# covid_level_short = covid_level.drop("covid-19_community_level", axis=1)
'''
'''
print(covid_level.loc[covid_level['county_population'].isnull(), 'health_service_area_population'].unique())
print("County with empty county_population: \n{}".format(covid_level['county'][covid_level['county_population']
                                                         .isnull()].unique()))
print()

print("County with empty health_service_area_population: \n{}".format(covid_level['county']
                                                                      [covid_level['health_service_area_population']
                                                                      .isnull()].unique()))
print()

print("County with empty covid_inpatient_bed_utilization: \n{}".format(covid_level['county']
                                                                       [covid_level['covid_inpatient_bed_utilization']
                                                                       .isnull()].unique()))
print()

print("County with empty covid_hospital_admissions_per_100k: \n{}".
      format(covid_level['county'][covid_level['covid_hospital_admissions_per_100k'].isnull()].unique()))
print()

print("County with empty covid-19_community_level: \n{}".format(covid_level['county']
                                                                [covid_level['covid-19_community_level'].isnull()]
                                                                .unique()))
print()

covid_level_remove = covid_level.dropna()
covid_level_remove = covid_level.dropna(subset=['county_population', 'covid_hospital_admissions_per_100k', 'covid_cases_per_100k'])
covid_level_short = covid_level_remove.drop(["county_fips", "state", "health_service_area_number",
                                             "health_service_area", "health_service_area_population",
                                             "covid_inpatient_bed_utilization", "covid_hospital_admissions_per_100k",
                                             "covid-19_community_level", "date_updated"], axis=1)  # drop some
# features info
'''
# print(covid_level_short)

# covid_level_short_remove = covid_level_short.dropna()  # remove/drop the rows where at least one element is missing.
'''
# print(covid_level_short.covid_cases_per_100k.sort_values(ascending=False, inplace=False))
print(covid_level_short.sort_values(by='covid_cases_per_100k', ascending=False)[0:30])  # Sort Descending the dataset by
# colum covid_cases_per_100k

# print(covid_level_short.describe())  # show statistics of the dataset
print("\n")

print(covid_level_short.sort_values(by='covid_cases_per_100k', ascending=False).groupby('county'))  # groupby()
# method of DataFrames, passing the name of the desired key column
print("\n")

print(covid_level_short.sort_values(by='covid_cases_per_100k', ascending=False).groupby('county').sum())
print("\n")
'''
'''
covid_level_split = covid_level_short.groupby('county').split()
print(covid_level_split[0:30])
print("\n")
'''
'''
covid_level_sum = covid_level_short.groupby('county').sum()
print(covid_level_sum[0:30])
print("\n")
'''
'''
covid_level_combine = covid_level_short.groupby('county').combine()
print(covid_level_combine[0:30])
print("\n")
'''

# print(covid_level_sum.groupby('county')['covid_cases_per_100k'].median())
# covid_level_sum.to_csv('data/ch05_07.csv')  # Save dataset back to *.csv
# print("\n")

''' Matplotlib graph
plt.style.use('_mpl-gallery')

# make the data
# np.random.seed(3)
x = covid_level_sum.county_population
y = covid_level_sum.covid_cases_per_100k
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors)   # , vmin=0, vmax=100)

# ax.set(xlim=(0, 200000), xticks=np.arange(1, 200000),
#       ylim=(0, 200000), yticks=np.arange(1, 200000))

plt.show()
'''
'''
covid_level_removeNaN = covid_level.dropna()
print(covid_level_removeNaN.isnull().any())  # check whether there were any rows with nulls

# covid_level_sum = covid_level_removeNaN.groupby('county').sum()
# print(covid_level_sum)
print(covid_level_removeNaN.pivot_table('covid_cases_per_100k',
                                        index='county', columns='covid-19_community_level'))
# covid_level_removeNaN.to_csv('data/ch05_07.csv')
hospital_number = pd.cut(covid_level_removeNaN['covid_hospital_admissions_per_100k'], [0, 10, 30])

print(covid_level_removeNaN.pivot_table('covid_cases_per_100k',
                                        ['county', hospital_number], 'covid-19_community_level')[0:10])
'''
'''
# print(covid_level)
covid_level_remove = covid_level.dropna(subset=['county_population', 'covid_hospital_admissions_per_100k',
                                                'covid_cases_per_100k'])
covid_level_short = covid_level_remove.drop(["county_fips", "state", "health_service_area_number",
                                             "health_service_area", "health_service_area_population",
                                             "covid_inpatient_bed_utilization"], axis=1)  # drop some
# features info "covid-19_community_level",,
#                                              "date_updated"
# covid_level_sum = covid_level_short.groupby('county').sum()
# print(covid_level_short)
# print(covid_level_short.loc[(covid_level_short['covid_hospital_admissions_per_100k'] > 0) &
#                            (covid_level_short['date_updated'] == "2/24/2022")])
covid_level_day = covid_level_short.loc[(covid_level_short['covid_hospital_admissions_per_100k'] > 0) &
                                        (covid_level_short['date_updated'] == "2/24/2022")]
print(covid_level_day)
# covid_level_day.to_csv('data/ch05_07.csv')
# sns.set() # use Seaborn styles
covid_level_day_sum = covid_level_day.groupby('county').sum()
print(covid_level_day_sum)
# covid_level_day_sum.to_csv('data/ch05_07.csv')
'''
'''
covid_level_day_sum = pd.read_csv("data/ch05.csv", encoding='latin-1')[0:50]
# print(covid_level_day_sum)
covid_level_day_sum_short = covid_level_day_sum.pivot_table('covid_cases_per_100k',
                                                            index='covid_hospital_admissions_per_100k',
                                                            columns='covid-19_community_level', aggfunc='sum')
# print(covid_level_day_sum)
print(covid_level_day_sum_short)
print(covid_level_day_sum_short.isnull().any())
covid_level_day_sum.pivot_table('covid_cases_per_100k', index='covid_hospital_admissions_per_100k',
                                columns='covid-19_community_level', aggfunc='sum').plot()
plt.ylabel('covid_cases_per_100k')
# ax.set_xticks(numpy.arange(0, 1, 0.1))
# ax.set_yticks(numpy.arange(0, 1., 0.1))

plt.grid()
plt.show()
'''

covid_level_sum = covid_level.groupby('date_updated').sum()
covid_level_short = covid_level_sum.drop(["county", "county_fips", "state", "county_population",
                                          "health_service_area_number", "health_service_area_population",
                                          "health_service_area",
                                          "covid-19_community_level"],
                                         axis=1)  # drop some features "covid_inpatient_bed_utilization",
# print(covid_level_short)
covid_level_short.to_csv('data/ch05.csv')
covid_level_date = pd.read_csv("data/ch05.csv", encoding='latin-1')
# print(covid_level_date.sort_values(by="date_updated", key=lambda x: np.argsort(index_natsorted(
# covid_level_date["date_updated"])))[0:20])

# checking datatype
# print(type(covid_level_date.date_updated[0]))

# convert to date
covid_level_date['date_updated'] = pd.to_datetime(covid_level_date['date_updated'])

# verify datatype
# print(type(covid_level_date.date_updated[0]))

# print(covid_level_date.sort_values(by='date_updated'))
# covid_level_date.sort_values(by='date_updated').to_csv('data/ch05.csv')
covid_level_date_short = covid_level_date.sort_values(by='date_updated')
print(covid_level_date_short)

# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO
year = covid_level_date_short.date_updated
population_by_continent = {
    'covid_inpatient_bed_utilization': covid_level_date_short.covid_inpatient_bed_utilization,
    'covid_hospital_admissions_per_100k': covid_level_date_short.covid_hospital_admissions_per_100k,
    'covid_cases_per_100k': covid_level_date_short.covid_cases_per_100k,
}

fig, ax = plt.subplots()
ax.stackplot(year, population_by_continent.values(),
             labels=population_by_continent.keys(), alpha=0.8)
ax.legend(loc='upper right', reverse=True)
ax.set_title('Covid Level in US')
ax.set_xlabel('Time Stamp')
ax.set_ylabel('Number of cases')
# plt.grid()
# plt.plot(range(200), '--bo', label='line with marker')
plt.show()
