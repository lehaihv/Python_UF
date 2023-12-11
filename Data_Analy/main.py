import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# citi_bikes = pd.read_csv("data/citibike.csv")

# citi_bikes = pd.DataFrame(pd.read_csv("data/citibike.csv"))
covid_level = pd.read_csv("data/covidlevel.csv", encoding='latin-1')

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
'''
covid_level_remove = covid_level.dropna()
covid_level_short = covid_level_remove.drop(["county_fips", "state", "health_service_area_number",
                                             "health_service_area", "health_service_area_population",
                                             "covid_inpatient_bed_utilization", "covid_hospital_admissions_per_100k",
                                             "covid-19_community_level", "date_updated"], axis=1)  # drop some
# features info

# print(covid_level_short)

# covid_level_short_remove = covid_level_short.dropna()  # remove/drop the rows where at least one element is missing.

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
covid_level_split = covid_level_short.groupby('county').split()
print(covid_level_split[0:30])
print("\n")
'''

covid_level_sum = covid_level_short.groupby('county').sum()
print(covid_level_sum[0:30])
print("\n")

'''
covid_level_combine = covid_level_short.groupby('county').combine()
print(covid_level_combine[0:30])
print("\n")
'''

# print(covid_level_sum.groupby('county')['covid_cases_per_100k'].median())
covid_level_sum.to_csv('data/ch05_07.csv')
print("\n")





















