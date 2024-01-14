import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from sklearn.metrics import mean_squared_error

# 6037 4013 17031 36047 36081
# 48215 36119 18089 5007 17201
# Read data
covid_Maricopa_4013 = pd.read_csv("JH_data/Hopkin_data/Maricopa_4013.0.csv")
covid_Benton_5007 = pd.read_csv("JH_data/Hopkin_data/Benton_5007.0.csv")
covid_Los_Angeles_6037 = pd.read_csv("JH_data/Hopkin_data/Los Angeles_6037.0.csv")
covid_Cook_17031 = pd.read_csv("JH_data/Hopkin_data/Cook_17031.0.csv")
covid_Winnebago_17201 = pd.read_csv("JH_data/Hopkin_data/Winnebago_17201.0.csv")
covid_Lake_18089 = pd.read_csv("JH_data/Hopkin_data/Lake_18089.0.csv")
covid_Kings_36047 = pd.read_csv("JH_data/Hopkin_data/Kings_36047.0.csv")
covid_Queens_36081 = pd.read_csv("JH_data/Hopkin_data/Queens_36081.0.csv")
covid_Westchester_36119 = pd.read_csv("JH_data/Hopkin_data/Westchester_36119.0.csv")
covid_Hidalgo_48215 = pd.read_csv("JH_data/Hopkin_data/Hidalgo_48215.0.csv")

print(covid_Maricopa_4013.dates[12:1155])

# Multiple lines using pyplot
# red dashes, blue squares and green triangles
plt.plot(covid_Los_Angeles_6037.dates[12:1155], covid_Los_Angeles_6037.deaths_by_pop[12:1155], 'r--',
         covid_Maricopa_4013.dates[12:1155], covid_Maricopa_4013.deaths_by_pop[12:1155], 'b-',
         covid_Cook_17031.dates[12:1155], covid_Cook_17031.deaths_by_pop[12:1155], 'g-.',
         covid_Kings_36047.dates[12:1155], covid_Kings_36047.deaths_by_pop[12:1155], 'c-',
         covid_Queens_36081.dates[12:1155], covid_Queens_36081.deaths_by_pop[12:1155], 'm--',
         covid_Hidalgo_48215.dates[12:1155], covid_Hidalgo_48215.deaths_by_pop[12:1155], 'y-.',
         covid_Westchester_36119.dates[12:1155], covid_Westchester_36119.deaths_by_pop[12:1155], 'k-',
         covid_Lake_18089.dates[12:1155], covid_Lake_18089.deaths_by_pop[12:1155], 'b--',
         covid_Benton_5007.dates[12:1155], covid_Benton_5007.deaths_by_pop[12:1155], 'r-.',
         covid_Winnebago_17201.dates[12:1155], covid_Winnebago_17201.deaths_by_pop[12:1155], 'y-')
plt.xlabel("Date")
plt.ylabel("Deaths per 100k population")
plt.legend(['Los_Angeles_6037', 'Maricopa_4013', 'Cook_17031', 'Kings_36047', 'Queens_36081', 'Hidalgo_48215',
            'Westchester_36119', 'Lake_18089', 'Benton_5007', 'Winnebago_17201'],
           ncol=1, loc='upper left')

plt.title('Covid Deaths by Pop in the US', fontsize=20, fontname='Times New Roman')

ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::100])
sns.set()
plt.grid()
plt.show()

'''
# 10 simple subplots
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1)
fig.suptitle('Covid Deaths by Pop in the US')

ax1.plot(covid_Los_Angeles_6037.dates[12:1155], covid_Los_Angeles_6037.deaths_by_pop[12:1155], 'o-')
ax1.set_ylabel('Los_Angeles_6037')

ax2.plot(covid_Maricopa_4013.dates[12:1155], covid_Maricopa_4013.deaths_by_pop[12:1155], '.-')
ax2.set_ylabel('Maricopa_4013')

ax3.plot(covid_Cook_17031.dates[12:1155], covid_Cook_17031.deaths_by_pop[12:1155], 'x-')
ax3.set_ylabel('Cook_17031')

ax4.plot(covid_Kings_36047.dates[12:1155], covid_Kings_36047.deaths_by_pop[12:1155], 'o-')
ax4.set_ylabel('Kings_36047')

ax5.plot(covid_Queens_36081.dates[12:1155], covid_Queens_36081.deaths_by_pop[12:1155], '.-')
ax5.set_ylabel('Queens_36081')

# 48215 36119 18089 5007 17201
ax6.plot(covid_Hidalgo_48215.dates[12:1155], covid_Hidalgo_48215.deaths_by_pop[12:1155], 'x-')
ax6.set_ylabel('Hidalgo_48215')

ax7.plot(covid_Westchester_36119.dates[12:1155], covid_Westchester_36119.deaths_by_pop[12:1155], 'o-')
ax7.set_ylabel('Westchester_36119')

ax8.plot(covid_Lake_18089.dates[12:1155], covid_Lake_18089.deaths_by_pop[12:1155], '.-')
ax8.set_ylabel('Lake_18089')

ax9.plot(covid_Benton_5007.dates[12:1155], covid_Benton_5007.deaths_by_pop[12:1155], 'x-')
ax9.set_ylabel('Benton_5007')

ax10.plot(covid_Winnebago_17201.dates[12:1155], covid_Winnebago_17201.deaths_by_pop[12:1155], 'x-')
ax10.set_ylabel('Winnebago_17201')

ax10.set_xlabel('Date')
ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::200])

plt.show()
'''
'''  
# 3 simple subplots
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(5, 2)
fig.suptitle('Covid Deaths in the US')

ax1.plot(covid_level_date_short.date_updated, covid_level_date_short.covid_inpatient_bed_utilization, 'o-')
ax1.set_ylabel('covid_inpatient_bed_utilization')

ax2.plot(covid_level_date_short.date_updated, covid_level_date_short.covid_hospital_admissions_per_100k, '.-')
ax2.set_ylabel('covid_hospital_admissions_per_100k')

ax3.plot(covid_level_date_short.date_updated, covid_level_date_short.covid_cases_per_100k, 'x-')
ax3.set_ylabel('covid_cases_per_100k')
ax3.set_xlabel('Date')

plt.show()
'''
