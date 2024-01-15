import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Choose data sheet 2
# 6037 4013 17031 39041 51179 36109
# 215, 110, 642, 2144, 3128, 1957

# Read data
covid_Los_Angeles_6037 = pd.read_csv("JH_data/Hopkin_data/Los Angeles_6037.0.csv")
covid_Maricopa_4013 = pd.read_csv("JH_data/Hopkin_data/Maricopa_4013.0.csv")
covid_Cook_17031 = pd.read_csv("JH_data/Hopkin_data/Cook_17031.0.csv")

covid_Delaware_39041 = pd.read_csv("JH_data/Hopkin_data/Delaware_39041.0.csv")
covid_Stafford_51179 = pd.read_csv("JH_data/Hopkin_data/Stafford_51179.0.csv")
covid_Tompkins_36109 = pd.read_csv("JH_data/Hopkin_data/Tompkins_36109.0.csv")

# print(covid_Maricopa_4013.dates[12:1155])

# Multiple lines using pyplot
# red dashes, blue squares and green triangles

plt.plot(covid_Los_Angeles_6037.dates[12:1155], covid_Los_Angeles_6037.deaths_by_pop[12:1155], 'r--',
         covid_Maricopa_4013.dates[12:1155], covid_Maricopa_4013.deaths_by_pop[12:1155], 'b-',
         covid_Cook_17031.dates[12:1155], covid_Cook_17031.deaths_by_pop[12:1155], 'g-.',
         covid_Delaware_39041.dates[12:1155], covid_Delaware_39041.deaths_by_pop[12:1155], 'k--',
         covid_Stafford_51179.dates[12:1155], covid_Stafford_51179.deaths_by_pop[12:1155], 'm-.',
         covid_Tompkins_36109.dates[12:1155], covid_Tompkins_36109.deaths_by_pop[12:1155], 'y-')

plt.xlabel("Date")
plt.ylabel("Deaths per 100k population")
plt.legend(['Los_Angeles_6037', 'Maricopa_4013', 'Cook_17031', 'Delaware_39041', 'Stafford_51179',
            'Tompkins_36109'], ncol=1, loc='upper left')

plt.title('Covid Deaths by Pop in the US', fontsize=20, fontname='Times New Roman')

ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::200])
# ax.set_yticks(ax.get_yticks()[::100])
sns.set()
plt.grid()
plt.show()


