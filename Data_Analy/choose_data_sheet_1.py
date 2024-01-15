import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Choose data sheet 1
# 51640 48311 21201 53033 6087 15007
# 3057, 2861, 1134, 3161, 241, 577
# Read data
covid_Galax_51640 = pd.read_csv("JH_data/Hopkin_data/Galax_51640.0.csv")
covid_Maricopa_4013 = pd.read_csv("JH_data/Hopkin_data/McMullen_48311.0.csv")
covid_Cook_17031 = pd.read_csv("JH_data/Hopkin_data/Robertson_21201.0.csv")

covid_Delaware_39041 = pd.read_csv("JH_data/Hopkin_data/King_53033.0.csv")
covid_Stafford_51179 = pd.read_csv("JH_data/Hopkin_data/Santa Cruz_6087.0.csv")
covid_Tompkins_36109 = pd.read_csv("JH_data/Hopkin_data/Kauai_15007.0.csv")

# print(covid_Maricopa_4013.dates[12:1155])

# Multiple lines using pyplot
# red dashes, blue squares and green triangles

plt.plot(covid_Galax_51640.dates[12:1155], covid_Galax_51640.deaths_by_pop[12:1155], 'r--',
         covid_Maricopa_4013.dates[12:1155], covid_Maricopa_4013.deaths_by_pop[12:1155], 'b-',
         covid_Cook_17031.dates[12:1155], covid_Cook_17031.deaths_by_pop[12:1155], 'g-.',
         covid_Delaware_39041.dates[12:1155], covid_Delaware_39041.deaths_by_pop[12:1155], 'k--',
         covid_Stafford_51179.dates[12:1155], covid_Stafford_51179.deaths_by_pop[12:1155], 'm-.',
         covid_Tompkins_36109.dates[12:1155], covid_Tompkins_36109.deaths_by_pop[12:1155], 'y-')

plt.xlabel("Date")
plt.ylabel("Deaths per 100k population")
plt.legend(['Galax_51640', 'McMullen_48311', 'Robertson_21201', 'King_53033', 'Santa Cruz_6087',
            'Kauai_15007'], ncol=1, loc='upper left')

plt.title('Covid Deaths by Pop in the US', fontsize=20, fontname='Times New Roman')

ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::200])
# ax.set_yticks(ax.get_yticks()[::100])
sns.set()
plt.grid()
plt.show()


