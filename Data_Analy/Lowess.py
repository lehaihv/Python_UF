import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from natsort import index_natsorted
import seaborn as sns
from moepy import lowess
from sklearn.metrics import mean_squared_error

# covid_data = pd.read_excel("Risk/Risk_levels_county_6001.xlsx")
covid_data = pd.read_excel("Risk/Risk_levels_Virus_concentration_county_6001.xlsx")
# Lowess
# Data generation
x = np.linspace(0, 564, num=224)  # CC: 894 WW: 174 pp: 159
y = np.array(covid_data.pcr_target_flowpop_lin[0:224]).reshape(-1)  # cases_by_cdc_case_earliest_date

# Model fitting
lowess_model = lowess.Lowess()
# frac default value of 0.4, where the nearest 40% of the data-points to the local regression will be used
lowess_model.fit(x, y, frac=0.03, num_fits=100)  # , num_fits=25

# Model prediction
x_pred = np.linspace(0, 564, 564)  # CC: 894 WW: 1247 pp: 159
y_pred = lowess_model.predict(x_pred)

# Plotting
plt.plot(x_pred, y_pred, '--', label='LOWESS', color='k', zorder=3)
plt.scatter(x, y, label='Noisy CC data', color='C1', s=5, zorder=1)
plt.legend(frameon=False)
plt.show()

# print(y)
print(y_pred)
# covid_data.concentration = pd.Series(y_pred)
# pd.DataFrame(y_pred).to_csv('data/covid_29019_2.csv')
pd.DataFrame(y_pred).to_excel("Risk/template_concen.xlsx", sheet_name='Sheet_name_1')
