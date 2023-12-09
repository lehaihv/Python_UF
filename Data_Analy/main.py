import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

citi_bikes = pd.read_csv("data/citibike.csv")

X_train, X_test, y_train, y_test = train_test_split(citi_bikes.tripduration, citi_bikes.bikeid,
                                                    random_state=1)

# print(citi_bikes.shape)  # show shape of dataset including number of data points and features
print("Shape of citibike data: {}".format(citi_bikes.shape))  # show shape of dataset including number of data points
# and features
# print("citibike.keys(): \n{}".format(citi_bikes.keys()))  # print all feature names of dataset
print("Feature names: \n{}".format(citi_bikes.keys()))  # print all feature names of dataset


print(X_train.shape)
print(X_test.shape)
print(citi_bikes[0:5])  # Print 5 first row of dataset
print(citi_bikes.tripduration[0:5])  # Print 5 first row of feature "tripduration" of dataset
print(citi_bikes[["tripduration", "bikeid"]][0:5])  # Print 5 first row feature "tripduration" and "bikeid" of dataset


# scaler = MinMaxScaler()
# scaler.fit(X_train)








'''
plt.semilogy(citi_bikes.starttime, citi_bikes.bikeid)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
plt.show()
'''
