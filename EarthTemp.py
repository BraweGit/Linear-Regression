import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as LinReg

# Exploration.
#df = pd.read_csv("E:/OneDrive/Documents/python/test/GlobalTemperatures.csv")
#print(df.head())
# df = df.ix[:,:2]
# print(df.describe())
# print(df.head())
# #print(df.describe())
# times = pd.DatetimeIndex(df["dt"])
# grouped = df.groupby([times.year]).mean()
# # plt.figure(figsize = (15, 5))
# # plt.plot(grouped['LandAverageTemperature'])

# Change features of the graph
# plt.title("Yearly Average Land Temperature 1750-2015")
# plt.xlabel("Year")
# plt.ylabel("Yearly Average Land Temperature")
# plt.show()

# #print(df[times.year == 1752])
# print(df[np.isnan(df["LandAverageTemperature"])])
# df["LandAverageTemperature"] = df["LandAverageTemperature"].fillna(method="ffill")
# print(df[times.year == 1752])

# Model.
df = pd.read_csv("E:/OneDrive/Documents/python/test/GlobalTemperatures.csv")
df = df.ix[:,:2]
times = pd.DatetimeIndex(df["dt"])
grouped = df.groupby([times.year]).mean()
x = grouped.index.values.reshape(-1,1)
y = grouped["LandAverageTemperature"].values

reg = LinReg()
reg.fit(x,y)
y_preds = reg.predict(x)
print('Accuracy: {}'.format(reg.score(x,y)))

plt.figure(figsize=(15,5))
plt.title('Linear Regression')
plt.scatter(x = x, y = y_preds)
plt.scatter(x = x, y = y, c = 'r')
plt.show()

print('Predicted temperature in 2050 will be {}'.format(reg.predict(2050)[0]))





