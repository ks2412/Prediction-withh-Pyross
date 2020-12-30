import pandas as df
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

D_data = df.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/confirmed.csv')
cols = D_data.keys()
confirmed = D_data.loc[:, cols[1]:cols[-1]]
print(confirmed)
Dates = confirmed.keys()
india_cases = []

for i in Dates:
    confirmed_sum = confirmed[i].sum()
    india_cases.append(confirmed_sum)


def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i] - data[i - 1])
    return d


india_daily = daily_increase(india_cases)
# print(india_daily)
start_day = np.array([i for i in range(len(Dates))]).reshape(-1, 1)
india_cases = np.array(india_cases).reshape(-1, 1)
# print(india_cases)

future = 55
future_forcast = np.array([i for i in range(len(Dates) + future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-55]
# print(future_forcast)

start = '03/14/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(adjusted_dates)
poly.fit(X_poly, india_cases)
lin2 = LinearRegression()
lin2.fit(X_poly, india_cases)
poly_con = lin2.predict(poly.fit_transform(adjusted_dates))
poly_pred = lin2.predict(poly.fit_transform(future_forcast))

print(poly_con.sum())
print(poly_pred.sum())

plt.xticks(rotation=70)
plt.scatter(adjusted_dates, india_cases, color='blue')
plt.plot(adjusted_dates, poly_con, color='red')
plt.title('# of COVID-19 Cases Over Time')
plt.xlabel('Days Since 14/03/2020')
plt.ylabel('# of Daily Cases')

plt.show()

plt.xticks(np.arange(0, 200, 7), (
    '14 Mar', '21 Mar', '28 Mar', '4 Apr', '11 Apr', '18 Apr', '25 Apr', '2 Apr', '9 May', '16 May', '23 May', '30 May',
    '6 Jun', '13 Jun', '20 Jun'), rotation=60)
plt.scatter(adjusted_dates, india_cases)
plt.plot(future_forcast_dates, poly_pred,)
plt.title('# of COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('# of Daily Cases')

plt.show()
